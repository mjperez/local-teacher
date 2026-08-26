import re
import time
import logging
import concurrent.futures
from typing import Iterator, Optional, Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from local_teacher.query.types import SemanticCacheProtocol
from langchain_core.retrievers import BaseRetriever
try:
    from langchain_duckduckgo import DuckDuckGoSearchRun
except ImportError:
    from langchain_community.tools import DuckDuckGoSearchRun

from local_teacher.metrics import QueryMetricsTracker
from local_teacher.query.optimizer import reescribir_consulta
from local_teacher.query.critic import evaluar_borrador, DecisionCritico
from local_teacher.query.graph_search import obtener_contexto_grafo
from local_teacher.query.reranker import recuperar_y_filtrar
from local_teacher.query.prompts import formatear_documentos, crear_cadena_tutor, MENSAJE_FALLBACK

_log = logging.getLogger(__name__)

class SupervisorLLM:
    """Clase auxiliar para manejar el bucle de generación y crítica."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        usar_critico: bool,
        llm_critic: Optional[BaseChatModel],
        busqueda_web_alternativa: bool,
        herramienta_busqueda: Any,
        web_filter: str,
        progreso_callback: Any
    ):
        self.llm = llm
        self.usar_critico = usar_critico
        self.llm_critic = llm_critic or llm
        self.busqueda_web_alternativa = busqueda_web_alternativa
        self.herramienta_busqueda = herramienta_busqueda
        self.web_filter = web_filter
        self.progreso_callback = progreso_callback

    def _progreso(self, paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False):
        if self.progreso_callback:
            self.progreso_callback(paso, total, mensaje, saltar_linea)
            
    def _ejecutar_busqueda_web(self, consulta_optimizada: str, texto_contexto: str, tracker: QueryMetricsTracker) -> tuple[str, str]:
        self._progreso(2, 4, "Consultando información adicional en la web...")
        mensaje_feedback = ""
        try:
            consulta_busqueda = consulta_optimizada
            if self.web_filter:
                consulta_busqueda = f"{consulta_optimizada} {self.web_filter}"

            t_web_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exec_web:
                futuro = exec_web.submit(self.herramienta_busqueda.invoke, consulta_busqueda)
                resultados_web = futuro.result(timeout=10)
            tracker.add_latency("Generacion_BusquedaWeb", t_web_start)
            tracker.set_meta("web_search_used", True)
            resultados_web_recortados = str(resultados_web)[:2000]
            texto_contexto += f"\n\n--- RESULTADOS DE BÚSQUEDA WEB ---\n{resultados_web_recortados}"
            tracker.set_meta("context_size_chars", len(texto_contexto))
        except concurrent.futures.TimeoutError:
            _log.warning("Timeout en la búsqueda web tras 10 segundos.")
            mensaje_feedback += "\n[!] La búsqueda web automática tomó demasiado tiempo. Responde basándote solo en el contexto previo o admite que no posees información."
        except Exception as e:
            _log.warning("Falló la búsqueda web: %s", e)
            mensaje_feedback += "\n[!] La búsqueda web automática falló. Responde basándote solo en el contexto previo o admite que no posees información."
        return texto_contexto, mensaje_feedback

    def generar(
        self,
        consulta: str,
        documentos: list[Document],
        contexto_grafo: str,
        intencion_filtro: str,
        historial_chat: list,
        consulta_optimizada: str,
        tracker: QueryMetricsTracker,
        cache: Optional[SemanticCacheProtocol]
    ) -> Iterator[dict]:
        cadena_tutor = crear_cadena_tutor(
            self.llm, self.busqueda_web_alternativa, intencion_filtro
        )
        texto_contexto = formatear_documentos(documentos)
        tracker.set_meta("num_docs", len(documentos))
        tracker.set_meta("context_size_chars", len(texto_contexto))
        mensaje_feedback = ""

        for intento in range(1, 4):
            tracker.set_meta("attempts", intento)
            tracker.init_latency(f"Generacion_LLM_Intento_{intento}")
            if self.usar_critico:
                tracker.init_latency(f"Critico_Intento_{intento}")
                self._progreso(2, 4, f"Generando respuesta (Intento {intento}/3)...")

            if intento == 2 and self.busqueda_web_alternativa and self.herramienta_busqueda:
                texto_contexto, fb = self._ejecutar_busqueda_web(consulta_optimizada, texto_contexto, tracker)
                mensaje_feedback += fb

            t_gen_start = time.time()
            args_invoke = {
                "chat_history": historial_chat,
                "input": consulta,
                "context": texto_contexto,
                "graph_context": contexto_grafo,
                "feedback": mensaje_feedback,
            }

            if self.usar_critico:
                self._progreso(3, 4, "Tutor local generando y evaluando (esto tomará unos segundos)...")
                respuesta = cadena_tutor.invoke(args_invoke)
                borrador = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
            else:
                self._progreso(3, 4, "Generando respuesta...", saltar_linea=True)
                borrador = ""
                for token_chunk in cadena_tutor.stream(args_invoke):
                    content = token_chunk.content if hasattr(token_chunk, "content") else str(token_chunk)
                    borrador += content
                    yield {"answer": content}

            tracker.add_latency(f"Generacion_LLM_Intento_{intento}", t_gen_start)

            borrador_limpio = re.sub(r"<think>.*?</think>", "", borrador, flags=re.DOTALL).strip()

            if "NO_INFO_EN_CONTEXTO" in borrador_limpio:
                if self.usar_critico:
                    self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                    yield {"answer": MENSAJE_FALLBACK}
                tracker.finish_and_log("REJECTED_SAFE")
                yield {"context_docs": []}
                return

            if "REQUIRE_WEB_SEARCH" in borrador_limpio and len(borrador_limpio) < 100:
                _log.info("El tutor solicitó búsqueda web (REQUIRE_WEB_SEARCH).")
                decision_critico = DecisionCritico.RECHAZADO.value
            elif any(w in borrador_limpio.lower()[:50] for w in ["lo siento", "no puedo", "hubo un error"]):
                decision_critico = DecisionCritico.RECHAZADO.value
            else:
                if self.usar_critico:
                    self._progreso(3, 4, "Supervisor evaluando precisión y alucinaciones...")
                    t_crit_start = time.time()
                    decision_critico = evaluar_borrador(self.llm_critic, texto_contexto, borrador_limpio)
                    tracker.add_latency(f"Critico_Intento_{intento}", t_crit_start)
                else:
                    decision_critico = DecisionCritico.APROBADO.value

            es_aprobado = decision_critico == DecisionCritico.APROBADO.value

            if es_aprobado:
                if self.usar_critico:
                    self._progreso(4, 4, "¡Respuesta Aprobada!", saltar_linea=True)
                if cache:
                    try:
                        cache.add_texts(texts=[consulta], metadatas=[{"respuesta": borrador}])
                    except Exception as e:
                        _log.warning("Error al guardar en caché semántico: %s", e)
                tracker.finish_and_log("APPROVED")

                if self.usar_critico:
                    for i in range(0, len(borrador), 15):
                        yield {"answer": borrador[i:i+15]}

                yield {"context_docs": documentos}
                return
            else:
                mensaje_feedback = "El revisor indicó que tu respuesta incluía afirmaciones no respaldadas. Por favor, sé más estricto."
                yield {"answer": "\n\n[!] El supervisor local detectó imprecisiones. Reintentando corregir la respuesta...\n\n"}
                continue

        self._progreso(4, 4, "Agotados los intentos.", saltar_linea=True)
        yield {"answer": MENSAJE_FALLBACK}
        tracker.finish_and_log("REJECTED_SAFE")
        yield {"context_docs": []}


class PipelineConsulta:
    """Orquestador del flujo RAG con caché, optimizador, grafos, reranking y supervisor."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        busqueda_web_alternativa: bool = False,
        web_filter: str = "",
        cache_store: Optional[SemanticCacheProtocol] = None,
        usar_critico: bool = True,
        llm_critic: Optional[BaseChatModel] = None,
        llm_fast: Optional[BaseChatModel] = None,
    ):
        self.recuperador = retriever
        self.llm = llm
        self.llm_fast = llm_fast
        self.busqueda_web_alternativa = busqueda_web_alternativa
        self.web_filter = web_filter
        self.cache = cache_store
        self.usar_critico = usar_critico
        self.llm_critic = llm_critic
        if self.usar_critico and not self.llm_critic:
            self.llm_critic = self.llm
        self.herramienta_busqueda = DuckDuckGoSearchRun() if busqueda_web_alternativa else None
        self.progreso_callback = None

    def _progreso(self, paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False) -> None:
        if self.progreso_callback:
            self.progreso_callback(paso, total, mensaje, saltar_linea)

    def ejecutar(
        self,
        consulta: str,
        historial_chat: list = None,
        progreso_callback=None,
    ) -> Iterator[dict]:
        if not consulta or not consulta.strip():
            return

        self.progreso_callback = progreso_callback
        tracker = QueryMetricsTracker(consulta, self.llm, self.llm_critic, self.usar_critico)
        historial_chat = historial_chat or []

        # 1. Caché semántico L1
        if self.cache:
            self._progreso(0, 4, "Consultando Caché Semántico L1...")
            try:
                cache_results = self.cache.similarity_search_with_score(consulta, k=1)
                if cache_results:
                    doc, score = cache_results[0]
                    if score > 0.95:
                        _log.info("Cache Hit en Redis! Score: %.3f", score)
                        self._progreso(4, 4, "¡Respuesta rápida desde Caché Semántico!", saltar_linea=True)
                        respuesta_cacheada = doc.metadata.get("respuesta", doc.page_content)
                        tracker.set_meta("cache_hit", True)
                        tracker.finish_and_log("CACHE_HIT")
                        yield {"answer": respuesta_cacheada}
                        yield {"context_docs": []}
                        return
            except Exception as e:
                _log.warning("Error al consultar caché semántico: %s", e)

        # 2. Optimización y reformulación de pregunta
        self._progreso(0, 4, "Optimizando pregunta...")
        t_opt_start = time.time()
        consulta_estructurada = reescribir_consulta(self.llm_fast or self.llm, consulta, historial_chat)
        consulta_optimizada = consulta_estructurada.consulta
        tracker.set_meta("optimized_query", consulta_optimizada)
        capitulo_filtro = consulta_estructurada.capitulo
        entidades_filtro = consulta_estructurada.entidades
        intencion_filtro = getattr(consulta_estructurada, "intencion", "conceptual")
        tracker.add_latency("Optimizacion_Reescritura", t_opt_start)

        # 3. Consulta y enriquecimiento concurrente
        self._progreso(1, 4, "Recuperando información del grafo y documentos vectoriales...")
        t_vec_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futuro_grafo = executor.submit(obtener_contexto_grafo, entidades_filtro)
            futuro_docs = executor.submit(
                recuperar_y_filtrar, self.recuperador, consulta_optimizada, capitulo_filtro, entidades_filtro
            )
            
            contexto_grafo, _ = futuro_grafo.result()
            documentos = futuro_docs.result()
            
        tracker.add_latency("Recuperacion_Grafo_y_Vectorial", t_vec_start)

        if not documentos:
            self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
            yield {"answer": MENSAJE_FALLBACK}
            tracker.finish_and_log("REJECTED_SAFE")
            yield {"context_docs": []}
            return

        supervisor = SupervisorLLM(
            llm=self.llm,
            usar_critico=self.usar_critico,
            llm_critic=self.llm_critic,
            busqueda_web_alternativa=self.busqueda_web_alternativa,
            herramienta_busqueda=self.herramienta_busqueda,
            web_filter=self.web_filter,
            progreso_callback=self.progreso_callback
        )
        
        yield from supervisor.generar(
            consulta,
            documentos,
            contexto_grafo,
            intencion_filtro,
            historial_chat,
            consulta_optimizada,
            tracker,
            self.cache
        )
