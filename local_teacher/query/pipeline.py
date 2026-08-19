import re
import time
import logging
from typing import Iterator, Optional, Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from local_teacher.query.types import SemanticCacheProtocol
from langchain_core.retrievers import BaseRetriever
from langchain_community.tools import DuckDuckGoSearchRun

from local_teacher.metrics import QueryMetricsTracker
from local_teacher.query.optimizer import reescribir_consulta
from local_teacher.query.critic import evaluar_borrador, DecisionCritico
from local_teacher.query.graph_search import obtener_contexto_grafo
from local_teacher.query.reranker import recuperar_y_filtrar
from local_teacher.query.prompts import formatear_documentos, crear_cadena_tutor, MENSAJE_FALLBACK

_log = logging.getLogger(__name__)


class PipelineConsulta:
    """Orquestador del flujo RAG con caché, optimizador, grafos, reranking y supervisor."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        busqueda_web_alternativa: bool = False,
        cache_store: Optional[SemanticCacheProtocol] = None,
        usar_critico: bool = True,
        llm_critic: Optional[BaseChatModel] = None,
        kuzu_path: str = "./local_teacher_kuzu",
        llm_fast: Optional[BaseChatModel] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.llm_fast = llm_fast
        self.busqueda_web_alternativa = busqueda_web_alternativa
        self.cache_store = cache_store
        self.usar_critico = usar_critico
        self.llm_critic = llm_critic
        self.kuzu_path = kuzu_path
        self.herramienta_busqueda = (
            DuckDuckGoSearchRun() if busqueda_web_alternativa else None
        )
        self.progreso_callback = None

    def _progreso(
        self,
        paso: int,
        total: int = 4,
        mensaje: str = "",
        saltar_linea: bool = False,
    ) -> None:
        if self.progreso_callback:
            self.progreso_callback(paso, total, mensaje, saltar_linea)

    def ejecutar(
        self,
        consulta: str,
        chat_history: list = None,
        progreso_callback=None,
    ) -> Iterator[dict]:
        """Ejecuta el pipeline completo de consulta transmitiendo fragmentos."""
        self.progreso_callback = progreso_callback
        tracker = QueryMetricsTracker(
            consulta, self.llm, self.llm_critic, self.usar_critico
        )
        chat_history = chat_history or []

        # 1. Caché semántico L1
        if self.cache_store:
            self._progreso(0, 4, "Consultando Caché Semántico L1...")
            try:
                cache_results = self.cache_store.similarity_search_with_score(
                    consulta, k=1
                )
                if cache_results:
                    doc, score = cache_results[0]
                    if score > 0.95:
                        _log.info("Cache Hit en Redis! Score: %.3f", score)
                        self._progreso(
                            4,
                            4,
                            "¡Respuesta rápida desde Caché Semántico!",
                            saltar_linea=True,
                        )
                        respuesta_cacheada = doc.metadata.get(
                            "respuesta", doc.page_content
                        )
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
        consulta_estructurada = reescribir_consulta(self.llm_fast or self.llm, consulta, chat_history)
        consulta_optimizada = consulta_estructurada.consulta
        tracker.set_meta("optimized_query", consulta_optimizada)
        capitulo_filtro = consulta_estructurada.capitulo
        entidades_filtro = consulta_estructurada.entidades
        intencion_filtro = getattr(consulta_estructurada, "intencion", "conceptual")
        tracker.add_latency("Optimizacion_Reescritura", t_opt_start)

        # 3. Consulta y expansión con Grafo de Conocimiento
        self._progreso(1, 4, "Buscando en Grafo de Conocimiento...")
        t_grafo_start = time.time()
        contexto_grafo, palabras_clave_grafo = obtener_contexto_grafo(
            entidades_filtro, db_path=self.kuzu_path
        )
        if palabras_clave_grafo:
            expansion = " ".join(palabras_clave_grafo[:5])
            consulta_optimizada = f"{consulta_optimizada} {expansion}"
            tracker.set_meta("optimized_query", consulta_optimizada)
        tracker.add_latency("Recuperacion_Grafo", t_grafo_start)

        # 4. Recuperación híbrida y reranking
        self._progreso(1, 4, "Recuperando documentos...")
        t_vec_start = time.time()
        docs = recuperar_y_filtrar(
            self.retriever, consulta_optimizada, capitulo_filtro, entidades_filtro
        )
        tracker.add_latency("Recuperacion_Vectorial", t_vec_start)

        if not docs:
            self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
            yield {
                "answer": MENSAJE_FALLBACK
            }
            tracker.finish_and_log("REJECTED_SAFE")
            yield {"context_docs": []}
            return

        yield from self._generar_con_supervisor(
            consulta,
            docs,
            contexto_grafo,
            intencion_filtro,
            chat_history,
            consulta_optimizada,
            tracker,
        )

    def _generar_con_supervisor(
        self,
        consulta: str,
        docs: list[Document],
        contexto_grafo: str,
        intencion_filtro: str,
        chat_history: list,
        consulta_optimizada: str,
        tracker: QueryMetricsTracker,
    ) -> Iterator[dict]:
        """Genera y supervisa respuestas en un ciclo con reintentos si el crítico detecta inconsistencias."""
        cadena_tutor = crear_cadena_tutor(
            self.llm, self.busqueda_web_alternativa, intencion_filtro
        )
        texto_contexto = formatear_documentos(docs)
        tracker.set_meta("num_docs", len(docs))
        tracker.set_meta("context_size_chars", len(texto_contexto))
        mensaje_feedback = ""

        for intento in range(1, 4):
            tracker.set_meta("attempts", intento)
            tracker.init_latency(f"Generacion_LLM_Intento_{intento}")
            if self.usar_critico:
                tracker.init_latency(f"Critico_Intento_{intento}")
                self._progreso(
                    2,
                    4,
                    f"Generando respuesta (Intento {intento}/3)...",
                    saltar_linea=True,
                )

            # Respaldo web en el intento 2 si está activado
            if intento == 2 and self.busqueda_web_alternativa and self.herramienta_busqueda:
                self._progreso(
                    2,
                    4,
                    "Consultando información adicional en la web...",
                    saltar_linea=True,
                )
                try:
                    import concurrent.futures
                    t_web_start = time.time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exec_web:
                        futuro = exec_web.submit(self.herramienta_busqueda.invoke, consulta_optimizada)
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

            t_gen_start = time.time()
            args_invoke = {
                "chat_history": chat_history,
                "input": consulta,
                "context": texto_contexto,
                "graph_context": contexto_grafo,
                "feedback": mensaje_feedback,
            }

            if self.usar_critico:
                self._progreso(3, 4, "Tutor local generando y evaluando (esto tomará unos segundos)...", saltar_linea=True)
                yield {"answer": f"\n*[Tutor local generando y evaluando internamente (Intento {intento}/3)...]*\n"}
                respuesta = cadena_tutor.invoke(args_invoke)
                borrador = (
                    respuesta.content
                    if hasattr(respuesta, "content")
                    else str(respuesta)
                )
            else:
                self._progreso(3, 4, "Generando respuesta...", saltar_linea=True)
                borrador = ""
                for token_chunk in cadena_tutor.stream(args_invoke):
                    content = (
                        token_chunk.content
                        if hasattr(token_chunk, "content")
                        else str(token_chunk)
                    )
                    borrador += content
                    yield {"answer": content}

            tracker.add_latency(f"Generacion_LLM_Intento_{intento}", t_gen_start)

            borrador_limpio = re.sub(
                r"<think>.*?</think>", "", borrador, flags=re.DOTALL
            ).strip()

            # Verificación de admisión honesta de ignorancia
            if (
                "No poseo información suficiente" in borrador_limpio
                or "no está cubierto" in borrador_limpio.lower()
            ):
                if self.usar_critico:
                    self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                    yield {"answer": borrador}
                tracker.finish_and_log("REJECTED_SAFE")
                yield {"context_docs": []}
                return

            if "REQUIRE_WEB_SEARCH" in borrador_limpio and len(borrador_limpio) < 100:
                _log.info("El tutor solicitó búsqueda web (REQUIRE_WEB_SEARCH).")
                decision_critico = DecisionCritico.RECHAZADO.value
            elif any(
                w in borrador_limpio.lower()[:50]
                for w in ["lo siento", "no puedo", "hubo un error"]
            ):
                decision_critico = DecisionCritico.RECHAZADO.value
            else:
                if self.usar_critico:
                    self._progreso(
                        3,
                        4,
                        "Crítico evaluando precisión y alucinaciones...",
                        saltar_linea=True,
                    )
                    t_crit_start = time.time()
                    decision_critico = evaluar_borrador(
                        self.llm_critic or self.llm, texto_contexto, borrador_limpio
                    )
                    tracker.add_latency(f"Critico_Intento_{intento}", t_crit_start)
                else:
                    decision_critico = DecisionCritico.APROBADO.value

            es_aprobado = decision_critico == DecisionCritico.APROBADO.value

            if es_aprobado:
                if self.usar_critico:
                    self._progreso(4, 4, "¡Respuesta Aprobada!", saltar_linea=True)
                if self.cache_store:
                    try:
                        self.cache_store.add_texts(
                            texts=[consulta], metadatas=[{"respuesta": borrador}]
                        )
                    except Exception as e:
                        _log.warning("Error al guardar en caché semántico: %s", e)
                tracker.finish_and_log("APPROVED")

                if self.usar_critico:
                    for i in range(0, len(borrador), 15):
                        yield {"answer": borrador[i:i+15]}
                        time.sleep(0.01)

                yield {"context_docs": docs}
                return
            else:
                mensaje_feedback = "El revisor indicó que tu respuesta incluía afirmaciones no respaldadas. Por favor, sé más estricto."
                yield {
                    "answer": "\n\n[!] El supervisor local detectó imprecisiones. Reintentando corregir la respuesta...\n\n"
                }
                continue

        self._progreso(4, 4, "Agotados los intentos.", saltar_linea=True)
        yield {
            "answer": MENSAJE_FALLBACK
        }
        tracker.finish_and_log("REJECTED_SAFE")
        yield {"context_docs": []}
