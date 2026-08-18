import re
import time
import json
import logging
import concurrent.futures
from typing import Any, Optional, Iterator

from local_teacher.query.optimizer import reescribir_consulta
from local_teacher.query.critic import evaluar_borrador

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from pathlib import Path
from langchain_community.tools import DuckDuckGoSearchRun
from flashrank import Ranker, RerankRequest

import kuzu
from local_teacher.metrics import QueryMetricsTracker

_log = logging.getLogger(__name__)

# Reranker instance
_ranker = None


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


class ConsultaReescrita(BaseModel):
    consulta: str = Field(
        description="La consulta optimizada traducida al inglés (si aplica) para búsqueda vectorial."
    )
    capitulo: Optional[str] = Field(
        description="El número de capítulo si la pregunta es exclusivamente sobre un capítulo, de lo contrario null."
    )
    entidades: list[str] = Field(
        description="Lista de conceptos núcleo o nombres propios extraídos de la pregunta."
    )


def _formatear_documentos(docs: list[Document]) -> str:
    """Formatea los documentos inyectando su metadata para el LLM."""
    res = []
    for i, d in enumerate(docs, 1):
        meta = []
        if d.metadata.get("curso"):
            meta.append(f"Curso: {d.metadata['curso']}")
        if d.metadata.get("fuente"):
            nombre = d.metadata.get("titulo") or Path(d.metadata["fuente"]).name
            meta.append(f"Archivo: {nombre}")
        if d.metadata.get("pagina") is not None:
            meta.append(f"Página: {d.metadata['pagina']}")

        sec = d.metadata.get("ruta_seccion")
        if not sec:
            enc = [
                d.metadata.get("encabezado_1"),
                d.metadata.get("encabezado_2"),
                d.metadata.get("encabezado_3"),
            ]
            sec = " > ".join([e for e in enc if e])
        if sec:
            meta.append(f"Sección: {sec}")

        meta_str = " | ".join(meta)
        res.append(
            f"--- FUENTE [{i}] ---\nMetadata: [{meta_str}]\nContenido:\n{d.page_content}\n--- FIN FUENTE [{i}] ---"
        )

    return "\n\n".join(res)


def _obtener_contexto_grafo(entidades_filtro: list[str]) -> tuple[str, list[str]]:
    """2. Consulta de Grafo de Conocimiento (GraphRAG) y Expansión de Búsqueda (Kùzu)"""
    contexto_grafo = "(No se detectaron entidades o no hay grafo disponible)"
    palabras_clave_grafo = []

    if not entidades_filtro:
        return contexto_grafo, palabras_clave_grafo

    db_path = "./local_teacher_kuzu"
    if not Path(db_path).exists():
        return contexto_grafo, palabras_clave_grafo

    try:
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        
        # Test si las tablas existen
        try:
            conn.execute("MATCH (n:Entity) RETURN n LIMIT 1")
        except RuntimeError:
            return contexto_grafo, palabras_clave_grafo
            
        conexiones = []
        nodos_encontrados = set()

        for entidad in entidades_filtro:
            # Buscar entidades que coincidan
            res = conn.execute("MATCH (n:Entity) WHERE n.name CONTAINS $ent RETURN n.name", parameters={"ent": entidad})
            while res.has_next():
                nodo = res.get_next()[0]
                nodos_encontrados.add(nodo)
                
        for nodo in nodos_encontrados:
            palabras_clave_grafo.append(nodo)
            
            # Relaciones salientes
            out_res = conn.execute("MATCH (a:Entity {name: $n})-[r:Rel]->(b:Entity) RETURN a.name, r.type, b.name", parameters={"n": nodo})
            while out_res.has_next():
                origen, rel, destino = out_res.get_next()
                palabras_clave_grafo.append(destino)
                conexiones.append(f"- {origen} [{rel}] {destino}")
                
            # Relaciones entrantes
            in_res = conn.execute("MATCH (a:Entity)-[r:Rel]->(b:Entity {name: $n}) RETURN a.name, r.type, b.name", parameters={"n": nodo})
            while in_res.has_next():
                origen, rel, destino = in_res.get_next()
                palabras_clave_grafo.append(origen)
                conexiones.append(f"- {origen} [{rel}] {destino}")

        conexiones = list(set(conexiones))
        if conexiones:
            contexto_grafo = "\n".join(conexiones)
            _log.info(
                f"[*] Se inyectaron {len(conexiones)} conexiones del Grafo de Conocimiento (Kùzu) al contexto."
            )

        return contexto_grafo, list(set(palabras_clave_grafo))
        
    except Exception as e:
        _log.error(f"Error consultando KùzuDB: {e}")
        return contexto_grafo, palabras_clave_grafo


def _recuperar_y_filtrar(
    retriever: BaseRetriever,
    consulta_optimizada: str,
    capitulo_filtro: Optional[str],
    entidades_filtro: list[str] = None,
) -> list[Document]:
    """3. Recuperación Híbrida y Reranking con FlashRank"""
    docs = []
    vistos_id = set()

    if entidades_filtro:
        for entidad in entidades_filtro:
            retriever.search_kwargs = {"k": 15}
            res = retriever.invoke(entidad)
            for d in res:
                content_hash = hash(d.page_content)
                if content_hash not in vistos_id:
                    docs.append(d)
                    vistos_id.add(content_hash)

    retriever.search_kwargs = {"k": 25}
    res_completa = retriever.invoke(consulta_optimizada)
    for d in res_completa:
        content_hash = hash(d.page_content)
        if content_hash not in vistos_id:
            docs.append(d)
            vistos_id.add(content_hash)

    if capitulo_filtro:
        docs_filtrados = []
        prefix = f"{capitulo_filtro}."
        for d in docs:
            ruta = d.metadata.get("ruta_seccion", "")
            if ruta.strip().startswith(prefix) or f" {prefix}" in ruta:
                docs_filtrados.append(d)
        if docs_filtrados:
            docs = docs_filtrados

    if not docs:
        return []

    # FlashRank Reranking
    ranker = get_ranker()
    passages = []
    for idx, d in enumerate(docs):
        passages.append({"id": idx, "text": d.page_content, "meta": d.metadata})

    rerankrequest = RerankRequest(query=consulta_optimizada, passages=passages)
    results = ranker.rerank(rerankrequest)

    final_docs = []
    # Filtrar por score mínimo para evitar pasar documentos irrelevantes al LLM
    MIN_RERANK_SCORE = 0.1
    for res in results[:10]:
        if res["score"] < MIN_RERANK_SCORE:
            continue
        meta = res["meta"]
        text = res["text"]
        doc = Document(page_content=text, metadata=meta)
        doc.metadata["rerank_score"] = res["score"]
        final_docs.append(doc)

    # Si todos los docs tienen score bajo, devolver los 3 mejores de todas formas
    if not final_docs and results:
        for res in results[:3]:
            meta = res["meta"]
            text = res["text"]
            doc = Document(page_content=text, metadata=meta)
            doc.metadata["rerank_score"] = res["score"]
            final_docs.append(doc)

    return final_docs


def _crear_cadena_tutor(llm: BaseChatModel, busqueda_web_alternativa: bool, intencion: str = "conceptual"):
    system_base = (
        "REGLA PRINCIPAL: Responde ÚNICAMENTE con información que aparezca en el contexto recuperado. "
        "Si la información no está en el contexto, responde: 'Este tema no está cubierto en el material cargado.' "
        "NUNCA uses tu conocimiento interno para complementar o enriquecer la respuesta.\n\n"
        "Eres un tutor educativo. "
    )
    
    if intencion == "ejercicio":
        system_base += "El alumno necesita ayuda con un EJERCICIO O PROBLEMA PRÁCTICO. NO le des la solución directa bajo ninguna circunstancia. Dale pistas progresivas (scaffolding), guíalo paso a paso y hazle preguntas reflexivas para que descubra la solución por sí mismo. "
    elif intencion == "aclaracion":
        system_base += "El alumno tiene una duda rápida. Responde de forma directa, concisa y sin rodeos. "
    else:
        system_base += "Adopta un tono pedagógico y formativo: desglosa los problemas teóricos, explica el porqué de las cosas, y fomenta la comprensión profunda. Puedes hacer una pregunta de control al final para asegurar que el alumno entendió. "
        
    system_base += (
        "\nINSTRUCCIONES:\n"
        "1. Basa tu respuesta EXCLUSIVAMENTE en el contexto recuperado. Puedes parafrasear y reformular para enseñar mejor, pero toda afirmación debe provenir del material.\n"
        "2. Si te preguntan de qué trata el texto o piden un resumen general, sintetiza los temas principales basados únicamente en el contexto recuperado.\n"
        "3. Incluye Citas en Línea (ej. '...el motor se enciende [2].') al final de CADA afirmación usando el número de fuente correspondiente.\n"
        "4. Si la respuesta NO ESTÁ en el contexto, responde: 'Este tema no está cubierto en el material cargado.' NO inventes.\n"
        "5. Si no sabes, responde SÓLO con: REQUIRE_WEB_SEARCH (solo aplicable si busqueda_web_alternativa=True)."
    )
    
    prompt_tutor = ChatPromptTemplate.from_messages(
        [
            ("system", system_base),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Utiliza los siguientes documentos recuperados para responder a mi pregunta.\n"
                "<documentos>\n{context}\n</documentos>\n\n"
                "A continuación hay relaciones cruzadas de conceptos extraídas del grafo de conocimiento que podrían ser útiles:\n"
                "<relaciones_grafo>\n{graph_context}\n</relaciones_grafo>\n\n"
                "Mi pregunta es:\n{input}\n\n"
                "{feedback}",
            ),
        ]
    )
    return prompt_tutor | llm


class PipelineConsulta:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        busqueda_web_alternativa: bool = False,
        cache_store: BaseRetriever = None,
        usar_critico: bool = True,
        llm_critic: BaseChatModel = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.busqueda_web_alternativa = busqueda_web_alternativa
        self.cache_store = cache_store
        self.usar_critico = usar_critico
        self.llm_critic = llm_critic
        self.herramienta_busqueda = DuckDuckGoSearchRun() if busqueda_web_alternativa else None

    def _progreso(self, paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False):
        if self.progreso_callback:
            self.progreso_callback(paso, total, mensaje, saltar_linea)

    def ejecutar(self, consulta: str, chat_history: list = None, progreso_callback=None) -> Iterator[dict]:
        self.progreso_callback = progreso_callback
        tracker = QueryMetricsTracker(consulta, self.llm, self.llm_critic, self.usar_critico)
        chat_history = chat_history or []

        # 1. Caché
        if self.cache_store:
            self._progreso(0, 4, "Consultando Caché Semántico L1...")
            try:
                cache_results = self.cache_store.similarity_search_with_score(consulta, k=1)
                if cache_results:
                    doc, score = cache_results[0]
                    if score > 0.95:
                        _log.info(f"Cache Hit! Score: {score:.3f}")
                        self._progreso(4, 4, "¡Respuesta rápida desde Caché Semántico!", saltar_linea=True)
                        respuesta_cacheada = doc.metadata.get("respuesta", doc.page_content)
                        tracker.set_meta("cache_hit", True)
                        tracker.finish_and_log("CACHE_HIT")
                        yield {"answer": respuesta_cacheada}
                        yield {"context_docs": []}
                        return
            except Exception as e:
                _log.warning(f"Error consultando caché semántico: {e}")

        # 2. Optimización
        self._progreso(0, 4, "Optimizando pregunta...")
        t_opt_start = time.time()
        consulta_estructurada = reescribir_consulta(self.llm, consulta, chat_history)
        consulta_optimizada = consulta_estructurada.consulta
        tracker.set_meta("optimized_query", consulta_optimizada)
        capitulo_filtro = consulta_estructurada.capitulo
        entidades_filtro = consulta_estructurada.entidades
        intencion_filtro = getattr(consulta_estructurada, "intencion", "conceptual")
        tracker.add_latency("Optimizacion_Reescritura", t_opt_start)

        # 3. Grafo
        self._progreso(1, 4, "Buscando en Grafo de Conocimiento...")
        t_grafo_start = time.time()
        contexto_grafo, palabras_clave_grafo = _obtener_contexto_grafo(entidades_filtro)
        if palabras_clave_grafo:
            expansion = " ".join(palabras_clave_grafo[:5])
            consulta_optimizada = consulta_optimizada + " " + expansion
            tracker.set_meta("optimized_query", consulta_optimizada)
        tracker.add_latency("Recuperacion_Grafo", t_grafo_start)

        # 4. Recuperación Híbrida
        self._progreso(1, 4, "Recuperando documentos...")
        t_vec_start = time.time()
        docs = _recuperar_y_filtrar(self.retriever, consulta_optimizada, capitulo_filtro, entidades_filtro)
        tracker.add_latency("Recuperacion_Vectorial", t_vec_start)

        if not docs:
            self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
            yield {"answer": "No he encontrado información sobre este tema en el material cargado. Al tratarse de un tutor basado estrictamente en el contenido provisto, no puedo responder esta pregunta."}
            tracker.finish_and_log("REJECTED_SAFE")
            yield {"context_docs": []}
            return

        yield from self._generar_con_supervisor(consulta, docs, contexto_grafo, intencion_filtro, chat_history, consulta_optimizada, tracker)

    def _generar_con_supervisor(self, consulta, docs, contexto_grafo, intencion_filtro, chat_history, consulta_optimizada, tracker):
        cadena_tutor = _crear_cadena_tutor(self.llm, self.busqueda_web_alternativa, intencion_filtro)
        texto_contexto = _formatear_documentos(docs)
        tracker.set_meta("num_docs", len(docs))
        tracker.set_meta("context_size_chars", len(texto_contexto))
        mensaje_feedback = ""

        for intento in range(1, 4):
            tracker.set_meta("attempts", intento)
            tracker.init_latency(f"Generacion_LLM_Intento_{intento}")
            if self.usar_critico:
                tracker.init_latency(f"Critico_Intento_{intento}")
                self._progreso(2, 4, f"Generando respuesta (Intento {intento}/3)...", saltar_linea=True)

            if intento == 2:
                if self.busqueda_web_alternativa and self.herramienta_busqueda:
                    self._progreso(2, 4, "Consultando información adicional en la web...", saltar_linea=True)
                    try:
                        t_web_start = time.time()
                        resultados_web = self.herramienta_busqueda.invoke(consulta_optimizada)
                        tracker.add_latency("Generacion_BusquedaWeb", t_web_start)
                        tracker.set_meta("web_search_used", True)
                        texto_contexto += f"\n\n--- RESULTADOS DE BÚSQUEDA WEB ---\n{resultados_web}"
                        tracker.set_meta("context_size_chars", len(texto_contexto))
                    except Exception as e:
                        _log.warning(f"Falló la búsqueda web: {e}")
                else:
                    self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                    yield {"answer": "No poseo información suficiente en los apuntes para responder a tu pregunta sin inventar."}
                    yield {"context_docs": []}
                    return

            t_gen_start = time.time()
            args_invoke = {
                "chat_history": chat_history,
                "input": consulta,
                "context": texto_contexto,
                "graph_context": contexto_grafo,
                "feedback": mensaje_feedback,
            }

            if self.usar_critico:
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

            if "No poseo información suficiente" in borrador_limpio or "no está cubierto" in borrador_limpio.lower():
                if self.usar_critico:
                    self._progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                    yield {"answer": borrador}
                tracker.finish_and_log("REJECTED_SAFE")
                yield {"context_docs": []}
                return

            if "REQUIRE_WEB_SEARCH" in borrador_limpio and len(borrador_limpio) < 100:
                _log.info("Tutor solicitó búsqueda web (REQUIRE_WEB_SEARCH).")
                decision_critico = "RECHAZADO"
            elif any(w in borrador_limpio.lower()[:50] for w in ["lo siento", "no puedo", "hubo un error"]):
                decision_critico = "RECHAZADO"
            else:
                if self.usar_critico:
                    self._progreso(3, 4, "Crítico evaluando precisión y alucinaciones...", saltar_linea=True)
                    t_crit_start = time.time()
                    decision_critico = evaluar_borrador(self.llm_critic or self.llm, texto_contexto, borrador_limpio)
                    tracker.add_latency(f"Critico_Intento_{intento}", t_crit_start)
                else:
                    decision_critico = "APROBADO"

            es_aprobado = "APROBADO" in decision_critico or "APPROVED" in decision_critico
            es_rechazado = "RECHAZADO" in decision_critico or "REJECTED" in decision_critico

            if es_aprobado and not es_rechazado:
                if self.usar_critico:
                    self._progreso(4, 4, "¡Respuesta Aprobada!", saltar_linea=True)
                if self.cache_store:
                    try:
                        self.cache_store.add_texts(texts=[consulta], metadatas=[{"respuesta": borrador}])
                    except Exception as e:
                        _log.warning(f"Error guardando en caché semántico: {e}")
                tracker.finish_and_log("APPROVED")
                
                if self.usar_critico:
                    yield {"answer": borrador}
                
                yield {"context_docs": docs}
                return
            else:
                mensaje_feedback = "El revisor indicó que tu respuesta incluía afirmaciones no respaldadas. Por favor, sé más estricto."
                yield {"answer": "\n\n[!] El supervisor local detectó imprecisiones. Reintentando corregir la respuesta...\n\n"}
                continue

        self._progreso(4, 4, "Agotados los intentos.", saltar_linea=True)
        yield {"answer": "\n\nNo poseo información explícita en los apuntes para responder tu pregunta sin inventar."}
        tracker.finish_and_log("REJECTED_SAFE")
        yield {"context_docs": []}

def stream_consulta(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    consulta: str,
    chat_history: list = None,
    busqueda_web_alternativa: bool = False,
    cache_store: BaseRetriever = None,
    usar_critico: bool = True,
    llm_critic: BaseChatModel = None,
) -> Iterator[dict]:
    pipeline = PipelineConsulta(
        retriever=retriever,
        llm=llm,
        busqueda_web_alternativa=busqueda_web_alternativa,
        cache_store=cache_store,
        usar_critico=usar_critico,
        llm_critic=llm_critic,
    )
    
    def _progreso_print(paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False):
        porcentaje = int((paso / total) * 100)
        barra = "█" * (porcentaje // 10) + "░" * (10 - (porcentaje // 10))
        texto = f"\r[{barra}] {porcentaje:3}% | {mensaje}" + " " * 30
        
        if paso >= 3 and not hasattr(_progreso_print, "ya_salto"):
            print("\n")
            _progreso_print.ya_salto = True
            
        if saltar_linea:
            try:
                print(texto, flush=True)
            except UnicodeEncodeError:
                print(texto.replace("█", "#").replace("░", "-"), flush=True)
        else:
            try:
                print(texto, end="", flush=True)
            except UnicodeEncodeError:
                print(texto.replace("█", "#").replace("░", "-"), end="", flush=True)

    yield from pipeline.ejecutar(consulta, chat_history, _progreso_print)

def procesar_consulta(*args, **kwargs) -> dict:
    ans = ""
    docs = []
    for chunk in stream_consulta(*args, **kwargs):
        if "answer" in chunk:
            ans += chunk["answer"]
        if "context_docs" in chunk:
            docs = chunk["context_docs"]
    return {"answer": ans, "context_docs": docs}

def ejecutar_consulta(*args, **kwargs) -> dict:
    return procesar_consulta(*args, **kwargs)
