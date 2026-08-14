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
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from langchain_community.tools import DuckDuckGoSearchRun
from flashrank import Ranker, RerankRequest

from local_teacher.ingestion.graph_builder import load_knowledge_graph
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
    for d in docs:
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
            f"--- INICIO FRAGMENTO ---\nMetadata: [{meta_str}]\nContenido:\n{d.page_content}\n--- FIN FRAGMENTO ---"
        )

    return "\n\n".join(res)


def _obtener_contexto_grafo(entidades_filtro: list[str]) -> tuple[str, list[str]]:
    """2. Consulta de Grafo de Conocimiento (GraphRAG) y Expansión de Búsqueda"""
    contexto_grafo = "(No se detectaron entidades o no hay grafo disponible)"
    palabras_clave_grafo = []

    if not entidades_filtro:
        return contexto_grafo, palabras_clave_grafo

    G = load_knowledge_graph()
    if G.number_of_nodes() == 0:
        return contexto_grafo, palabras_clave_grafo

    conexiones = []
    nodos_en_grafo = list(G.nodes())

    for entidad in entidades_filtro:
        patron = re.compile(rf"\b{re.escape(entidad)}\b", re.IGNORECASE)
        nodos_encontrados = [n for n in nodos_en_grafo if patron.search(n)]

        for nodo in nodos_encontrados:
            palabras_clave_grafo.append(nodo)
            for _, target, data in G.out_edges(nodo, data=True):
                palabras_clave_grafo.append(target)
                rel = data.get("relacion", "->")
                conexiones.append(f"- {nodo} [{rel}] {target}")
            for source, _, data in G.in_edges(nodo, data=True):
                palabras_clave_grafo.append(source)
                rel = data.get("relacion", "->")
                conexiones.append(f"- {source} [{rel}] {nodo}")

    conexiones = list(set(conexiones))
    if conexiones:
        contexto_grafo = "\n".join(conexiones)
        _log.info(
            f"[*] Se inyectaron {len(conexiones)} conexiones del Grafo de Conocimiento al contexto."
        )

    return contexto_grafo, list(set(palabras_clave_grafo))


def _recuperar_y_filtrar(
    vectorstore: QdrantVectorStore,
    consulta_optimizada: str,
    capitulo_filtro: Optional[str],
    entidades_filtro: list[str] = None,
) -> list[Document]:
    """3. Recuperación Híbrida y Reranking con FlashRank"""
    docs = []
    vistos_id = set()

    if entidades_filtro:
        for entidad in entidades_filtro:
            res = vectorstore.similarity_search(entidad, k=15)
            for d in res:
                content_hash = hash(d.page_content)
                if content_hash not in vistos_id:
                    docs.append(d)
                    vistos_id.add(content_hash)

    res_completa = vectorstore.similarity_search(consulta_optimizada, k=25)
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
    # Usar solo Top-K y delegar el filtrado semántico estricto al Crítico (LLM)
    for res in results[:10]:
        meta = res["meta"]
        text = res["text"]
        doc = Document(page_content=text, metadata=meta)
        doc.metadata["rerank_score"] = res["score"]
        final_docs.append(doc)

    return final_docs


def _crear_cadena_tutor(llm: BaseChatModel, busqueda_web_alternativa: bool, intencion: str = "conceptual"):
    system_base = (
        "Eres un tutor educativo experto. Tu objetivo es guiar al alumno respondiendo ESTRICTAMENTE con los apuntes recuperados. "
    )
    
    if intencion == "ejercicio":
        system_base += "El alumno necesita ayuda con un EJERCICIO O PROBLEMA PRÁCTICO. NO le des la solución directa bajo ninguna circunstancia. Dale pistas progresivas (scaffolding), guíalo paso a paso y hazle preguntas reflexivas para que descubra la solución por sí mismo. "
    elif intencion == "aclaracion":
        system_base += "El alumno tiene una duda rápida. Responde de forma directa, concisa y sin rodeos. "
    else:
        system_base += "Adopta un tono pedagógico y formativo: desglosa los problemas teóricos, explica el porqué de las cosas, y fomenta la comprensión profunda. Puedes hacer una pregunta de control al final para asegurar que el alumno entendió. "
        
    system_base += (
        "\nINSTRUCCIONES FINALES OBLIGATORIAS:\n"
        "1. Si la respuesta está en el contexto recuperado, responde basándote en él.\n"
        "2. Si te preguntan de qué trata el texto o piden un resumen general, sintetiza los temas principales basados únicamente en el contexto recuperado.\n"
        "3. Si la respuesta NO ESTÁ en el contexto o el contexto está vacío, indica explícitamente que el tema no está cubierto en el material cargado. TIENES PROHIBIDO inventar.\n"
        "4. Si no sabes, responde SÓLO con: REQUIRE_WEB_SEARCH (solo aplicable si busqueda_web_alternativa=True)."
    )
    
    prompt_tutor = ChatPromptTemplate.from_messages(
        [
            ("system", system_base),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Contexto de texto recuperado:\n{context}\n\n"
                "Conexiones Conceptuales Transversales (GraphRAG):\n{graph_context}\n\n"
                "PREGUNTA DEL ALUMNO: {input}\n\n"
                "{feedback}",
            ),
        ]
    )
    return prompt_tutor | llm


def _generador_procesar_consulta(
    vectorstore: QdrantVectorStore,
    llm: BaseChatModel,
    consulta: str,
    chat_history: list = None,
    busqueda_web_alternativa: bool = False,
    cache_store: QdrantVectorStore = None,
    progreso_callback=None,
    usar_critico: bool = True,
    llm_critic: BaseChatModel = None,
):
    tracker = QueryMetricsTracker(consulta, llm, llm_critic, usar_critico)

    if chat_history is None:
        chat_history = []

    def _progreso(
        paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False
    ):
        if progreso_callback:
            progreso_callback(paso, total, mensaje, saltar_linea)

    if cache_store:
        _progreso(0, 4, "Consultando Caché Semántico L1...")
        try:
            cache_results = cache_store.similarity_search_with_score(consulta, k=1)
            if cache_results:
                doc, score = cache_results[0]
                if score > 0.95:
                    _log.info(f"Cache Hit! Score: {score:.3f}")
                    _progreso(
                        4,
                        4,
                        "¡Respuesta rápida desde Caché Semántico!",
                        saltar_linea=True,
                    )
                    respuesta_cacheada = doc.metadata.get("respuesta", doc.page_content)
                    tracker.set_meta("cache_hit", True)
                    tracker.finish_and_log("CACHE_HIT")
                    yield {"answer": respuesta_cacheada}
                    yield {"context_docs": []}
                    return
        except Exception as e:
            _log.warning(f"Error consultando caché semántico: {e}")

    _progreso(0, 4, "Optimizando pregunta...")
    t_opt_start = time.time()

    consulta_estructurada = reescribir_consulta(llm, consulta)
    consulta_optimizada = consulta_estructurada.consulta
    tracker.set_meta("optimized_query", consulta_optimizada)
    capitulo_filtro = consulta_estructurada.capitulo
    entidades_filtro = consulta_estructurada.entidades
    intencion_filtro = getattr(consulta_estructurada, "intencion", "conceptual")

    tracker.add_latency("Optimizacion_Reescritura", t_opt_start)

    _progreso(1, 4, "Buscando en Grafo de Conocimiento...")
    t_grafo_start = time.time()

    contexto_grafo, palabras_clave_grafo = _obtener_contexto_grafo(entidades_filtro)
    if palabras_clave_grafo:
        expansion = " ".join(palabras_clave_grafo[:5])
        consulta_optimizada = consulta_optimizada + " " + expansion
        tracker.set_meta("optimized_query", consulta_optimizada)

    tracker.add_latency("Recuperacion_Grafo", t_grafo_start)

    _progreso(1, 4, "Recuperando documentos...")
    t_vec_start = time.time()
    docs = _recuperar_y_filtrar(
        vectorstore, consulta_optimizada, capitulo_filtro, entidades_filtro
    )
    tracker.add_latency("Recuperacion_Vectorial", t_vec_start)

    if not docs:
        _progreso(4, 4, "¡Finalizado!", saltar_linea=True)
        yield {
            "answer": "No he encontrado información sobre este tema en el material cargado. Al tratarse de un tutor basado estrictamente en el contenido provisto, no puedo responder esta pregunta."
        }
        tracker.finish_and_log("REJECTED_SAFE")
        yield {"context_docs": []}
        return

    cadena_tutor = _crear_cadena_tutor(llm, busqueda_web_alternativa, intencion_filtro)
    herramienta_busqueda = DuckDuckGoSearchRun() if busqueda_web_alternativa else None

    texto_contexto = _formatear_documentos(docs)
    tracker.set_meta("num_docs", len(docs))
    tracker.set_meta("context_size_chars", len(texto_contexto))
    mensaje_feedback = ""

    for intento in range(1, 4):
        tracker.set_meta("attempts", intento)
        tracker.init_latency(f"Generacion_LLM_Intento_{intento}")
        if usar_critico:
            tracker.init_latency(f"Critico_Intento_{intento}")
        _progreso(
            2, 4, f"Generando respuesta (Intento {intento}/3)...", saltar_linea=True
        )

        if intento == 2:
            if busqueda_web_alternativa and herramienta_busqueda:
                _progreso(
                    2,
                    4,
                    "Consultando información adicional en la web...",
                    saltar_linea=True,
                )
                try:
                    t_web_start = time.time()
                    resultados_web = herramienta_busqueda.invoke(consulta_optimizada)
                    tracker.add_latency("Generacion_BusquedaWeb", t_web_start)
                    tracker.set_meta("web_search_used", True)
                    texto_contexto += (
                        f"\n\n--- RESULTADOS DE BÚSQUEDA WEB ---\n{resultados_web}"
                    )
                    tracker.set_meta("context_size_chars", len(texto_contexto))
                except Exception as e:
                    _log.warning(f"Falló la búsqueda web: {e}")
            else:
                respuesta_segura = "No poseo información suficiente en los apuntes para responder a tu pregunta sin inventar."
                _progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                yield {"answer": respuesta_segura}
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

        if usar_critico:
            respuesta = cadena_tutor.invoke(args_invoke)
            borrador = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
        else:
            # Hacer streaming verdadero y devolver tokens inmediatamente
            borrador = ""
            # Si no hay crítico, la impresión de progreso en consola interfiere con el texto, 
            # así que forzamos un salto de línea limpio antes de empezar a escupir tokens.
            _progreso(3, 4, "Generando respuesta en tiempo real...", saltar_linea=True)
            for token_chunk in cadena_tutor.stream(args_invoke):
                content = token_chunk.content if hasattr(token_chunk, "content") else str(token_chunk)
                borrador += content
                yield {"answer": content}

        tracker.add_latency(f"Generacion_LLM_Intento_{intento}", t_gen_start)

        if (
            "No poseo información suficiente" in borrador
            or "no está cubierto" in borrador.lower()
        ):
            if usar_critico:
                _progreso(4, 4, "¡Finalizado!", saltar_linea=True)
            tracker.finish_and_log("REJECTED_SAFE")
            yield {"context_docs": []}
            return

        borrador_limpio = re.sub(
            r"<think>.*?</think>", "", borrador, flags=re.DOTALL
        ).strip()

        if "REQUIRE_WEB_SEARCH" in borrador_limpio and len(borrador_limpio) < 100:
            _log.info("Tutor solicitó búsqueda web (REQUIRE_WEB_SEARCH).")
            decision_critico = "RECHAZADO"
        elif any(
            w in borrador_limpio.lower()[:50]
            for w in ["lo siento", "no puedo", "hubo un error"]
        ):
            decision_critico = "RECHAZADO"
        else:
            if usar_critico:
                _progreso(
                    3,
                    4,
                    "Crítico evaluando precisión y alucinaciones...",
                    saltar_linea=True,
                )
                t_crit_start = time.time()
                decision_critico = evaluar_borrador(
                    llm_critic or llm, texto_contexto, borrador_limpio
                )
                tracker.add_latency(f"Critico_Intento_{intento}", t_crit_start)
            else:
                decision_critico = "APROBADO"

        es_aprobado = "APROBADO" in decision_critico or "APPROVED" in decision_critico
        es_rechazado = "RECHAZADO" in decision_critico or "REJECTED" in decision_critico

        if es_aprobado and not es_rechazado:
            if usar_critico:
                _progreso(4, 4, "¡Respuesta Aprobada!", saltar_linea=True)
            if cache_store:
                try:
                    cache_store.add_texts(
                        texts=[consulta], metadatas=[{"respuesta": borrador}]
                    )
                except Exception as e:
                    _log.warning(f"Error guardando en caché semántico: {e}")
            tracker.finish_and_log("APPROVED")
            
            if usar_critico:
                # Solo yeildeamos la respuesta final si el crítico estaba activo 
                # (si no, ya la fuimos yieldando token por token)
                yield {"answer": borrador}
            
            yield {"context_docs": docs}
            return
        else:
            mensaje_feedback = "El revisor indicó que tu respuesta incluía afirmaciones no respaldadas. Por favor, sé más estricto."
            yield {
                "answer": "\n\n[!] El supervisor local detectó imprecisiones. Reintentando corregir la respuesta...\n\n"
            }
            continue

    _progreso(4, 4, "Agotados los intentos.", saltar_linea=True)
    yield {
        "answer": "\n\nNo poseo información explícita en los apuntes para responder tu pregunta sin inventar."
    }
    tracker.finish_and_log("REJECTED_SAFE")
    yield {"context_docs": []}


def procesar_consulta(*args, **kwargs) -> dict:
    ans = ""
    docs = []
    for chunk in _generador_procesar_consulta(*args, **kwargs):
        if "answer" in chunk:
            ans += chunk["answer"]
        if "context_docs" in chunk:
            docs = chunk["context_docs"]
    return {"answer": ans, "context_docs": docs}


def stream_consulta(
    vectorstore: QdrantVectorStore,
    llm: BaseChatModel,
    consulta: str,
    chat_history: list = None,
    busqueda_web_alternativa: bool = False,
    cache_store: QdrantVectorStore = None,
    usar_critico: bool = True,
    llm_critic: BaseChatModel = None,
) -> Iterator[dict]:
    """Generador para devolver la respuesta poco a poco (o el estado)."""

    def _progreso_print(
        paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False
    ):
        porcentaje = int((paso / total) * 100)
        barra = "█" * (porcentaje // 10) + "░" * (10 - (porcentaje // 10))
        texto = f"\r[{barra}] {porcentaje:3}% | {mensaje}" + " " * 30
        
        # Si es el paso 3 o 4 (después del streaming), bajamos una línea para no sobreescribir los tokens
        if paso >= 3 and not hasattr(_progreso_print, "ya_salto"):
            print("\n")
            _progreso_print.ya_salto = True
            
        if saltar_linea:
            print(texto, flush=True)
        else:
            print(texto, end="", flush=True)

    yield from _generador_procesar_consulta(
        vectorstore,
        llm,
        consulta,
        chat_history,
        busqueda_web_alternativa,
        cache_store,
        _progreso_print,
        usar_critico,
        llm_critic,
    )


def ejecutar_consulta(
    vectorstore: QdrantVectorStore,
    llm: BaseChatModel,
    consulta: str,
    chat_history: list = None,
    busqueda_web_alternativa: bool = False,
    cache_store: QdrantVectorStore = None,
    usar_critico: bool = True,
    llm_critic: BaseChatModel = None,
) -> dict:
    """Ejecución síncrona, devuelve directamente el diccionario."""
    return procesar_consulta(
        vectorstore,
        llm,
        consulta,
        chat_history,
        busqueda_web_alternativa,
        cache_store,
        None,
        usar_critico,
        llm_critic,
    )
