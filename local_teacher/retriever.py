import re
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from langchain_community.tools import DuckDuckGoSearchRun

from local_teacher.graph_builder import load_knowledge_graph

_log = logging.getLogger(__name__)


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


def _reescribir_consulta(llm: BaseChatModel, consulta: str) -> ConsultaReescrita:
    """1. Agente de Reformulación (Rewrite)"""
    prompt_reescritura = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un bibliotecario experto. Tu tarea es optimizar la pregunta del estudiante para buscarla en una base de datos vectorial e híbrida.\n"
                "Reglas:\n"
                "1. MANTÉN ESTRICTAMENTE el idioma original de la consulta (si está en español, déjala en español).\n"
                "2. Extrae el número de CAPÍTULO SÓLO si la pregunta es exclusivamente sobre ese capítulo (ej. 'Resume el capítulo 4').\n"
                "3. Extrae los CONCEPTOS NÚCLEO o NOMBRES PROPIOS de la pregunta, en una lista de strings.",
            ),
            ("human", "{input}"),
        ]
    )

    try:
        cadena_reescritura = prompt_reescritura | llm.with_structured_output(
            ConsultaReescrita
        )
        respuesta = cadena_reescritura.invoke({"input": consulta})
        return respuesta
    except Exception as e:
        _log.error(f"[!] Falló la extracción estructurada: {e}. Usando fallback.")
        return ConsultaReescrita(consulta=consulta, capitulo=None, entidades=[])


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
        # Búsqueda usando límite de palabra para mayor precisión (regex)
        patron = re.compile(rf"\b{re.escape(entidad)}\b", re.IGNORECASE)
        nodos_encontrados = [n for n in nodos_en_grafo if patron.search(n)]

        for nodo in nodos_encontrados:
            palabras_clave_grafo.append(nodo)
            # Encontrar aristas salientes y entrantes (Vecindario de 1 salto)
            for _, target, data in G.out_edges(nodo, data=True):
                palabras_clave_grafo.append(target)
                rel = data.get("relacion", "->")
                conexiones.append(f"- {nodo} [{rel}] {target}")
            for source, _, data in G.in_edges(nodo, data=True):
                palabras_clave_grafo.append(source)
                rel = data.get("relacion", "->")
                conexiones.append(f"- {source} [{rel}] {nodo}")

    # Limpiar duplicados
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
    """3. Recuperación Vectorial (Retrieve) con Post-Filtrado y Entidades"""
    docs = []
    vistos_id = set()

    # Búsqueda independiente de entidades (muy útil para glosarios y conceptos sueltos)
    if entidades_filtro:
        for entidad in entidades_filtro:
            res = vectorstore.similarity_search(entidad, k=30)
            for d in res:
                content_hash = hash(d.page_content)
                if content_hash not in vistos_id:
                    docs.append(d)
                    vistos_id.add(content_hash)

    # Búsqueda semántica de la consulta completa
    res_completa = vectorstore.similarity_search(consulta_optimizada, k=40)
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

    # Re-ranking simple: Priorizar fragmentos que contienen menciones exactas de las entidades
    if entidades_filtro:
        def score_doc(d):
            content = d.page_content.lower()
            return sum(content.count(e.lower()) for e in entidades_filtro)
        docs = sorted(docs, key=score_doc, reverse=True)

    # Limitar el número de documentos finales para no inundar el contexto
    return docs[:20]


def _evaluar_borrador(llm: BaseChatModel, texto_contexto: str, borrador: str) -> str:
    """5. Agente Crítico (Self-RAG Evaluator)"""
    prompt_critico = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un clasificador binario. Tu única salida válida es APROBADO o RECHAZADO, sin ensayos ni explicaciones.",
            ),
            (
                "human",
                "Contexto Original: Guido van Rossum.\nRespuesta Generada: Python fue inventado en 1991.\n¿Alucinó datos técnicos? Responde APROBADO o RECHAZADO:",
            ),
            ("ai", "RECHAZADO"),
            (
                "human",
                "Contexto Original: SICP enseña Scheme.\nRespuesta Generada: SICP es un libro de Scheme.\n¿Alucinó datos técnicos? Responde APROBADO o RECHAZADO:",
            ),
            ("ai", "APROBADO"),
            (
                "human",
                "Contexto Original:\n{context}\n\nRespuesta Generada:\n{draft}\n\n"
                "INSTRUCCIÓN FINAL:\n"
                "1. Si la respuesta inicia con 'No encontré esta información en los apuntes...', responde APROBADO.\n"
                "2. Evalúa con flexibilidad: si la respuesta contiene información correcta que parece provenir del contexto (aunque esté parafraseada o resumida), responde APROBADO.\n"
                "3. Solo responde RECHAZADO si la respuesta inventa datos flagrantemente falsos o contradice el contexto de manera evidente.\n"
                "¿Aprobado o Rechazado? Escribe SOLO UNA PALABRA:",
            ),
        ]
    )
    cadena_critico = prompt_critico | llm | StrOutputParser()
    respuesta = cadena_critico.invoke({"context": texto_contexto, "draft": borrador})
    respuesta_limpia = re.sub(
        r"<think>.*?</think>", "", respuesta, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    return respuesta_limpia.upper()


def ejecutar_consulta(
    vectorstore: QdrantVectorStore,
    llm: BaseChatModel,
    consulta: str,
    transmitir: bool = False,
    busqueda_web_alternativa: bool = False,
) -> Any:

    def _progreso(
        paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False
    ):
        porcentaje = int((paso / total) * 100)
        barra = "█" * (porcentaje // 10) + "░" * (10 - (porcentaje // 10))
        texto = f"\r[{barra}] {porcentaje:3}% | {mensaje}".ljust(80)
        if saltar_linea:
            print(texto, flush=True)
        else:
            print(texto, end="", flush=True)

    _progreso(0, 4, "Optimizando pregunta...")

    # 1. Reescritura
    consulta_estructurada = _reescribir_consulta(llm, consulta)
    consulta_optimizada = consulta_estructurada.consulta
    capitulo_filtro = consulta_estructurada.capitulo
    entidades_filtro = consulta_estructurada.entidades

    _progreso(1, 4, "Buscando en Grafo de Conocimiento...")

    # 2. GraphRAG
    contexto_grafo, palabras_clave_grafo = _obtener_contexto_grafo(entidades_filtro)
    if palabras_clave_grafo:
        expansion = " ".join(palabras_clave_grafo[:5])
        consulta_optimizada = consulta_optimizada + " " + expansion

    # Preparar agente tutor
    prompt_tutor = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un tutor educativo experto. Tu objetivo es ayudar al alumno respondiendo ESTRICTAMENTE con los apuntes recuperados.",
            ),
            (
                "human",
                "Contexto de texto recuperado:\n{context}\n\n"
                "Conexiones Conceptuales Transversales (GraphRAG):\n{graph_context}\n\n"
                "PREGUNTA DEL ALUMNO: {input}\n\n"
                "INSTRUCCIONES FINALES OBLIGATORIAS:\n"
                "1. Si la respuesta está en el contexto recuperado, responde basándote en él y cita tus fuentes.\n"
                "2. Si la respuesta NO ESTÁ en el contexto (ej. un resumen general de un libro que no está en el texto), TIENES PROHIBIDO inventar.\n"
                "3. En su lugar, debes responder EXACTAMENTE con esta única palabra mágica para activar la búsqueda en internet:\n"
                "REQUIRE_WEB_SEARCH\n\n"
                "{feedback}",
            ),
        ]
    )
    cadena_tutor = prompt_tutor | llm

    herramienta_busqueda = None
    if busqueda_web_alternativa:
        herramienta_busqueda = DuckDuckGoSearchRun()

    def _generar_y_evaluar(docs):
        texto_contexto = _formatear_documentos(docs)
        mensaje_feedback = ""

        for intento in range(1, 4):
            _progreso(2, 4, f"Generando borrador inicial (Intento {intento}/3)...")

            if intento == 2:
                if busqueda_web_alternativa and herramienta_busqueda:
                    _progreso(2, 4, "Consultando información adicional en la web...")
                    try:
                        resultados_web = herramienta_busqueda.invoke(
                            consulta_optimizada
                        )
                        texto_contexto += (
                            f"\n\n--- RESULTADOS DE BÚSQUEDA WEB ---\n{resultados_web}"
                        )
                    except Exception as e:
                        _log.warning(f"Falló la búsqueda web: {e}")
                else:
                    respuesta_segura = (
                        "No poseo información suficiente en los apuntes para responder a tu pregunta sin inventar. "
                        "\n\n*Tip: Si quieres que busque la respuesta en Internet, ejecuta el comando añadiendo `--web-fallback` al final.*"
                    )
                    _progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                    return respuesta_segura

            # Generar borrador
            respuesta = cadena_tutor.invoke(
                {
                    "input": consulta,
                    "context": texto_contexto,
                    "graph_context": contexto_grafo,
                    "feedback": mensaje_feedback,
                }
            )
            borrador = (
                respuesta.content if hasattr(respuesta, "content") else str(respuesta)
            )

            if "No poseo información suficiente" in borrador:
                _progreso(4, 4, "¡Finalizado!", saltar_linea=True)
                return borrador  # Es seguro

            borrador_limpio = re.sub(
                r"<think>.*?</think>", "", borrador, flags=re.DOTALL
            ).strip()

            if "REQUIRE_WEB_SEARCH" in borrador_limpio and len(borrador_limpio) < 100:
                _log.info("Tutor solicitó búsqueda web (REQUIRE_WEB_SEARCH).")
                decision_critico = "RECHAZADO"
            else:
                _progreso(3, 4, "Crítico evaluando precisión y alucinaciones...")
                decision_critico = _evaluar_borrador(llm, texto_contexto, borrador_limpio)

            # DEFAULT DENY: Solo aprobamos si vemos APROBADO o APPROVED y NO vemos RECHAZADO o REJECTED
            es_aprobado = ("APROBADO" in decision_critico or "APPROVED" in decision_critico)
            es_rechazado = ("RECHAZADO" in decision_critico or "REJECTED" in decision_critico)
            
            if es_aprobado and not es_rechazado:
                _progreso(4, 4, "¡Respuesta Aprobada!", saltar_linea=True)
                return borrador
            else:
                _log.info(
                    f"Crítico rechazó borrador en intento {intento}. Razón: {decision_critico}"
                )
                mensaje_feedback = (
                    "El revisor indicó que tu respuesta incluía afirmaciones no respaldadas "
                    "por el texto recuperado. Por favor, sé más estricto y responde SÓLO "
                    "con la información del contexto."
                )
                continue

        _progreso(4, 4, "Agotados los intentos.", saltar_linea=True)
        return "No poseo información explícita en los apuntes para responder tu pregunta sin inventar (El filtro crítico de alucinaciones bloqueó todos los intentos)."

    if not transmitir:
        _progreso(1, 4, "Recuperando documentos...")
        docs = _recuperar_y_filtrar(vectorstore, consulta_optimizada, capitulo_filtro, entidades_filtro)
        respuesta_final = _generar_y_evaluar(docs)
        return {"answer": respuesta_final}

    def _generador_transmision():
        _progreso(1, 4, "Recuperando documentos...")
        docs = _recuperar_y_filtrar(vectorstore, consulta_optimizada, capitulo_filtro)
        respuesta_final = _generar_y_evaluar(docs)
        yield {"answer": respuesta_final}

    return _generador_transmision()
