from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MENSAJE_FALLBACK = "No he encontrado información sobre este tema en el material cargado. Al tratarse de un tutor basado estrictamente en el contenido provisto, no puedo responder esta pregunta sin inventar."

def formatear_documentos(docs: list[Document]) -> str:
    """Formatea la lista de documentos recuperados inyectando sus metadatos para el modelo."""
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


def crear_cadena_tutor(
    llm: BaseChatModel, busqueda_web_alternativa: bool, intencion: str = "conceptual"
):
    """Construye la cadena de generación pedagógica según la intención del estudiante."""
    system_base = (
        "REGLA PRINCIPAL: Responde ÚNICAMENTE con información que aparezca en el contexto recuperado. "
        "Si la información no está en el contexto, responde: 'Este tema no está cubierto en el material cargado.' "
        "NUNCA uses tu conocimiento interno para complementar o enriquecer la respuesta.\n\n"
        "Eres un tutor educativo. "
    )

    if intencion == "ejercicio":
        system_base += (
            "El alumno necesita ayuda con un EJERCICIO O PROBLEMA PRÁCTICO. "
            "NO le des la solución directa bajo ninguna circunstancia. "
            "Dale pistas progresivas (scaffolding), guíalo paso a paso y hazle preguntas reflexivas "
            "para que descubra la solución por sí mismo. "
        )
    elif intencion == "aclaracion":
        system_base += "El alumno tiene una duda rápida. Responde de forma directa, concisa y sin rodeos. "
    else:
        system_base += (
            "Adopta un tono pedagógico y formativo: desglosa los problemas teóricos, explica el porqué de las cosas, "
            "y fomenta la comprensión profunda. Puedes hacer una pregunta de control al final para asegurar que el alumno entendió. "
        )

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
