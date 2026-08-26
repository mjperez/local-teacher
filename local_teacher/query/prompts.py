import os
import re
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
        "Eres un tutor educativo riguroso. Tu objetivo es explicar la materia apoyándote en el material provisto.\n\n"
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
            "Adopta un tono pedagógico y formativo: desglosa los conceptos, explica el porqué de las cosas, "
            "y fomenta la comprensión profunda del estudiante. "
        )

    system_base += (
        "\nINSTRUCCIONES:\n"
        "1. Basa tu explicación en los documentos recuperados y las conexiones conceptuales del grafo. Puedes resumir y estructurar la información para enseñar claramente.\n"
        "2. Si el contexto contiene información sobre el tema, desarróllala detallando lo que explican los documentos y cita las fuentes con corchetes (ej. '[1]', '[2]').\n"
        "3. Si los documentos y el grafo no contienen absolutamente ninguna mención del tema consultado, responde EXCLUSIVAMENTE: 'NO_INFO_EN_CONTEXTO'\n"
        "4. Si la información local es insuficiente y está activa la búsqueda web, responde: REQUIRE_WEB_SEARCH."
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


def _print_sources(docs, respuesta_final=None):
    if not docs:
        return

    citados = set()
    if respuesta_final:
        for match in re.finditer(r'\[([\d,\s]+)\]', respuesta_final):
            numeros = match.group(1).replace(',', ' ').split()
            for num in numeros:
                if num.isdigit():
                    citados.add(int(num))

    docs_a_imprimir = []
    if citados:
        for i, d in enumerate(docs, 1):
            if i in citados:
                docs_a_imprimir.append((i, d))
    else:
        docs_a_imprimir = list(enumerate(docs, 1))

    if not docs_a_imprimir:
        return

    print("\n\n---\nFuentes citadas:")

    grupos = {}
    for i, d in docs_a_imprimir:
        meta = d.metadata
        fuente = meta.get("fuente", "Desconocida")
        fuente_nombre = os.path.basename(fuente)
        pagina = meta.get("pagina", "N/A")
        seccion = meta.get("ruta_seccion", "N/A")

        if not pagina or str(pagina).strip() == "":
            pagina = "N/A"

        seccion_str = str(seccion).strip()
        if not seccion_str:
            seccion = "N/A"
        else:
            seccion = re.sub(r"^#+\s*", "", seccion_str).strip()

        tipo = meta.get("tipo_archivo", "texto")
        ruta_recurso = meta.get("ruta_recurso", "N/A")

        clave = (fuente_nombre, pagina, seccion, tipo, ruta_recurso)
        if clave not in grupos:
            grupos[clave] = []
        grupos[clave].append(i)

    for clave, indices in grupos.items():
        fuente_nombre, pagina, seccion, tipo, ruta_recurso = clave
        inds_str = ", ".join(f"[{i}]" for i in sorted(indices))
        info = (
            f"{inds_str} Archivo: {fuente_nombre} | Pag: {pagina} | Sección: {seccion}"
        )
        if tipo in ("figura", "tabla"):
            info += f" | Tipo: {tipo} | Ruta: {ruta_recurso}"
        print(info)

    print("---")
