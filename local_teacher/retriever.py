from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from pathlib import Path


def _format_docs(docs: list[Document]) -> str:
    """Formatea los documentos inyectando su metadata para el LLM."""
    res = []
    for d in docs:
        meta = []
        if d.metadata.get("curso"): 
            meta.append(f"Curso: {d.metadata['curso']}")
        if d.metadata.get("fuente"): 
            nombre = d.metadata.get("titulo") or Path(d.metadata['fuente']).name
            meta.append(f"Archivo: {nombre}")
        if d.metadata.get("pagina") is not None: 
            meta.append(f"Página: {d.metadata['pagina']}")
            
        sec = d.metadata.get("ruta_seccion")
        if not sec:
            enc = [d.metadata.get("encabezado_1"), d.metadata.get("encabezado_2"), d.metadata.get("encabezado_3")]
            sec = " > ".join([e for e in enc if e])
        if sec: 
            meta.append(f"Sección: {sec}")
            
        meta_str = " | ".join(meta)
        res.append(f"--- INICIO FRAGMENTO ---\nMetadata: [{meta_str}]\nContenido:\n{d.page_content}\n--- FIN FRAGMENTO ---")
    
    return "\n\n".join(res)

def ejecutar_query(vectorstore: QdrantVectorStore, llm: BaseChatModel, query: str, stream: bool = False) -> Any:
    # 1. Agente de Reformulación (Rewrite)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un bibliotecario experto. Tu tarea es optimizar la pregunta del estudiante para buscarla en una base de datos vectorial e híbrida.\n"
            "Reglas:\n"
            "1. Traduce la consulta al inglés si el material original suele estar en ese idioma.\n"
            "2. Si el estudiante pide un resumen o pregunta 'de qué trata' un capítulo específico (ej. 'Capítulo 4'), DEBES extraer ese número.\n"
            "3. RESPONDE ESTRICTAMENTE EN ESTE FORMATO:\n"
            "QUERY: <tu búsqueda optimizada>\n"
            "CAPITULO: <número del capítulo, o N/A si no mencionó uno>\n\n"
            "Ejemplos:\n"
            "Usuario: '¿De qué trata el capítulo 4?'\n"
            "Tu respuesta:\n"
            "QUERY: Chapter 4 summary concepts\n"
            "CAPITULO: 4\n\n"
            "Usuario: '¿Cómo funciona el garbage collector?'\n"
            "Tu respuesta:\n"
            "QUERY: How does the garbage collector work?\n"
            "CAPITULO: N/A"
        ),
        ("human", "{input}")
    ])
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    raw_query = rewrite_chain.invoke({"input": query})
    
    # Extraer QUERY y CAPITULO
    optimized_query = query
    capitulo_filtro = None
    for linea in raw_query.split('\n'):
        linea = linea.strip()
        if linea.startswith("QUERY:"):
            optimized_query = linea.replace("QUERY:", "").strip()
        elif linea.startswith("CAPITULO:") and "N/A" not in linea:
            capitulo_filtro = linea.replace("CAPITULO:", "").strip()

    print(f"\n[Agente] Consulta original: '{query}'")
    print(f"[Agente] Consulta optimizada para buscar: '{optimized_query}'")
    if capitulo_filtro:
        print(f"[Agente] Aplicando post-filtro para el Capítulo: {capitulo_filtro}\n")
    else:
        print("")

    # 2. Recuperación (Retrieve) con Post-Filtrado
    def _recuperar_y_filtrar():
        # Traemos 800 documentos para asegurar que capturamos el capítulo entero
        retriever_masivo = vectorstore.as_retriever(search_kwargs={"k": 800})
        docs = retriever_masivo.invoke(optimized_query)
        
        if capitulo_filtro:
            # Filtrar documentos cuya ruta_seccion empiece por el número del capítulo
            docs_filtrados = []
            prefix = f"{capitulo_filtro}."
            for d in docs:
                ruta = d.metadata.get("ruta_seccion", "")
                if ruta.strip().startswith(prefix) or f" {prefix}" in ruta:
                    docs_filtrados.append(d)
                    
            if not docs_filtrados:
                return docs[:15]
                
            # Ordenamos todo el capítulo cronológicamente
            docs_ordenados_por_lectura = sorted(docs_filtrados, key=lambda x: x.metadata.get("chunk_index", 999999))
            
            seleccion = []
            vistos = set()
            
            # 1. Muestreo Uniforme ("Lectura Rápida")
            # Tomamos 8 fragmentos equiespaciados de principio a fin del capítulo
            if len(docs_ordenados_por_lectura) > 8:
                step = len(docs_ordenados_por_lectura) / 8.0
                for i in range(8):
                    idx_to_take = int(i * step)
                    d = docs_ordenados_por_lectura[idx_to_take]
                    idx = d.metadata.get("chunk_index")
                    if idx not in vistos:
                        seleccion.append(d)
                        vistos.add(idx)
            else:
                for d in docs_ordenados_por_lectura:
                    idx = d.metadata.get("chunk_index")
                    if idx not in vistos:
                        seleccion.append(d)
                        vistos.add(idx)
                        
            # 2. Relevancia (Para responder preguntas específicas)
            # Rellenamos hasta 15 con los fragmentos más relevantes matemáticamente
            for d in docs_filtrados:
                idx = d.metadata.get("chunk_index")
                if idx not in vistos:
                    seleccion.append(d)
                    vistos.add(idx)
                if len(seleccion) >= 15:
                    break
                    
            # Ordenar selección final para que el Tutor lea en orden natural
            return sorted(seleccion, key=lambda x: x.metadata.get("chunk_index", 999999))
            
        # Si no hay filtro de capítulo, traemos 15 normales
        return sorted(docs[:15], key=lambda x: x.metadata.get("chunk_index", 999999))

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un tutor educativo inteligente. Responde a la pregunta del alumno basándote ESTRICTAMENTE en el contexto proporcionado.\n\n"
            "El contexto viene de varios documentos y apuntes. Cada fragmento tiene metadata asociada (Curso, Archivo, Página, Sección). "
            "Es MUY IMPORTANTE que en tu respuesta cites de dónde sacaste la información (mencionando Archivo y Página) para que el alumno pueda estudiar. "
            "Si la respuesta a la pregunta no está en el contexto, di explícitamente que no tienes suficiente información. NUNCA inventes información ni documentos.\n\n"
            "Contexto recuperado:\n{context}\n\n"
            "Si la respuesta a la pregunta no está en el contexto, responde exactamente: 'No poseo información suficiente en los apuntes proporcionados'."
        ),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    if not stream:
        docs = _recuperar_y_filtrar()
        context_str = _format_docs(docs)
        response = chain.invoke({"input": query, "context": context_str})
        return {"answer": response.content}

    def _stream_generator():
        docs = _recuperar_y_filtrar()
        context_str = _format_docs(docs)
        
        has_yielded = False
        for chunk in chain.stream({"input": query, "context": context_str}):
            has_yielded = True
            text_chunk = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield {"answer": text_chunk}
        
        if not has_yielded:
            yield {"answer": "\n[Error: El modelo no devolvió ninguna respuesta. Verifica la conexión.]\n"}

    return _stream_generator()