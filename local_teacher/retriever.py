from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from local_teacher.graph_builder import load_knowledge_graph
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

def ejecutar_query(vectorstore: QdrantVectorStore, llm: BaseChatModel, query: str, stream: bool = False, web_fallback: bool = False) -> Any:
    # 1. Agente de Reformulación (Rewrite)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un bibliotecario experto. Tu tarea es optimizar la pregunta del estudiante para buscarla en una base de datos vectorial e híbrida.\n"
            "Reglas:\n"
            "1. Traduce la consulta al inglés si el material original suele estar en ese idioma.\n"
            "2. Extrae el número de CAPÍTULO SÓLO si la pregunta es exclusivamente sobre ese capítulo (ej. 'Resume el capítulo 4'). Si la pregunta compara conceptos o fuentes distintas, pon CAPITULO: N/A para no bloquear la búsqueda cruzada.\n"
            "3. Extrae los CONCEPTOS NÚCLEO o NOMBRES PROPIOS de la pregunta, separados por comas. Debes ser conciso para que coincidan con un Grafo.\n"
            "   Por ejemplo: si el usuario dice 'el paper de thompson', extrae solo 'Thompson'. Si dice 'teoría de compiladores', extrae 'Compiladores'.\n"
            "4. RESPONDE ESTRICTAMENTE EN ESTE FORMATO:\n"
            "QUERY: <tu búsqueda optimizada>\n"
            "CAPITULO: <número del capítulo, o N/A>\n"
            "ENTIDADES: <entidad1, entidad2, o N/A>\n\n"
            "Ejemplos:\n"
            "Usuario: '¿De qué trata el capítulo 4?'\n"
            "Tu respuesta:\n"
            "QUERY: Chapter 4 summary concepts\n"
            "CAPITULO: 4\n"
            "ENTIDADES: N/A\n\n"
            "Usuario: '¿Qué relación hay entre compiler y object code?'\n"
            "Tu respuesta:\n"
            "QUERY: Relationship between compiler and object code\n"
            "CAPITULO: N/A\n"
            "ENTIDADES: Compiler, Object Code"
        ),
        ("human", "{input}")
    ])
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    raw_query = rewrite_chain.invoke({"input": query})
    
    # Extraer QUERY, CAPITULO y ENTIDADES
    optimized_query = query
    capitulo_filtro = None
    entidades_filtro = []
    
    for linea in raw_query.split('\n'):
        linea = linea.strip()
        if linea.startswith("QUERY:"):
            optimized_query = linea.replace("QUERY:", "").strip()
        elif linea.startswith("CAPITULO:") and "N/A" not in linea:
            capitulo_filtro = linea.replace("CAPITULO:", "").strip()
        elif linea.startswith("ENTIDADES:") and "N/A" not in linea:
            ents = linea.replace("ENTIDADES:", "").strip()
            entidades_filtro = [e.strip() for e in ents.split(",")]

    print(f"\n[Agente] Consulta original: '{query}'")
    print(f"[Agente] Consulta optimizada para buscar: '{optimized_query}'")
    if capitulo_filtro:
        print(f"[Agente] Aplicando post-filtro para el Capítulo: {capitulo_filtro}")
    if entidades_filtro:
        print(f"[Agente] Entidades extraídas para GraphRAG: {', '.join(entidades_filtro)}")
    print("")

    # 2. Consulta de Grafo de Conocimiento (GraphRAG) y Expansión de Búsqueda
    graph_context = "(No se detectaron entidades o no hay grafo disponible)"
    graph_keywords = []
    
    if entidades_filtro:
        G = load_knowledge_graph()
        if G.number_of_nodes() > 0:
            conexiones = []
            nodos_en_grafo = list(G.nodes())
            
            for entidad in entidades_filtro:
                # Búsqueda substring case-insensitive
                e_lower = entidad.lower()
                nodos_encontrados = [n for n in nodos_en_grafo if e_lower in n.lower()]
                
                for nodo in nodos_encontrados:
                    graph_keywords.append(nodo)
                    # Encontrar aristas salientes y entrantes (Vecindario de 1 salto)
                    for _, target, data in G.out_edges(nodo, data=True):
                        graph_keywords.append(target)
                        rel = data.get('relacion', '->')
                        conexiones.append(f"- {nodo} [{rel}] {target}")
                    for source, _, data in G.in_edges(nodo, data=True):
                        graph_keywords.append(source)
                        rel = data.get('relacion', '->')
                        conexiones.append(f"- {source} [{rel}] {nodo}")
            
            # Limpiar duplicados
            conexiones = list(set(conexiones))
            if conexiones:
                graph_context = "\n".join(conexiones)
                print(f"[*] Se inyectaron {len(conexiones)} conexiones del Grafo de Conocimiento al contexto.")
                
                # Graph-Augmented Query Expansion
                # Añadimos los conceptos vecinos más relevantes a la búsqueda vectorial
                graph_keywords = list(set(graph_keywords))
                if graph_keywords:
                    # Limitamos a los 5 primeros para no saturar los embeddings
                    expansion = " ".join(graph_keywords[:5])
                    print(f"[*] Expansión de Búsqueda (Graph-Augmented): Añadiendo '{expansion}' a Qdrant.\n")
                    optimized_query = optimized_query + " " + expansion

    # 3. Recuperación Vectorial (Retrieve) con Post-Filtrado
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
            for d in docs_filtrados:
                idx = d.metadata.get("chunk_index")
                if idx not in vistos:
                    seleccion.append(d)
                    vistos.add(idx)
                if len(seleccion) >= 15:
                    break
                    
            return sorted(seleccion, key=lambda x: x.metadata.get("chunk_index", 999999))
            
        return sorted(docs[:15], key=lambda x: x.metadata.get("chunk_index", 999999))

    # 4. Agente Tutor (Generador del Borrador)
    tutor_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un tutor educativo experto. Tu objetivo es ayudar al alumno respondiendo ESTRICTAMENTE con los apuntes recuperados."
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
            "{feedback}"
        ),
    ])

    tutor_chain = tutor_prompt | llm

    # 5. Agente Crítico (Self-RAG Evaluator)
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un clasificador binario. Tu única salida válida es APROBADO o RECHAZADO, sin ensayos ni explicaciones."),
        ("human", "Contexto Original: Guido van Rossum.\nRespuesta Generada: Python fue inventado en 1991.\n¿Alucinó datos técnicos? Responde APROBADO o RECHAZADO:"),
        ("ai", "RECHAZADO"),
        ("human", "Contexto Original: SICP enseña Scheme.\nRespuesta Generada: SICP es un libro de Scheme.\n¿Alucinó datos técnicos? Responde APROBADO o RECHAZADO:"),
        ("ai", "APROBADO"),
        ("human", "Contexto Original:\n{context}\n\nRespuesta Generada:\n{draft}\n\n"
                  "INSTRUCCIÓN FINAL: Si la respuesta inicia con 'No encontré esta información en los apuntes...', responde APROBADO. "
                  "Si la respuesta inventa datos afirmando que están en el contexto, responde RECHAZADO. "
                  "¿Aprobado o Rechazado? Escribe SOLO UNA PALABRA:")
    ])
    critic_chain = critic_prompt | llm | StrOutputParser()

    def _generar_y_evaluar(docs):
        context_str = _format_docs(docs)
        feedback_msg = ""
        
        for intento in range(1, 4):
            if intento > 1:
                print(f"\n[*] Intento {intento}: El Crítico rechazó la respuesta anterior. Reintentando con feedback...", flush=True)
                
            if intento == 2:
                if web_fallback:
                    print(f"[*] (Fallback Web) Consultando en internet: '{optimized_query}'...", flush=True)
                    try:
                        from langchain_community.tools import DuckDuckGoSearchRun
                        search = DuckDuckGoSearchRun()
                        web_results = search.invoke(optimized_query)
                        context_str += f"\n\n--- RESULTADOS DE BÚSQUEDA WEB ---\n{web_results}"
                        print("[*] (Fallback Web) Resultados inyectados en el contexto.", flush=True)
                    except Exception as e:
                        print(f"[!] Falló la búsqueda web: {e}", flush=True)
                else:
                    return (
                        "No poseo información suficiente en los apuntes para responder a tu pregunta sin inventar. "
                        "\n\n💡 *Tip: Si quieres que busque la respuesta en Internet, ejecuta el comando añadiendo `--web-fallback` al final.*"
                    )
                
            # Generar borrador
            response = tutor_chain.invoke({
                "input": query, 
                "context": context_str, 
                "graph_context": graph_context,
                "feedback": feedback_msg
            })
            draft = response.content if hasattr(response, "content") else str(response)
            
            if "No poseo información suficiente" in draft:
                return draft # Es seguro
                
            print(f"\n[*] Tutor generó un borrador (Intento {intento}):\n{draft}\n", flush=True)
            
            import re
            draft_clean = re.sub(r'<think>.*?</think>', '', draft, flags=re.DOTALL).strip()
            
            if "REQUIRE_WEB_SEARCH" in draft_clean and len(draft_clean) < 100:
                print("[*] Tutor solicitó búsqueda web (REQUIRE_WEB_SEARCH). Abortando evaluación y forzando reintento...", flush=True)
                critic_decision = "RECHAZADO"
            else:
                print(f"[*] Pasando al Crítico (Self-RAG)...", flush=True)
                # Evaluar
                critic_decision = critic_chain.invoke({"context": context_str, "draft": draft}).strip().upper()
            
            # DEFAULT DENY: Solo aprobamos si dice explícitamente APROBADO y NO dice RECHAZADO.
            if "APROBADO" in critic_decision and "RECHAZADO" not in critic_decision:
                print(f"[*] Crítico aprobó la respuesta. (Razón cruda: {critic_decision})", flush=True)
                return draft
            else:
                print(f"[*] Crítico detectó ALUCINACIÓN o falló la evaluación. (Razón cruda: {critic_decision})", flush=True)
                
                # Preparamos el feedback para el siguiente intento
                feedback_msg = (
                    "FEEDBACK DEL EVALUADOR (IMPORTANTE): Tu respuesta anterior fue RECHAZADA por inventar información. "
                    "Inténtalo de nuevo. Si el usuario pide un resumen y no hay uno oficial en el texto, usa las "
                    "Conexiones Conceptuales del Grafo y los temas recurrentes para sintetizar un resumen general, "
                    "pero NO inventes datos duros, cantidades de capítulos ni autores. Si es imposible responder sin inventar, ríndete admitiendo que no lo sabes."
                )
                continue
            
        return "No poseo información explícita en los apuntes para responder tu pregunta sin inventar (El filtro crítico de alucinaciones bloqueó todos los intentos)."

    if not stream:
        docs = _recuperar_y_filtrar()
        final_answer = _generar_y_evaluar(docs)
        return {"answer": final_answer}

    def _stream_generator():
        docs = _recuperar_y_filtrar()
        print("\n[*] Modo seguro activo: El texto no aparecerá letra por letra. Generando y evaluando...", flush=True)
        
        final_answer = _generar_y_evaluar(docs)
        yield {"answer": final_answer}

    return _stream_generator()