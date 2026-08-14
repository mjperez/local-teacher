import networkx as nx
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_log = logging.getLogger(__name__)

def build_knowledge_graph(docs: list[Document], llm: BaseChatModel, output_path: Path | str = "conocimiento.graphml"):
    """
    Extrae tripletas (Entidad -> Relación -> Entidad) de los documentos usando el LLM
    y construye un grafo usando NetworkX.
    """
    _log.info(f"Iniciando extracción de Grafo de Conocimiento (GraphRAG) para {len(docs)} fragmentos...")
    
    # Prompt optimizado para modelos pequeños (3B) usando formato de texto simple en lugar de JSON
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un experto en extraer Grafos de Conocimiento. "
            "Tu tarea es leer el texto y extraer las relaciones clave entre conceptos o entidades.\n"
            "Reglas:\n"
            "1. Extrae solo las relaciones más importantes.\n"
            "2. Usa nombres cortos para las entidades (ej. 'Compiler', 'Source Code').\n"
            "3. RESPONDE ÚNICAMENTE usando este formato exacto por línea, sin markdown ni explicaciones:\n"
            "[Entidad A] ||| [Relación] ||| [Entidad B]\n\n"
            "Ejemplo:\n"
            "Compiler ||| translates ||| Source Code\n"
            "Ken Thompson ||| wrote ||| Reflections on Trusting Trust"
        ),
        ("human", "Texto:\n{text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Cargar grafo existente si vamos a anexar conocimiento de varios libros
    if Path(output_path).exists():
        _log.info(f"Cargando grafo existente desde {output_path} para expandirlo...")
        G = nx.read_graphml(str(output_path))
    else:
        G = nx.DiGraph()
    
    for i, doc in enumerate(docs):
        _log.info(f"Extrayendo grafo del fragmento {i+1}/{len(docs)}...")
        try:
            # Extraer tripletas
            raw_output = chain.invoke({"text": doc.page_content})
            
            # Parseo tolerante a fallos
            tripletas_extraidas = 0
            for linea in raw_output.split('\n'):
                linea = linea.strip()
                if "|||" in linea:
                    partes = [p.strip() for p in linea.split("|||")]
                    if len(partes) >= 3:
                        origen, relacion, destino = partes[0], partes[1], partes[2]
                        # Limpiar corchetes si el modelo los incluyó por error
                        origen = origen.replace("[", "").replace("]", "").strip()
                        relacion = relacion.replace("[", "").replace("]", "").strip()
                        destino = destino.replace("[", "").replace("]", "").strip()
                        
                        if origen and destino and relacion:
                            # Añadir al grafo de NetworkX
                            G.add_edge(origen, destino, relacion=relacion)
                            tripletas_extraidas += 1
            _log.info(f"Ok ({tripletas_extraidas} tripletas extraídas)")
            
            # Guardado incremental cada 20 fragmentos para evitar pérdida de datos en textos grandes
            if (i + 1) % 20 == 0:
                nx.write_graphml(G, str(output_path))
                _log.info(f"Guardado incremental: {G.number_of_nodes()} nodos actuales")
                
        except Exception as e:
            _log.error(f"Error: {e}")
            
    _log.info(f"Grafo construido: {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
    
    # Guardar en disco
    nx.write_graphml(G, str(output_path))
    _log.info(f"Grafo guardado en {output_path}")
    return G

def load_knowledge_graph(path: Path | str = "conocimiento.graphml") -> nx.DiGraph:
    """Carga el grafo desde el disco si existe."""
    if Path(path).exists():
        return nx.read_graphml(str(path))
    return nx.DiGraph()
