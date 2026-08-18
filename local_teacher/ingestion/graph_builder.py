import json
import kuzu
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from local_teacher.ingestion.state_manager import get_state_manager

_log = logging.getLogger(__name__)

STOP_WORDS_GENERIC = {
    # Spanish generics
    "ejemplo", "capitulo", "universidad", "profesor", "alumno", "estudiante", "seccion",
    "figura", "tabla", "grafico", "introduccion", "conclusion", "resumen", "parte",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a", "ante", "bajo", "cabe", "con",
    # English generics
    "example", "chapter", "university", "professor", "student", "section",
    "figure", "table", "chart", "introduction", "conclusion", "summary", "part",
    "the", "a", "an", "of", "to", "in", "for", "with", "on", "at", "from", "by", "about",
}

def _is_valid_entity(text: str) -> bool:
    """Filtra entidades que son demasiado genéricas o cortas."""
    text_clean = text.lower().strip()
    if len(text_clean) <= 2:
        return False
    if text_clean in STOP_WORDS_GENERIC:
        return False
    # Rechazar si es solo un número
    if text_clean.isnumeric():
        return False
    return True

# Inicialización Lazy de GLiNER
_gliner_model = None

def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        _log.info("Cargando modelo GLiNER (esto puede tomar un momento la primera vez)...")
        from gliner import GLiNER
        # "urchade/gliner_medium-v2.1" es rápido y muy preciso
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    return _gliner_model

def build_knowledge_graph(docs: list[Document], llm: BaseChatModel, output_path: Path | str = "./local_teacher_kuzu"):
    """
    Extrae un Grafo de Co-ocurrencia de Entidades usando GLiNER.
    
    Soporta reanudación: si el proceso se interrumpe, al volver a correr continuará
    desde el último fragmento procesado.
    """
    _log.info(f"Iniciando extracción de Grafo de Conocimiento (GraphRAG) para {len(docs)} fragmentos...")
    
    # Cargar base de datos Kùzu
    db_path = str(output_path)
    _log.info(f"Conectando a KùzuDB en {db_path}...")
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    
    try:
        conn.execute("CREATE NODE TABLE Entity (name STRING, PRIMARY KEY (name))")
        conn.execute("CREATE REL TABLE Rel (FROM Entity TO Entity, type STRING)")
    except RuntimeError:
        _log.info("Tablas ya existen. Expandiendo grafo...")
    
    # Cargar checkpoint si existe
    sm = get_state_manager()
    processed_indices = sm.get_graph_chunks()
    
    total_docs = len(docs)
    aristas_creadas = 0
    saltados = len(processed_indices)
    
    print(f"[*] Construyendo grafo de conocimiento para {total_docs} fragmentos...")
    if saltados:
        print(f"    ({saltados} ya procesados, {total_docs - saltados} pendientes)")
    
    labels = ["Person", "Organization", "Technology", "Concept", "Tool", "Process", "Algorithm", "Metric"]
    
    model = _get_gliner()
    
    for i, doc in enumerate(docs):
        # Saltar fragmentos ya procesados
        if i in processed_indices:
            continue
            
        print(f"\r    - Fragmento {i+1}/{total_docs} ({len(processed_indices)}/{total_docs} completados)...", end="", flush=True)
        try:
            # Predecir entidades
            entities = model.predict_entities(doc.page_content, labels, threshold=0.5)
            
            # Limpiar y filtrar nombres
            nombres = set()
            for e in entities:
                text = e["text"]
                if _is_valid_entity(text):
                    nombres.add(text.strip().title())
            nombres = list(nombres)
            
            # Crear grafo de co-ocurrencia: vincular todas las entidades encontradas en este fragmento
            tripletas_extraidas = 0
            for j in range(len(nombres)):
                for k in range(j + 1, len(nombres)):
                    origen = nombres[j]
                    destino = nombres[k]
                    relacion = "CO_OCCURS_WITH"
                    
                    try:
                        conn.execute("MERGE (a:Entity {name: $name})", parameters={"name": origen})
                        conn.execute("MERGE (b:Entity {name: $name})", parameters={"name": destino})
                        conn.execute(
                            "MATCH (a:Entity {name: $o}), (b:Entity {name: $d}) MERGE (a)-[r:Rel {type: $rel}]->(b)", 
                            parameters={"o": origen, "d": destino, "rel": relacion}
                        )
                        tripletas_extraidas += 1
                        aristas_creadas += 1
                    except Exception as e:
                        _log.error(f"KùzuDB Error en arista: {e}")
            
            processed_indices.add(i)
            # Guardamos el progreso
            sm.mark_graph_chunk(i)
                
        except Exception as e:
            _log.error(f"Error procesando fragmento {i}: {e}")
            
    print(f"\n[+] Grafo construido con {aristas_creadas} nuevas aristas de co-ocurrencia.")
    _log.info(f"Grafo construido en disco con {aristas_creadas} nuevas aristas.")
    
    return db
