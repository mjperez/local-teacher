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
    
    GLINER_BATCH_SIZE = 32

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
    
    # Procesar en lotes para aprovechar batch_predict_entities
    for batch_start in range(0, total_docs, GLINER_BATCH_SIZE):
        batch_end = min(batch_start + GLINER_BATCH_SIZE, total_docs)
        
        # Filtrar fragmentos ya procesados dentro de este lote
        batch_indices = []
        batch_texts = []
        for i in range(batch_start, batch_end):
            if i not in processed_indices:
                batch_indices.append(i)
                batch_texts.append(docs[i].page_content)
        
        if not batch_texts:
            continue

        try:
            # Predicción en batch: una sola pasada por el modelo para N textos
            all_entities = model.inference(batch_texts, labels, threshold=0.5)
            

            
            # Validar longitud
            if len(all_entities) != len(batch_texts):
                _log.error(f"Inconsistencia en inferencia: esperadas {len(batch_texts)} salidas, obtenidas {len(all_entities)}.")
                continue

            for idx, entities, text in zip(batch_indices, all_entities, batch_texts):
                nombres = set()
                for e in entities:
                    ent_text = e["text"]
                    if _is_valid_entity(ent_text):
                        nombres.add(ent_text.strip().title())
                nombres = list(nombres)
                
                # Generar pares de co-ocurrencia
                chunk_nodes = set()
                chunk_edges = []
                for j in range(len(nombres)):
                    for k in range(j + 1, len(nombres)):
                        chunk_nodes.add(nombres[j])
                        chunk_nodes.add(nombres[k])
                        chunk_edges.append((nombres[j], nombres[k]))

                # Transacción por chunk: aísla los fallos de modo que un texto problemático no descarte a los demás
                conn.execute("BEGIN TRANSACTION")
                transaction_open = True
                chunk_has_errors = False
                try:
                    for node_name in chunk_nodes:
                        try:
                            conn.execute("MERGE (a:Entity {name: $name})", parameters={"name": node_name})
                        except Exception as e:
                            _log.error(f"KùzuDB Error en nodo '{node_name}': {e}")
                            chunk_has_errors = True
                    
                    for origen, destino in chunk_edges:
                        try:
                            conn.execute(
                                "MATCH (a:Entity {name: $o}), (b:Entity {name: $d}) MERGE (a)-[r:Rel {type: $rel}]->(b)", 
                                parameters={"o": origen, "d": destino, "rel": "CO_OCCURS_WITH"}
                            )
                            aristas_creadas += 1
                        except Exception as e:
                            _log.error(f"KùzuDB Error en arista '{origen}'-'{destino}': {e}")
                            chunk_has_errors = True

                    if chunk_has_errors:
                        _log.warning(f"Errores en chunk {idx}; ejecutando ROLLBACK para este fragmento.")
                        conn.execute("ROLLBACK")
                        transaction_open = False
                    else:
                        conn.execute("COMMIT")
                        transaction_open = False
                        processed_indices.add(idx)
                        sm.mark_graph_chunk(idx)
                except Exception as tx_err:
                    _log.error(f"Excepción en el chunk {idx}: {tx_err}")
                    if transaction_open:
                        try:
                            conn.execute("ROLLBACK")
                            transaction_open = False
                        except Exception:
                            pass
                finally:
                    if transaction_open:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass

            # Actualizar progreso DESPUÉS del lote para reflejar conteo real
            print(f"\r    - {len(processed_indices)}/{total_docs} fragmentos completados...", end="", flush=True)
                
        except Exception as e:
            _log.error(f"Error procesando lote {batch_start}-{batch_end}: {e}")

            
    print(f"\n[+] Grafo construido con {aristas_creadas} nuevas aristas de co-ocurrencia.")
    _log.info(f"Grafo construido en disco con {aristas_creadas} nuevas aristas.")
    
    return db
