import json
import kuzu
import logging
import torch
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _log.info(f"Cargando modelo GLiNER en {device.upper()}...")
        print(f"[*] Cargando modelo GLiNER en {device.upper()} (GPU)..." if device == "cuda" else "[*] Cargando modelo GLiNER en CPU...")
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to(device)
    return _gliner_model

def build_knowledge_graph(docs: list[Document], llm: BaseChatModel, output_path: Path | str = "./local_teacher_kuzu"):
    """
    Extrae un Grafo de Co-ocurrencia de Entidades usando GLiNER acelerado por GPU e inserciones UNWIND en KùzuDB.
    
    Soporta reanudación: si el proceso se interrumpe, al volver a correr continuará
    desde el último fragmento procesado.
    """
    _log.info(f"Iniciando extracción de Grafo de Conocimiento (GraphRAG) para {len(docs)} fragmentos...")
    
    GLINER_BATCH_SIZE = 64

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
    
    # Procesar en lotes para aprovechar batch_predict_entities / inference
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
            # Adaptador para robustez entre versiones de GLiNER
            if hasattr(model, "batch_predict_entities"):
                all_entities = model.batch_predict_entities(batch_texts, labels, threshold=0.5)
            elif hasattr(model, "inference"):
                all_entities = model.inference(batch_texts, labels, threshold=0.5)
            elif hasattr(model, "predict_entities"):
                all_entities = [model.predict_entities(text, labels, threshold=0.5) for text in batch_texts]
            else:
                _log.error("GLiNER model no soporta extracción de entidades (versión incompatible).")
                continue

            # Validar longitud
            if len(all_entities) != len(batch_texts):
                _log.error(f"Inconsistencia en inferencia: esperadas {len(batch_texts)} salidas, obtenidas {len(all_entities)}.")
                continue

            all_batch_nodes = set()
            all_batch_edges = []

            for idx, entities, text in zip(batch_indices, all_entities, batch_texts):
                nombres = set()
                for e in entities:
                    ent_text = e["text"]
                    if _is_valid_entity(ent_text):
                        nombres.add(ent_text.strip().title())
                nombres = list(nombres)
                
                # Generar pares de co-ocurrencia
                for j in range(len(nombres)):
                    for k in range(j + 1, len(nombres)):
                        all_batch_nodes.add(nombres[j])
                        all_batch_nodes.add(nombres[k])
                        all_batch_edges.append([nombres[j], nombres[k]])

            # Inserción atómica en bloque usando UNWIND de Kùzu
            try:
                if all_batch_nodes:
                    nodes_json = json.dumps(list(all_batch_nodes))
                    conn.execute(f"UNWIND {nodes_json} AS n MERGE (a:Entity {{name: n}})")

                if all_batch_edges:
                    edges_json = json.dumps(all_batch_edges)
                    conn.execute(
                        f"UNWIND {edges_json} AS e "
                        "MATCH (a:Entity {name: e[1]}), (b:Entity {name: e[2]}) "
                        "MERGE (a)-[r:Rel {type: 'CO_OCCURS_WITH'}]->(b)"
                    )
                    aristas_creadas += len(all_batch_edges)

                for idx in batch_indices:
                    processed_indices.add(idx)
                    sm.mark_graph_chunk(idx)

            except Exception as db_err:
                _log.error(f"Error al insertar lote en KùzuDB: {db_err}")

            # Actualizar progreso DESPUÉS del lote para reflejar conteo real
            print(f"\r    - {len(processed_indices)}/{total_docs} fragmentos completados...", end="", flush=True)
                
        except Exception as e:
            _log.error(f"Error procesando lote {batch_start}-{batch_end}: {e}")

    print(f"\n[+] Grafo construido con {aristas_creadas} nuevas aristas de co-ocurrencia.")
    _log.info(f"Grafo construido en disco con {aristas_creadas} nuevas aristas.")
    
    return db

