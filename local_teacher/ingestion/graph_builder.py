import json
import kuzu
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

_log = logging.getLogger(__name__)

_CHECKPOINT_PATH = Path("./.graph_checkpoint.json")

def _load_checkpoint() -> set[int]:
    """Carga los índices de fragmentos ya procesados desde el checkpoint."""
    if _CHECKPOINT_PATH.exists():
        try:
            data = json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
            indices = set(data.get("processed", []))
            print(f"[*] Checkpoint encontrado: {len(indices)} fragmentos ya procesados. Reanudando...")
            return indices
        except Exception:
            pass
    return set()

def _save_checkpoint(processed: set[int]) -> None:
    """Guarda los índices de fragmentos ya procesados al disco."""
    try:
        _CHECKPOINT_PATH.write_text(
            json.dumps({"processed": list(processed)}, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        _log.warning(f"No se pudo guardar el checkpoint: {e}")

def _clear_checkpoint() -> None:
    """Elimina el checkpoint al terminar exitosamente."""
    try:
        if _CHECKPOINT_PATH.exists():
            _CHECKPOINT_PATH.unlink()
    except Exception:
        pass

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
    processed_indices = _load_checkpoint()
    
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
            
            # Limpiar nombres
            nombres = set([e["text"].strip().title() for e in entities if len(e["text"]) > 2])
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
            # Guardamos cada 10 iteraciones para no hacer IO constante
            if i % 10 == 0:
                _save_checkpoint(processed_indices)
                
        except Exception as e:
            _log.error(f"Error procesando fragmento {i}: {e}")
            
    _save_checkpoint(processed_indices)
            
    print(f"\n[+] Grafo construido con {aristas_creadas} nuevas aristas de co-ocurrencia.")
    _log.info(f"Grafo construido en disco con {aristas_creadas} nuevas aristas.")
    
    # Limpiar checkpoint al terminar exitosamente
    _clear_checkpoint()
    
    return db
