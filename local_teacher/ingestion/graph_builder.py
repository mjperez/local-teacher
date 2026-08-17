import concurrent.futures
import json
import kuzu
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

def build_knowledge_graph(docs: list[Document], llm: BaseChatModel, output_path: Path | str = "./local_teacher_kuzu"):
    """
    Extrae tripletas (Entidad -> Relación -> Entidad) de los documentos usando el LLM
    y construye un grafo persistente usando KùzuDB.
    
    Soporta reanudación: si el proceso se interrumpe, al volver a correr continuará
    desde el último fragmento procesado.
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
    
    for i, doc in enumerate(docs):
        # Saltar fragmentos ya procesados
        if i in processed_indices:
            continue

        print(f"\r    - Fragmento {i+1}/{total_docs} ({len(processed_indices)}/{total_docs} completados)...", end="", flush=True)
        _log.info(f"Extrayendo grafo del fragmento {i+1}/{total_docs}...")
        try:
            # Timeout de 30s por fragmento para evitar que el LLM congele el proceso
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(chain.invoke, {"text": doc.page_content})
            try:
                raw_output = future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                print(f"\r    ⚠ Fragmento {i+1}/{total_docs} saltado (timeout).         ", flush=True)
                executor.shutdown(wait=False)
                # Marcar como procesado para no reintentar en futuras ejecuciones
                processed_indices.add(i)
                _save_checkpoint(processed_indices)
                continue
            finally:
                executor.shutdown(wait=False)
            
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
                                
            _log.info(f"Ok ({tripletas_extraidas} tripletas extraídas e ingestadas en KùzuDB)")
            
            # Marcar como procesado y guardar checkpoint
            processed_indices.add(i)
            _save_checkpoint(processed_indices)
                
        except Exception as e:
            _log.error(f"Error: {e}")
        
    print(f"\n[+] Grafo construido con {aristas_creadas} nuevas aristas.")
    _log.info(f"Grafo construido en disco con {aristas_creadas} nuevas aristas.")
    
    # Limpiar checkpoint al terminar exitosamente
    _clear_checkpoint()
    
    return db
