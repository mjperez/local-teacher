import logging
import torch
from typing import Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from local_teacher.graph import MemgraphClient, get_memgraph_client
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
        _log.info("Cargando modelo GLiNER en %s...", device.upper())
        print(f"[*] Cargando modelo GLiNER en {device.upper()} (GPU)..." if device == "cuda" else "[*] Cargando modelo GLiNER en CPU...")
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to(device)
    return _gliner_model


def build_knowledge_graph(
    docs: list[Document],
    llm: Optional[BaseChatModel] = None,
    client: Optional[MemgraphClient] = None,
) -> MemgraphClient:
    """
    Extrae un Grafo de Co-ocurrencia de Entidades usando GLiNER e inserciones UNWIND en Memgraph.
    
    Aplica pesos incrementales a las aristas y ejecuta detección de comunidades (Louvain)
    con MAGE al finalizar. Soporta reanudación desde checkpoint.
    """
    _log.info("Iniciando extracción de Grafo de Conocimiento (GraphRAG) para %d fragmentos...", len(docs))
    
    GLINER_BATCH_SIZE = 64

    memgraph = client or get_memgraph_client()
    memgraph.ensure_schema()
    
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
    
    for batch_start in range(0, total_docs, GLINER_BATCH_SIZE):
        batch_end = min(batch_start + GLINER_BATCH_SIZE, total_docs)
        
        batch_indices = []
        batch_texts = []
        for i in range(batch_start, batch_end):
            if i not in processed_indices:
                batch_indices.append(i)
                batch_texts.append(docs[i].page_content)
        
        if not batch_texts:
            continue

        try:
            if hasattr(model, "batch_predict_entities"):
                all_entities = model.batch_predict_entities(batch_texts, labels, threshold=0.5)
            elif hasattr(model, "inference"):
                all_entities = model.inference(batch_texts, labels, threshold=0.5)
            elif hasattr(model, "predict_entities"):
                all_entities = [model.predict_entities(text, labels, threshold=0.5) for text in batch_texts]
            else:
                _log.error("GLiNER model no soporta extracción de entidades.")
                continue

            if len(all_entities) != len(batch_texts):
                _log.error("Inconsistencia en inferencia: esperadas %d salidas, obtenidas %d.", len(batch_texts), len(all_entities))
                continue

            all_batch_nodes = set()
            all_batch_edges = []

            for idx, entities, text in zip(batch_indices, all_entities, batch_texts):
                nombres = set()
                for e in entities:
                    ent_text = e["text"]
                    if _is_valid_entity(ent_text):
                        nombres.add(ent_text.strip().title())
                nombres_list = list(nombres)
                
                # Generar pares de co-ocurrencia
                for j in range(len(nombres_list)):
                    for k in range(j + 1, len(nombres_list)):
                        all_batch_nodes.add(nombres_list[j])
                        all_batch_nodes.add(nombres_list[k])
                        all_batch_edges.append({
                            "source": nombres_list[j],
                            "target": nombres_list[k],
                        })

            try:
                if all_batch_nodes:
                    memgraph.execute_write(
                        "UNWIND $nodes AS node_name "
                        "MERGE (a:Entity {name: node_name})",
                        {"nodes": list(all_batch_nodes)},
                    )

                if all_batch_edges:
                    memgraph.execute_write(
                        "UNWIND $edges AS e "
                        "MATCH (a:Entity {name: e.source}), (b:Entity {name: e.target}) "
                        "MERGE (a)-[r:Rel {type: 'CO_OCCURS_WITH'}]->(b) "
                        "ON CREATE SET r.weight = 1 "
                        "ON MATCH SET r.weight = r.weight + 1",
                        {"edges": all_batch_edges},
                    )
                    aristas_creadas += len(all_batch_edges)

                processed_indices.update(batch_indices)
                sm.mark_graph_chunks_batch(batch_indices)

            except Exception as db_err:
                _log.error("Error al insertar lote en Memgraph: %s", db_err)

            print(f"\r    - {len(processed_indices)}/{total_docs} fragmentos completados...", end="", flush=True)
                
        except Exception as e:
            _log.error("Error procesando lote %d-%d: %s", batch_start, batch_end, e)

    print(f"\n[+] Grafo en Memgraph actualizado con {aristas_creadas} aristas de co-ocurrencia.")
    _log.info("Grafo en Memgraph actualizado con %d aristas.", aristas_creadas)
    
    # Ejecutar detección de comunidades al terminar
    memgraph.run_louvain()
    
    return memgraph
