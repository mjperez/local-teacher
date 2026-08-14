import os
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

def get_qdrant_store(
    embeddings: Embeddings,
    documentos: list[Document] | None = None,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    force_recreate: bool = False,
) -> QdrantVectorStore:
    """Conecta a Qdrant con Búsqueda Híbrida y opcionalmente ingesta documentos."""
    url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = collection_name or os.getenv("QDRANT_COLLECTION", "local_teacher")
    
    # Inicializar el modelo BM25 (sparse) localmente para búsqueda híbrida exacta
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    if documentos:
        import time
        print(f"\n[*] Ingestando {len(documentos)} documentos en lotes (evitando error WSAENOBUFS en Windows)...")
        batch_size = 50
        
        # El primer lote inicializa y recrea la colección (Qdrant necesita al menos 1 doc para saber el tamaño del vector)
        primer_lote = documentos[:batch_size]
        store = QdrantVectorStore.from_documents(
            documents=primer_lote,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            url=url,
            collection_name=collection,
            force_recreate=force_recreate,
        )
        print(f"    - Lote 1/{ (len(documentos) + batch_size - 1) // batch_size } completado.")
        
        # Ingestamos el resto en lotes con pausas para que Windows libere sockets
        for i in range(batch_size, len(documentos), batch_size):
            lote = documentos[i : i + batch_size]
            store.add_documents(lote)
            print(f"    - Lote { (i // batch_size) + 1 }/{ (len(documentos) + batch_size - 1) // batch_size } completado.")
            time.sleep(0.5)
            
        return store

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=url,
        collection_name=collection,
    )
