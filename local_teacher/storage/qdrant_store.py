import os
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

def get_qdrant_store(
    embeddings: Embeddings,
    documentos: list[Document] | None = None,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    force_recreate: bool = False,
) -> QdrantVectorStore:
    """Conecta a Qdrant y opcionalmente ingesta documentos."""
    url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = collection_name or os.getenv("QDRANT_COLLECTION", "test")
    
    if documentos:
        return QdrantVectorStore.from_documents(
            documents=documentos,
            embedding=embeddings,
            url=url,
            collection_name=collection,
            force_recreate=force_recreate,
        )

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=url,
        collection_name=collection,
    )