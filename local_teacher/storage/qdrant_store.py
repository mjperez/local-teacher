# aquí irá la implementación concreta de Qdrant local.
# esta capa solo sabe guardar y buscar chunks.
# Nota: si se cambia de vector DB, esto es lo unico que cambia.
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def get_qdrant_store(embeddings: Embeddings, documentos: list[Document] | None = None, qdrant_url: str = "http://localhost:6333") -> QdrantVectorStore:
    collection_name = "test"
    
    if documentos:
        return QdrantVectorStore.from_documents(
            documents=documentos,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name,
            force_recreate = True # TODO: Eliminar force_recreate al implementar actualizacion incremental
        )
    else:
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name
        )