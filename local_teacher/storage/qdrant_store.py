import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore


def get_qdrant_store(
    embeddings: Embeddings,
    documentos: list[Document] | None = None,
    qdrant_url: str | None = None,
) -> QdrantVectorStore:
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = "test"

    if documentos:
        # force_recreate=True elimina la colección anterior y usa la
        # dimensión real del embedding (768 con Nomic, 1536 con OpenAI).
        return QdrantVectorStore.from_documents(
            documents=documentos,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name,
            force_recreate=True,
        )

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=qdrant_url,
        collection_name=collection_name,
    )