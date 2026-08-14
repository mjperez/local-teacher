import os
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def _doc_serializer(doc: Document) -> bytes:
    return json.dumps(doc.dict()).encode("utf-8")

def _doc_deserializer(b: bytes) -> Document:
    return Document(**json.loads(b.decode("utf-8")))


def get_qdrant_retriever(
    embeddings: Embeddings,
    documentos: list[Document] | None = None,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    force_recreate: bool = False,
) -> ParentDocumentRetriever:
    """Conecta a Qdrant y devuelve un ParentDocumentRetriever."""
    url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = collection_name or os.getenv("QDRANT_COLLECTION", "local_teacher")

    # Inicializar el modelo BM25 (sparse) localmente para búsqueda híbrida exacta
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Almacenamiento local para los documentos padre (en disco para que sea persistente y no requiera Redis forzosamente)
    fs = LocalFileStore("./.local_teacher_parents")
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=_doc_serializer,
        value_deserializer=_doc_deserializer
    )

    # El child_splitter corta los padres (1500 chars) en trozos pequeños para Qdrant (300 chars)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    if documentos and force_recreate:
        # Truco para forzar la recreación de la colección en Qdrant con las configuraciones correctas
        vectorstore = QdrantVectorStore.from_documents(
            documents=[Document(page_content="Init collection")],
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            url=url,
            collection_name=collection,
            force_recreate=True,
        )
    else:
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            url=url,
            collection_name=collection,
        )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=None  # Los documentos de entrada YA son los padres
    )

    if documentos:
        print(f"[*] Ingestando {len(documentos)} documentos jerárquicos en lotes...")
        batch_size = 50

        for i in range(0, len(documentos), batch_size):
            lote = documentos[i : i + batch_size]
            retriever.add_documents(lote, ids=None)
            print(
                f"    - Lote { (i // batch_size) + 1 }/{ (len(documentos) + batch_size - 1) // batch_size } completado."
            )
            time.sleep(0.5)

    return retriever
