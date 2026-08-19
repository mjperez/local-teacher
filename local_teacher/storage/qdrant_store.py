import os
from tenacity import retry, wait_exponential, stop_after_attempt

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import logging

_log = logging.getLogger(__name__)

def _doc_serializer(doc: Document) -> bytes:
    data = {"page_content": doc.page_content, "metadata": doc.metadata}
    return json.dumps(data).encode("utf-8")

def _doc_deserializer(b: bytes) -> Document:
    data = json.loads(b.decode("utf-8"))
    return Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {}))


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

    # El child_splitter corta los padres (1500 chars) en trozos para Qdrant (500 chars)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

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
        from local_teacher.ingestion.state_manager import get_state_manager
        sm = get_state_manager()
        
        if force_recreate:
            sm.clear()
            processed_batches = set()
        else:
            processed_batches = sm.get_qdrant_batches()
            if processed_batches:
                print(f"[*] Checkpoint encontrado: {len(processed_batches)} lotes ya ingestados en Qdrant. Reanudando...")

        print(f"[*] Ingestando {len(documentos)} documentos jerárquicos en lotes...")
        try:
            batch_size = max(1, int(os.getenv("QDRANT_BATCH_SIZE", "100")))
        except (ValueError, TypeError):
            _log.warning("QDRANT_BATCH_SIZE tiene un valor inválido; usando 100 por defecto.")
            batch_size = 100
        total_lotes = (len(documentos) + batch_size - 1) // batch_size
        saltados = len(processed_batches)
        if saltados:
            print(f"    ({saltados} ya procesados, {total_lotes - saltados} pendientes)")

        @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
        def _ingestar_lote_seguro(lote_docs):
            retriever.add_documents(lote_docs, ids=None)

        for i in range(0, len(documentos), batch_size):
            lote_num = (i // batch_size) + 1
            if lote_num in processed_batches:
                continue

            lote = documentos[i : i + batch_size]
            _ingestar_lote_seguro(lote)
            
            # Guardar checkpoint
            sm.mark_qdrant_batch(lote_num)
                
            print(f"\r    - Lote {lote_num}/{total_lotes} completado.", end="", flush=True)
            
        print()  # salto de línea al terminar

    return retriever


# Alias para compatibilidad
get_qdrant_store = get_qdrant_retriever
