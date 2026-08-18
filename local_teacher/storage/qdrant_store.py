import os
from tenacity import retry, wait_exponential, stop_after_attempt

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
        from pathlib import Path
        checkpoint_path = Path("./.qdrant_checkpoint.json")
        processed_batches = set()
        
        if force_recreate:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        elif checkpoint_path.exists():
            try:
                processed_batches = set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("processed_batches", []))
                if processed_batches:
                    print(f"[*] Checkpoint encontrado: {len(processed_batches)} lotes ya ingestados en Qdrant. Reanudando...")
            except Exception:
                pass

        print(f"[*] Ingestando {len(documentos)} documentos jerárquicos en lotes...")
        batch_size = 50
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
            processed_batches.add(lote_num)
            try:
                checkpoint_path.write_text(
                    json.dumps({"processed_batches": list(processed_batches)}, ensure_ascii=False),
                    encoding="utf-8"
                )
            except Exception:
                pass
                
            print(f"\r    - Lote {lote_num}/{total_lotes} completado.", end="", flush=True)
            
        # Limpiar checkpoint al terminar
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
        print()  # salto de línea al terminar

    return retriever
