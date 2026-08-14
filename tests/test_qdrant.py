import pytest
import os
from langchain_core.documents import Document
from local_teacher.storage.qdrant_store import get_qdrant_store

# Mock embeddings simple
class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]
    def embed_query(self, text):
        return [0.1, 0.2]

@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="Requiere Qdrant local")
def test_qdrant_store_integration():
    docs = [
        Document(page_content="El cielo es azul", metadata={"id": 1}),
        Document(page_content="El sol es amarillo", metadata={"id": 2})
    ]
    
    # Usamos colección temporal
    store = get_qdrant_store(
        embeddings=DummyEmbeddings(),
        documentos=docs,
        collection_name="test_collection",
        force_recreate=True
    )
    
    assert store is not None
    
    # Recuperación híbrida básica (algunos modelos sparse bajan localmente al ejecutar)
    res = store.similarity_search("cielo", k=1)
    assert len(res) > 0
    assert "cielo" in res[0].page_content
