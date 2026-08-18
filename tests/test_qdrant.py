import socket
import pytest
from langchain_core.documents import Document
from local_teacher.storage.qdrant_store import get_qdrant_store


def _qdrant_disponible() -> bool:
    try:
        s = socket.create_connection(("localhost", 6333), timeout=1.0)
        s.close()
        return True
    except Exception:
        return False


class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t) % 10), 0.5] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 10), 0.5]


@pytest.mark.skipif(not _qdrant_disponible(), reason="Requiere servidor Qdrant activo en localhost:6333")
def test_qdrant_store_integration():
    docs = [
        Document(page_content="El cielo es azul brillante y despejado", metadata={"id": 1}),
        Document(page_content="El sol es amarillo en la tarde", metadata={"id": 2}),
    ]

    store = get_qdrant_store(
        embeddings=DummyEmbeddings(),
        documentos=docs,
        collection_name="test_collection",
        force_recreate=True,
    )

    assert store is not None

    res = store.invoke("cielo")
    assert len(res) > 0
    assert any("cielo" in d.page_content for d in res)
