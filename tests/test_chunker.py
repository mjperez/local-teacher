import pytest
from langchain_core.documents import Document
from local_teacher.ingestion.chunker import dividir_texto

def test_dividir_texto_markdown():
    docs = [
        Document(
            page_content="# Titulo\n\n## Subtitulo\n\nEste es el contenido bajo subtitulo.",
            metadata={"tipo_archivo": "markdown", "fuente": "test.md"}
        )
    ]
    chunks = dividir_texto(docs, chunk_size=100, chunk_overlap=10)
    
    assert len(chunks) > 0
    assert "encabezado_1" in chunks[0].metadata
    assert chunks[0].metadata["encabezado_1"] == "Titulo"
    assert chunks[0].metadata["encabezado_2"] == "Subtitulo"

def test_dividir_texto_plano():
    docs = [
        Document(
            page_content="Este es un texto largo " * 50,
            metadata={"tipo_archivo": "pdf", "fuente": "test.pdf"}
        )
    ]
    chunks = dividir_texto(docs, chunk_size=200, chunk_overlap=20)
    
    assert len(chunks) > 1
    # Check overlapping sizes roughly
    assert len(chunks[0].page_content) <= 200
