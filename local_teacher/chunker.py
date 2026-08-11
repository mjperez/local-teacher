from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def dividir_texto(
    documentos: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Divide documentos en fragmentos (chunks) para indexar.

    - Los archivos **Markdown** (``.md``/``.markdown``) se dividen primero por
      jerarquía de headers (``#``/``##``) para conservar la estructura.
    - El resto (incluido el texto plano que sale de los PDFs) NO tiene
      headers Markdown, así que se divide directamente por tamaño con un
      splitter general (por párrafos, frases y palabras).
    """
    if not documentos:
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )
    general_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[Document] = []
    for documento in documentos:
        source = str(documento.metadata.get("source", "")).lower()
        if source.endswith((".md", ".markdown")):
            partes = header_splitter.split_text(documento.page_content)
        else:
            partes = [documento]

        for parte in partes:
            # conserva el metadata original (source) junto con la jerarquía
            parte.metadata = {**documento.metadata, **parte.metadata}
            chunks.extend(general_splitter.split_documents([parte]))

    return chunks