from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter


def dividir_texto(documentos: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    if not documentos:
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )
    markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for documento in documentos:
        header_chunks = header_splitter.split_text(documento.page_content)
        for chunk in header_chunks:
            # conserva el metadata original (source) junto con la jerarquía de headers
            chunk.metadata = {**documento.metadata, **chunk.metadata}
            chunks.extend(markdown_splitter.split_documents([chunk]))

    return chunks