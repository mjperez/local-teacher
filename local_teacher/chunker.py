from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def dividir_texto(documentos: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    if not documentos:
        return []
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documentos)