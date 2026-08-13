from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

def _get_page(start_index: int, page_map: list[tuple[int, int]]) -> int | None:
    """Calcula a qué página pertenece un fragmento basado en su índice de inicio."""
    if not page_map:
        return None
    current_page = page_map[0][1]
    for idx, p in page_map:
        if start_index >= idx:
            current_page = p
        else:
            break
    return current_page


def _get_headings(start_index: int, heading_map: list[tuple[int, list[str]]]) -> list[str]:
    """Calcula a qué encabezados pertenece un fragmento basado en su índice de inicio."""
    if not heading_map:
        return []
    current_headings = heading_map[0][1]
    for idx, h in heading_map:
        if start_index >= idx:
            current_headings = h
        else:
            break
    return current_headings


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
        headers_to_split_on=[
            ("#", "encabezado_1"), 
            ("##", "encabezado_2"),
            ("###", "encabezado_3")
        ]
    )
    general_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )

    chunks: list[Document] = []
    for documento in documentos:
        # Detectar si es markdown puro u otro (ej. tabla exportada o archivo .md)
        tipo = str(documento.metadata.get("tipo_archivo", "")).lower()
        if tipo in ("markdown", "tabla"):
            partes = header_splitter.split_text(documento.page_content)
        else:
            partes = [documento]

        for parte in partes:
            # conserva el metadata original junto con la jerarquía
            parte.metadata = {**documento.metadata, **parte.metadata}
            sub_chunks = general_splitter.split_documents([parte])
            
            for i, chunk in enumerate(sub_chunks):
                chunk.metadata["chunk_index"] = i
                
                # Calcular la página si tenemos el page_map
                if "page_map" in chunk.metadata and "start_index" in chunk.metadata:
                    chunk.metadata["pagina"] = _get_page(
                        chunk.metadata["start_index"], 
                        chunk.metadata["page_map"]
                    )
                    
                # Calcular encabezados si tenemos el heading_map
                if "heading_map" in chunk.metadata and "start_index" in chunk.metadata:
                    headings = _get_headings(
                        chunk.metadata["start_index"], 
                        chunk.metadata["heading_map"]
                    )
                    if headings:
                        ruta = " > ".join(headings)
                        chunk.metadata["ruta_seccion"] = ruta
                        # Inyectar en el contenido para que el buscador vectorial y BM25 lo "lean"
                        chunk.page_content = f"[{ruta}]\n{chunk.page_content}"
                
                # Limpiar metadata temporal
                chunk.metadata.pop("page_map", None)
                chunk.metadata.pop("heading_map", None)
                chunk.metadata.pop("start_index", None)
                
                chunks.append(chunk)

    return chunks