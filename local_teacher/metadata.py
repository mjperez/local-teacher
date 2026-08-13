from typing import TypedDict, Optional, List, Tuple

class CoreMetadata(TypedDict, total=False):
    fuente: str           # Ruta absoluta o relativa del archivo original
    titulo: Optional[str] # Título real del documento (extraído de metadatos o filename)
    tipo_archivo: str     # 'pdf', 'markdown', 'texto', 'jsonl', 'figura', 'tabla'
    
    # Específico del contexto jerárquico/educativo
    curso: Optional[str]  # Extraído de la carpeta padre si es posible
    
    # Específico de documentos paginados (PDF)
    pagina: Optional[int]
    page_map: Optional[List[Tuple[int, int]]] # Temporal: [(start_index, pagina), ...]
    
    # Específico de jerarquías (Markdown)
    encabezado_1: Optional[str]
    encabezado_2: Optional[str]
    encabezado_3: Optional[str]
    ruta_seccion: Optional[str] # Jerarquía completa (ej: "Capítulo 1 > 1.1 Intro")
    
    # Específico de figuras/tablas
    leyenda: Optional[str]      # caption
    ruta_recurso: Optional[str] # Dónde quedó guardada la imagen o el CSV
    
    # Para control interno del chunker
    chunk_index: Optional[int]
