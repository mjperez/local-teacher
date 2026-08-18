import json
import logging
import re
from pathlib import Path
from langchain_core.documents import Document

from local_teacher.ingestion.loaders.youtube import extraer_transcripcion_youtube

_log = logging.getLogger(__name__)

def cargar_jsonl(ruta: Path, curso: str | None = None) -> list[Document]:
    """Carga un .jsonl con formato Natural Questions."""
    docs = []
    with open(ruta, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            if "page_content" in data and "metadata" in data:
                docs.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
                continue
                
            if "question" in data and "answer" in data:
                resp = (
                    ", ".join(data["answer"])
                    if isinstance(data["answer"], list)
                    else str(data["answer"])
                )
                text = f"Q: {data['question']}\nA: {resp}"
            else:
                text = json.dumps(data, ensure_ascii=False)

            meta = {"fuente": str(ruta), "tipo_archivo": "jsonl"}
            if curso:
                meta["curso"] = curso
            docs.append(Document(page_content=text, metadata=meta))
    return docs

def procesar_texto_plano(item: Path, curso: str | None = None) -> list[Document]:
    ext = item.suffix.lower()
    tipo = "markdown" if ext == ".md" else "texto"
    nuevos_docs = []
    try:
        contenido = item.read_text(encoding="utf-8", errors="replace")
        meta = {"fuente": str(item), "tipo_archivo": tipo}
        if curso:
            meta["curso"] = curso
            
        # Extraer enlaces de YouTube del contenido
        youtube_urls = re.findall(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[0-9A-Za-z_-]{11}[^\s]*)", contenido)
        for y_url in set(youtube_urls):
            transcripcion = extraer_transcripcion_youtube(y_url)
            if transcripcion:
                y_meta = {"fuente": y_url, "tipo_archivo": "youtube"}
                if curso:
                    y_meta["curso"] = curso
                nuevos_docs.append(Document(page_content=f"Transcripción de video ({y_url}):\n{transcripcion}", metadata=y_meta))
        
        nuevos_docs.append(Document(page_content=contenido, metadata=meta))
    except Exception as exc:
        _log.error("Error leyendo archivo %s: %s", item.name, exc)
    return nuevos_docs
