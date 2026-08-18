import json
import logging
import hashlib
from pathlib import Path
from langchain_core.documents import Document
import concurrent.futures

from local_teacher.ingestion.state_manager import get_state_manager
from local_teacher.ingestion.loaders.document import cargar_docling, _limpiar_formulas
from local_teacher.ingestion.loaders.media import cargar_video_local
from local_teacher.ingestion.loaders.text import cargar_jsonl, procesar_texto_plano

_log = logging.getLogger(__name__)


def _calcular_hash(ruta_archivo: Path) -> str:
    """Calcula el SHA-256 de un archivo leyendo por bloques."""
    hasher = hashlib.sha256()
    try:
        with open(ruta_archivo, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        _log.warning("No se pudo calcular hash para %s: %s", ruta_archivo, e)
        return str(ruta_archivo)


def _procesar_un_archivo(args: tuple) -> tuple[Path, list[Document]]:
    """Procesa un archivo individual según su extensión."""
    (
        item,
        ruta_carpeta,
        extraer_figuras,
        extraer_tablas,
        carpeta_figuras,
        enriquecer_formulas,
    ) = args

    curso = None
    if isinstance(ruta_carpeta, Path) and ruta_carpeta.is_dir():
        curso = None if item.parent == ruta_carpeta else item.parent.name

    ext = item.suffix.lower()
    nuevos_docs = []

    if ext in (".txt", ".md"):
        nuevos_docs.extend(procesar_texto_plano(item, curso=curso))
    elif ext in (".pdf", ".docx", ".pptx"):
        nuevos_docs.extend(
            cargar_docling(
                item,
                extraer_figuras,
                extraer_tablas,
                carpeta_figuras,
                enriquecer_formulas,
                curso=curso,
            )
        )
    elif ext in (".mp4", ".mkv"):
        nuevos_docs.extend(cargar_video_local(item, curso=curso))
    elif ext == ".jsonl":
        nuevos_docs.extend(cargar_jsonl(item, curso=curso))

    return item, nuevos_docs


def cargar_archivos(
    ruta_carpeta: Path | str,
    extraer_figuras: bool = False,
    extraer_tablas: bool = False,
    carpeta_figuras: Path | str | None = None,
    enriquecer_formulas: bool = True,
    max_workers: int = 1,
) -> list[Document]:
    """Carga y procesa archivos de una ruta convirtiéndolos en documentos de LangChain.

    Formatos soportados: .txt, .md, .pdf, .docx, .pptx, .jsonl, .mp4, .mkv
    """
    ruta_carpeta = Path(ruta_carpeta)
    if not ruta_carpeta.exists():
        raise FileNotFoundError(f"No existe la ruta de ingesta: {ruta_carpeta}")

    docs: list[Document] = []
    items = [ruta_carpeta] if ruta_carpeta.is_file() else list(ruta_carpeta.rglob("*"))

    cache_path = Path(str(ruta_carpeta)).with_suffix(".jsonl")
    sm = get_state_manager()
    archivos_procesados = sm.get_processed_files()

    items_a_procesar = []
    item_hashes = {}

    for item in items:
        if not item.is_file() or item.name.endswith("_toc.txt"):
            continue

        file_hash = _calcular_hash(item)
        if file_hash in archivos_procesados:
            continue

        items_a_procesar.append(item)
        item_hashes[item] = file_hash

    if items_a_procesar:
        print(f"[*] Extrayendo {len(items_a_procesar)} archivos pendientes...")

        args_list = [
            (
                item,
                ruta_carpeta,
                extraer_figuras,
                extraer_tablas,
                carpeta_figuras,
                enriquecer_formulas,
            )
            for item in items_a_procesar
        ]

        workers_reales = max(1, int(max_workers))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_reales) as executor:
            for item, nuevos_docs in executor.map(_procesar_un_archivo, args_list):
                if nuevos_docs:
                    docs.extend(nuevos_docs)
                    try:
                        with open(cache_path, "a", encoding="utf-8") as f:
                            for d in nuevos_docs:
                                json.dump(
                                    {"page_content": d.page_content, "metadata": d.metadata},
                                    f,
                                    ensure_ascii=False,
                                )
                                f.write("\n")

                        file_hash = item_hashes[item]
                        sm.mark_file_processed(file_hash)
                    except Exception as e:
                        _log.warning("No se pudo guardar el progreso de %s: %s", item.name, e)

    if archivos_procesados and cache_path.exists():
        return cargar_jsonl(cache_path)

    return docs


def guardar_cache_jsonl(docs: list[Document], ruta_original: Path | str) -> None:
    """Guarda la lista de documentos en formato JSONL como caché local."""
    cache_path = Path(ruta_original).with_suffix(".jsonl")
    if str(ruta_original).endswith(".jsonl") or cache_path.exists():
        return

    _log.info("Guardando caché en %s...", cache_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        for d in docs:
            json.dump(
                {"page_content": d.page_content, "metadata": d.metadata},
                f,
                ensure_ascii=False,
            )
            f.write("\n")
