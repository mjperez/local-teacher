import json
import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

def _calcular_hash(ruta_archivo: Path) -> str:
    """Calcula el SHA-256 de un archivo leyendo por fragmentos."""
    hasher = hashlib.sha256()
    try:
        with open(ruta_archivo, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        _log.warning(f"No se pudo calcular hash para {ruta_archivo}: {e}")
        # Retorna el nombre si falla, como fallback
        return str(ruta_archivo)

# Hacer que Docling muestre progreso en la consola (movido a CLI u otro sitio)

_log = logging.getLogger(__name__)
_docling_converters: dict[bool, Any] = {}

# Secuencias de glifos math corrompidos (CM, AMS, STIX…) generados por Docling sin VLM
_MATH_CHARS = (
    "\u2200-\u22ff"  # Mathematical Operators
    "\u27c0-\u27ef"  # Misc Math Symbols-A
    "\u2980-\u29ff"  # Misc Math Symbols-B
    "\u2a00-\u2aff"  # Supplemental Math Operators
    "\u2300-\u23ff"  # Misc Technical
    "\ufb00-\ufb06"  # Ligatures (ff, fi, fl)
    "\u0300-\u036f"  # Combining Diacriticals
    "\u2190-\u21ff"  # Arrows
)
_RE_GARBLED_MATH = re.compile(
    rf"[{_MATH_CHARS}]{{2,}}(?:[\s/\\(){{}}\[\]^_|]?[{_MATH_CHARS}]{{1,}})*"
)

# ---------------------------------------------------------------------------
# Limpieza de fórmulas
# ---------------------------------------------------------------------------


def _limpiar_formulas(texto: str) -> str:
    """Limpia artefactos de fórmulas mal parseadas por Docling.

    Orden de operaciones:
      1. Bloques LaTeX vacíos / fallidos  → placeholder.
      2. Comandos LaTeX con espacios entre letras  → colapsados.
      3. Caracteres de control y zero-width del backend PDF  → eliminados.
      4. Secuencias de glifos math-font corrompidos  → placeholder.
    """
    if not texto:
        return texto

    # 1 — bloques vacíos y marcadores de fallo de Docling
    texto = texto.replace("$$$$", "[fórmula]")
    texto = texto.replace("<!-- formula-not-decoded -->", "[fórmula no decodificada]")
    texto = re.sub(r"\$\$\s*\$\$", "[fórmula]", texto)

    # 2 — "\f r a c" → "\frac"  (el modelo VLM a veces espacía las letras)
    def _colapsar(m: re.Match) -> str:
        return "\\" + m.group(0).replace(" ", "").lstrip("\\")

    texto = re.sub(r"\\([a-zA-Z])(?:\s+([a-zA-Z]))+", _colapsar, texto)

    # 3 — control chars, zero-width, soft-hyphen, BOM
    texto = re.sub(
        "[\u0000-\u0008\u000b\u000c\u000e-\u001f"
        "\u007f-\u009f"
        "\u200b-\u200f\u2028\u2029"
        "\u00ad\ufeff]",
        "",
        texto,
    )

    # 4 — secuencias de glifos math corrompidos (CM, AMS, STIX…)
    texto = _RE_GARBLED_MATH.sub("[fórmula]", texto)

    # colapsar placeholders repetidos
    texto = re.sub(r"(\[fórmula(?:\s+no decodificada)?\]\s*)+", "[fórmula] ", texto)

    return texto.strip()


# ---------------------------------------------------------------------------
# Converter Docling (lazy singleton por configuración)
# ---------------------------------------------------------------------------


def _get_docling_converter(enriquecer_formulas: bool, extraer_tablas: bool):
    """Devuelve un DocumentConverter de Docling, creándolo si no existe."""
    key = (enriquecer_formulas, extraer_tablas)
    if key not in _docling_converters:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.generate_picture_images = True
        opts.generate_page_images = False
        opts.images_scale = 2.0
        opts.do_ocr = False
        opts.do_table_structure = extraer_tablas
        opts.do_formula_enrichment = enriquecer_formulas
        opts.layout_options.engine_options.compile_model = False

        # Forzar el uso de la GPU (CUDA) si está disponible, sino CPU
        opts.accelerator_options = AcceleratorOptions(
            num_threads=1, device=AcceleratorDevice.AUTO
        )

        _docling_converters[key] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _docling_converters[key]


# ---------------------------------------------------------------------------
# Extracción de figuras y tablas
# ---------------------------------------------------------------------------


def _carpeta_destino(ruta_pdf: Path, carpeta: Path | str | None) -> Path:
    """Resuelve la carpeta donde guardar figuras/tablas de un PDF."""
    base = Path(carpeta) if carpeta is not None else ruta_pdf.parent / "figuras"
    return base / ruta_pdf.stem


def _caption_docling(doc, picture) -> str:
    """Devuelve el caption asociado por Docling o el más cercano en la página."""
    caption = (picture.caption_text(doc) or "").strip()
    if caption or not picture.prov:
        return caption

    page = picture.prov[0].page_no
    box = picture.prov[0].bbox
    patron = re.compile(r"(?:figure|figura|fig\.?|fig\s+)\s*\d", re.I)

    candidatos = []
    for text in doc.texts:
        if not text.prov or text.prov[0].page_no != page:
            continue
        tb = text.prov[0].bbox
        alineado = tb.r >= box.l - 35 and tb.l <= box.r + 35
        dist = max(0.0, tb.t - box.b, box.t - tb.b)
        if alineado and dist <= 80 and patron.match(text.text.strip()):
            candidatos.append((dist, text.text.strip()))

    return min(candidatos, key=lambda c: c[0])[1] if candidatos else ""


def _extraer_figuras(ruta_pdf: Path, doc, destino: Path) -> list[dict]:
    """Exporta figuras con caption y escribe un manifest JSON."""
    destino.mkdir(parents=True, exist_ok=True)
    figuras = []

    for i, pic in enumerate(doc.pictures, 1):
        caption = _caption_docling(doc, pic)
        imagen = pic.get_image(doc)
        if not caption or imagen is None:
            continue
        pagina = pic.prov[0].page_no if pic.prov else 0
        ruta = destino / f"fig_p{pagina:03d}_n{i:03d}.png"
        imagen.save(str(ruta), "PNG")
        figuras.append(
            {"indice": i, "pagina": pagina, "caption": caption, "ruta": str(ruta)}
        )

    if figuras:
        try:
            (destino / "captions.json").write_text(
                json.dumps(figuras, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            _log.warning("No se pudo escribir captions.json: %s", exc)

    return figuras


def _extraer_tablas(doc, destino: Path) -> list[dict]:
    """Exporta tablas como Markdown (con fórmulas limpias) y CSV."""
    destino.mkdir(parents=True, exist_ok=True)
    tablas = []

    for i, table in enumerate(doc.tables, 1):
        try:
            md = _limpiar_formulas(table.export_to_markdown(doc))
            df = table.export_to_dataframe(doc)
        except Exception:
            continue
        if not md or not md.strip():
            continue

        base = destino / f"tabla_{i:02d}"
        base.with_suffix(".md").write_text(md, encoding="utf-8")
        if df is not None and not df.empty:
            df.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8")

        tablas.append(
            {
                "indice": i,
                "ruta_markdown": str(base.with_suffix(".md")),
                "ruta_csv": str(base.with_suffix(".csv")),
                "filas": len(df) if df is not None else 0,
                "columnas": len(df.columns) if df is not None else 0,
            }
        )

    return tablas


# ---------------------------------------------------------------------------
# Conversión a documentos LangChain
# ---------------------------------------------------------------------------


def _docs_figuras(
    figuras: list[dict], fuente: Path, curso: str | None = None
) -> list[Document]:
    """Convierte figuras exportadas en Documents para el RAG."""
    docs = []
    for fig in figuras:
        meta = {
            "fuente": str(fuente),
            "tipo_archivo": "figura",
            "pagina": fig["pagina"],
            "ruta_recurso": fig["ruta"],
            "leyenda": fig["caption"],
        }
        if curso:
            meta["curso"] = curso
        docs.append(
            Document(
                page_content=f"{fig['caption']}\nImagen: {fig['ruta']}", metadata=meta
            )
        )
    return docs


def _docs_tablas(
    tablas: list[dict], fuente: Path, curso: str | None = None
) -> list[Document]:
    """Convierte tablas exportadas en Documents para el RAG."""
    docs = []
    for t in tablas:
        meta = {
            "fuente": str(fuente),
            "tipo_archivo": "tabla",
            "ruta_recurso": t["ruta_markdown"],
        }
        if curso:
            meta["curso"] = curso
        docs.append(
            Document(
                page_content=f"Tabla:\n{Path(t['ruta_markdown']).read_text(encoding='utf-8')}",
                metadata=meta,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Cargadores por formato
# ---------------------------------------------------------------------------


def _extraer_titulo_archivo(ruta: Path) -> str | None:
    """Intenta extraer el título interno del archivo si es un PDF."""
    if ruta.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(ruta))
        if reader.metadata and reader.metadata.title:
            t = reader.metadata.title.strip()
            # A veces los metadatos son basura generada por word/latex
            if t and not t.lower().startswith("microsoft"):
                return t
    except Exception:
        pass
    return None


def _cargar_docling(
    ruta: Path,
    extraer_figuras: bool,
    extraer_tablas: bool,
    carpeta_figuras,
    enriquecer_formulas: bool = True,
    curso: str | None = None,
) -> list[Document]:
    """Carga un documento (PDF, DOCX, PPTX) con Docling y devuelve texto, figuras y tablas usando 2 pasadas si hay fórmulas."""
    try:
        from docling.chunking import HierarchicalChunker
        from docling.datamodel.document import TextItem

        _log.info("Iniciando pasada rápida de Docling...")
        # 1. Pasada Rápida (sin VLM de fórmulas, tablas solo si extraer_tablas=True)
        doc_fast = (
            _get_docling_converter(False, extraer_tablas).convert(str(ruta)).document
        )

        paginas_con_formulas = set()
        if enriquecer_formulas:
            for item, level in doc_fast.iterate_items():
                if isinstance(item, TextItem) and item.text:
                    if _RE_GARBLED_MATH.search(item.text):
                        if item.prov and len(item.prov) > 0:
                            paginas_con_formulas.add(item.prov[0].page_no)

        chunker = HierarchicalChunker()
        textos_por_pagina = {}

        # Agrupar chunks rápidos por página
        for c in chunker.chunk(doc_fast):
            p = (
                c.meta.doc_items[0].prov[0].page_no
                if c.meta.doc_items and c.meta.doc_items[0].prov
                else 1
            )
            headings = c.meta.headings if hasattr(c.meta, "headings") else []
            if p not in textos_por_pagina:
                textos_por_pagina[p] = []
            
            chunk_text = c.text
            if headings and c.text:
                contexto = " > ".join(headings)
                # Evitar duplicar si por alguna razón ya está en el texto
                if headings[-1] not in c.text[:200]:
                    chunk_text = f"[{contexto}]\n{c.text}"
                    
            textos_por_pagina[p].append((chunk_text, headings))

        # 2. Pasada Lenta (solo si se encontraron fórmulas y está activado)
        if paginas_con_formulas:
            paginas = sorted(paginas_con_formulas)
            _log.info(
                f"Fórmulas detectadas en {len(paginas)} páginas: {paginas}. Ejecutando VLM (esto tomará un tiempo)..."
            )

            # Agrupar en rangos continuos
            rangos = []
            inicio = fin = paginas[0]
            for p in paginas[1:]:
                if p == fin + 1:
                    fin = p
                else:
                    rangos.append((inicio, fin))
                    inicio = fin = p
            rangos.append((inicio, fin))

            converter_vlm = _get_docling_converter(True, extraer_tablas)
            for inicio, fin in rangos:
                _log.info(f"  -> Procesando VLM para páginas {inicio}-{fin}...")
                doc_lento = converter_vlm.convert(
                    str(ruta), page_range=(inicio, fin)
                ).document

                reemplazados = set()
                for c in chunker.chunk(doc_lento):
                    p = (
                        c.meta.doc_items[0].prov[0].page_no
                        if c.meta.doc_items and c.meta.doc_items[0].prov
                        else inicio
                    )
                    headings = c.meta.headings if hasattr(c.meta, "headings") else []
                    if p not in reemplazados:
                        textos_por_pagina[p] = []
                        reemplazados.add(p)
                        
                    chunk_text = c.text
                    if headings and c.text:
                        contexto = " > ".join(headings)
                        if headings[-1] not in c.text[:200]:
                            chunk_text = f"[{contexto}]\n{c.text}"
                            
                    textos_por_pagina[p].append((chunk_text, headings))

        # El curso viene inyectado desde la función principal si aplica

        # Armar el markdown final, page_map y heading_map
        texto = ""
        page_map = []
        heading_map = []
        for p in sorted(textos_por_pagina.keys()):
            for chunk_texto, headings in textos_por_pagina[p]:
                # Limpiamos fórmulas por cada pedacito
                cleaned = _limpiar_formulas(chunk_texto)
                if not cleaned:
                    continue

                start_idx = len(texto)
                texto += cleaned + "\n\n"
                page_map.append((start_idx, p))
                heading_map.append((start_idx, headings))
        meta = {
            "fuente": str(ruta),
            "tipo_archivo": ruta.suffix.lower().strip("."),
            "page_map": page_map,
            "heading_map": heading_map,
        }

        titulo = _extraer_titulo_archivo(ruta)
        if titulo:
            meta["titulo"] = titulo

        if curso:
            meta["curso"] = curso

        docs: list[Document] = []

        destino = _carpeta_destino(ruta, carpeta_figuras)

        if extraer_figuras:
            figs = _extraer_figuras(ruta, doc_fast, destino)
            docs.extend(_docs_figuras(figs, ruta, curso))

        if extraer_tablas:
            tabs = _extraer_tablas(doc_fast, destino / "tablas")
            docs.extend(_docs_tablas(tabs, ruta, curso))

        if texto:
            docs.append(Document(page_content=texto, metadata=meta))
        return docs

    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        _log.error("Error procesando %s con Docling: %s", ruta.name, exc)
        return []


def _cargar_jsonl(ruta: Path, curso: str | None = None) -> list[Document]:
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


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def _extraer_transcripcion_youtube(url: str) -> str | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not match:
            return None
        video_id = match.group(1)
        
        # En versión 1.2.4, list_transcripts no existe y la forma correcta es:
        if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            
        try:
            # Intentar es, en
            transcript = transcript_list.find_transcript(['es', 'en'])
        except Exception:
            # Fallback a la primera disponible (puede ser auto-generada u otro idioma)
            transcript = next(iter(transcript_list))
            
        texto_list = transcript.fetch()
        
        # Soportar tanto dataclasses (v1.2.4) como diccionarios (otras versiones)
        if texto_list and hasattr(texto_list[0], 'text'):
            texto = " ".join([t.text for t in texto_list])
        else:
            texto = " ".join([t['text'] for t in texto_list])
            
        return texto
    except Exception as e:
        _log.warning(f"No se pudo extraer transcripción de YouTube para {url}: {e}")
def _cargar_video_local(ruta: Path, curso: str | None = None) -> list[Document]:
    import whisper
    import cv2
    import easyocr
    from difflib import SequenceMatcher
    
    docs = []
    _log.info(f"Procesando video local: {ruta.name}")
    
    # 1. Extraer audio temporal
    audio_path = ruta.with_suffix(".wav")
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", str(ruta), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Transcribir audio
        _log.info("Transcribiendo audio del video...")
        model = whisper.load_model("base")
        transcription = model.transcribe(str(audio_path), language="es")
        texto_audio = transcription["text"]
        
        meta_audio = {"fuente": str(ruta), "tipo_archivo": "video_audio"}
        if curso:
            meta_audio["curso"] = curso
        docs.append(Document(page_content=f"Transcripción de audio ({ruta.name}):\n{texto_audio}", metadata=meta_audio))
    except Exception as e:
        _log.warning(f"Error procesando audio de {ruta.name}: {e}")
    finally:
        if audio_path.exists():
            try:
                audio_path.unlink()
            except:
                pass
            
    # 3. Procesar video (frames) para OCR
    try:
        _log.info("Procesando frames del video (OCR)...")
        reader = easyocr.Reader(['es', 'en'], gpu=True)
        cap = cv2.VideoCapture(str(ruta))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30
            
        # 1 frame cada 30 segundos
        frame_interval = int(fps * 30)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        texto_visual = []
        last_text = ""
        frame_count = 0
        
        while frame_count < total_frames:
            # Saltar directamente al frame deseado en lugar de decodificar todos los intermedios
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
                
            # OCR en este frame
            result = reader.readtext(frame, detail=0)
            text = " ".join(result)
            
            if text.strip():
                # Deduplicación (80% similitud)
                similitud = SequenceMatcher(None, text, last_text).ratio()
                if similitud < 0.8:
                    minuto = (frame_count / fps) / 60
                    texto_visual.append(f"[Minuto {minuto:.1f}] Texto en pantalla: {text}")
                    last_text = text
                        
            frame_count += frame_interval
            
        cap.release()
        
        if texto_visual:
            meta_visual = {"fuente": str(ruta), "tipo_archivo": "video_visual"}
            if curso:
                meta_visual["curso"] = curso
            contenido_visual = "\n".join(texto_visual)
            docs.append(Document(page_content=f"Contenido visual (diapositivas) de {ruta.name}:\n{contenido_visual}", metadata=meta_visual))
            
    except Exception as e:
        _log.warning(f"Error procesando video visualmente {ruta.name}: {e}")
        
    return docs


def _procesar_un_archivo(args):
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
        tipo = "markdown" if ext == ".md" else "texto"
        try:
            contenido = item.read_text(encoding="utf-8", errors="replace")
            meta = {"fuente": str(item), "tipo_archivo": tipo}
            if curso:
                meta["curso"] = curso
                
            # Extraer enlaces de YouTube del contenido
            import re
            youtube_urls = re.findall(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[0-9A-Za-z_-]{11}[^\s]*)", contenido)
            for y_url in set(youtube_urls):
                transcripcion = _extraer_transcripcion_youtube(y_url)
                if transcripcion:
                    y_meta = {"fuente": y_url, "tipo_archivo": "youtube"}
                    if curso:
                        y_meta["curso"] = curso
                    nuevos_docs.append(Document(page_content=f"Transcripción de video ({y_url}):\n{transcripcion}", metadata=y_meta))
            
            nuevos_docs.append(Document(page_content=contenido, metadata=meta))
        except Exception as exc:
            _log.error("Error leyendo archivo %s: %s", item.name, exc)
    elif ext in (".pdf", ".docx", ".pptx"):
        nuevos_docs.extend(
            _cargar_docling(
                item,
                extraer_figuras,
                extraer_tablas,
                carpeta_figuras,
                enriquecer_formulas,
                curso=curso,
            )
        )
    elif ext in (".mp4", ".mkv"):
        nuevos_docs.extend(_cargar_video_local(item, curso=curso))
    elif ext == ".jsonl":
        nuevos_docs.extend(_cargar_jsonl(item, curso=curso))
        
    return item, nuevos_docs


def cargar_archivos(
    ruta_carpeta: Path | str,
    extraer_figuras: bool = False,
    extraer_tablas: bool = False,
    carpeta_figuras: Path | str | None = None,
    enriquecer_formulas: bool = True,
) -> list[Document]:
    """Carga archivos de una ruta como documentos.

    Formatos soportados: .txt, .md, .pdf, .docx, .pptx, .jsonl
    """
    ruta_carpeta = Path(ruta_carpeta)
    if not ruta_carpeta.exists():
        raise FileNotFoundError(f"No existe la ruta de ingesta: {ruta_carpeta}")

    docs: list[Document] = []
    items = [ruta_carpeta] if ruta_carpeta.is_file() else list(ruta_carpeta.rglob("*"))
    
    # Manejo de reanudación y caché
    cache_path = Path(str(ruta_carpeta)).with_suffix(".jsonl")
    checkpoint_path = Path("./.loader_checkpoint.json")
    archivos_procesados = set()
    
    # Intentar cargar progreso previo
    if checkpoint_path.exists() and cache_path.exists():
        try:
            archivos_procesados = set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("procesados", []))
            if archivos_procesados:
                print(f"[*] Checkpoint de extracción encontrado: {len(archivos_procesados)} archivos ya procesados. Reanudando...")
        except Exception:
            pass
    elif not cache_path.exists():
        # Si no hay caché, empezar de cero limpiando el checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
    items_a_procesar = []
    item_hashes = {}

    for item in items:
        if not item.is_file():
            continue
        if item.name.endswith("_toc.txt"):
            continue
            
        file_hash = _calcular_hash(item)
        if file_hash in archivos_procesados:
            continue
            
        items_a_procesar.append(item)
        item_hashes[item] = file_hash
        
    import concurrent.futures
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
        
        # Usar ProcessPoolExecutor con max_workers=1 para evitar OOM de CUDA
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
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
                        archivos_procesados.add(file_hash)
                        checkpoint_path.write_text(
                            json.dumps({"procesados": list(archivos_procesados)}, ensure_ascii=False),
                            encoding="utf-8"
                        )
                    except Exception as e:
                        _log.warning(f"No se pudo guardar el progreso de {item.name}: {e}")

    # Al terminar la carga, devolver todos los documentos (los de esta sesión + los cacheados anteriormente si existen)
    # Si acabamos de procesar todos los archivos desde 0, `docs` tiene todos los documentos.
    # Pero si reanudamos, `docs` solo tiene los nuevos. En ese caso, leemos el caché completo.
    if archivos_procesados and not items_a_procesar and cache_path.exists():
        # Todo ya estaba en caché, o se reanudó sin nuevos archivos
        return _cargar_jsonl(cache_path)
    elif archivos_procesados and items_a_procesar and cache_path.exists():
        # Mezcla de archivos reanudados + nuevos. Es más seguro simplemente cargar el caché completo que contiene todo.
        return _cargar_jsonl(cache_path)
        
    return docs


def guardar_cache_jsonl(docs: list[Document], ruta_original: Path | str) -> None:
    """Guarda en formato jsonl los documentos procesados para evitar re-parsear PDFs/Markdown pesados."""
    cache_path = Path(ruta_original).with_suffix(".jsonl")
    if str(ruta_original).endswith(".jsonl") or cache_path.exists():
        return

    _log.info(f"Guardando caché en {cache_path}... (por favor espera)")
    with open(cache_path, "w", encoding="utf-8") as f:
        for d in docs:
            json.dump(
                {"page_content": d.page_content, "metadata": d.metadata},
                f,
                ensure_ascii=False,
            )
            f.write("\n")
