import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

_log = logging.getLogger(__name__)
_docling_converters: dict[bool, Any] = {}

# Secuencias de glifos math corrompidos
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

def _limpiar_formulas(texto: str) -> str:
    if not texto:
        return texto
    texto = texto.replace("$$$$", "[fórmula]")
    texto = texto.replace("<!-- formula-not-decoded -->", "[fórmula no decodificada]")
    texto = re.sub(r"\$\$\s*\$\$", "[fórmula]", texto)
    def _colapsar(m: re.Match) -> str:
        return "\\" + m.group(0).replace(" ", "").lstrip("\\")
    texto = re.sub(r"\\([a-zA-Z])(?:\s+([a-zA-Z]))+", _colapsar, texto)
    texto = re.sub(
        "[\u0000-\u0008\u000b\u000c\u000e-\u001f"
        "\u007f-\u009f"
        "\u200b-\u200f\u2028\u2029"
        "\u00ad\ufeff]",
        "",
        texto,
    )
    texto = _RE_GARBLED_MATH.sub("[fórmula]", texto)
    texto = re.sub(r"(\[fórmula(?:\s+no decodificada)?\]\s*)+", "[fórmula] ", texto)
    return texto.strip()

def _get_docling_converter(enriquecer_formulas: bool, extraer_tablas: bool, extraer_figuras: bool = False):
    key = (enriquecer_formulas, extraer_tablas, extraer_figuras)
    if key not in _docling_converters:
        import torch
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.generate_picture_images = bool(extraer_figuras)
        opts.generate_page_images = False
        opts.images_scale = 2.0
        opts.do_ocr = False
        opts.do_table_structure = extraer_tablas
        opts.do_formula_enrichment = enriquecer_formulas
        opts.layout_options.engine_options.compile_model = False

        # Escalar hilos por número de workers para evitar sobresuscripción.
        try:
            workers = max(1, int(os.getenv("INGEST_WORKERS", "1")))
        except (ValueError, TypeError):
            workers = 1
        hilos_cpu = max(1, min(4, (os.cpu_count() or 1) // workers))
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.AUTO
        opts.accelerator_options = AcceleratorOptions(
            num_threads=hilos_cpu, device=device
        )

        _docling_converters[key] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _docling_converters[key]

def _carpeta_destino(ruta_pdf: Path, carpeta: Path | str | None) -> Path:
    base = Path(carpeta) if carpeta is not None else ruta_pdf.parent / "figuras"
    return base / ruta_pdf.stem

def _caption_docling(doc, picture) -> str:
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
    captions_file = destino / "captions.json"
    if captions_file.exists():
        try:
            figuras = json.loads(captions_file.read_text(encoding="utf-8"))
            if figuras and all(Path(f.get("ruta", "")).exists() for f in figuras):
                return figuras
        except Exception:
            pass

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
    if destino.exists():
        tablas_md = list(destino.glob("tabla_*.md"))
        if tablas_md:
            tablas = []
            for i, p_md in enumerate(sorted(tablas_md), 1):
                p_csv = p_md.with_suffix(".csv")
                tablas.append(
                    {
                        "indice": i,
                        "ruta_markdown": str(p_md),
                        "ruta_csv": str(p_csv) if p_csv.exists() else "",
                        "filas": 0,
                        "columnas": 0,
                    }
                )
            return tablas

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

def _docs_figuras(
    figuras: list[dict], fuente: Path, curso: str | None = None
) -> list[Document]:
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

def _extraer_titulo_archivo(ruta: Path) -> str | None:
    if ruta.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(ruta))
        if reader.metadata and reader.metadata.title:
            t = reader.metadata.title.strip()
            if t and not t.lower().startswith("microsoft"):
                return t
    except Exception:
        pass
    return None

def cargar_docling(
    ruta: Path,
    extraer_figuras: bool,
    extraer_tablas: bool,
    carpeta_figuras,
    enriquecer_formulas: bool = True,
    curso: str | None = None,
) -> list[Document]:
    try:
        from docling.chunking import HierarchicalChunker
        from docling.datamodel.document import TextItem

        _log.info("Iniciando pasada rápida de Docling...")
        doc_fast = (
            _get_docling_converter(False, extraer_tablas, extraer_figuras).convert(str(ruta)).document
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
                if headings[-1] not in c.text[:200]:
                    chunk_text = f"[{contexto}]\n{c.text}"
                    
            textos_por_pagina[p].append((chunk_text, headings))

        if paginas_con_formulas:
            paginas = sorted(paginas_con_formulas)
            _log.info(
                f"Fórmulas detectadas en {len(paginas)} páginas: {paginas}. Ejecutando VLM..."
            )
            rangos = []
            inicio = fin = paginas[0]
            for p in paginas[1:]:
                if p == fin + 1:
                    fin = p
                else:
                    rangos.append((inicio, fin))
                    inicio = fin = p
            rangos.append((inicio, fin))

            converter_vlm = _get_docling_converter(True, extraer_tablas, extraer_figuras)
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

        texto = ""
        page_map = []
        heading_map = []
        for p in sorted(textos_por_pagina.keys()):
            for chunk_texto, headings in textos_por_pagina[p]:
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
