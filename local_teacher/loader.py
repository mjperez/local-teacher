"""Capa de carga: lee archivos y los convierte en documentos."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# Hacer que Docling muestre progreso en la consola
logging.getLogger("docling").setLevel(logging.INFO)

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

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
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
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
            num_threads=1, 
            device=AcceleratorDevice.AUTO
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
        figuras.append({"indice": i, "pagina": pagina, "caption": caption, "ruta": str(ruta)})

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

        tablas.append({
            "indice": i,
            "ruta_markdown": str(base.with_suffix(".md")),
            "ruta_csv": str(base.with_suffix(".csv")),
            "filas": len(df) if df is not None else 0,
            "columnas": len(df.columns) if df is not None else 0,
        })

    return tablas


# ---------------------------------------------------------------------------
# Conversión a documentos LangChain
# ---------------------------------------------------------------------------

def _docs_figuras(figuras: list[dict], fuente: Path) -> list[Document]:
    """Convierte figuras exportadas en Documents para el RAG."""
    return [
        Document(
            page_content=f"{fig['caption']}\nImagen: {fig['ruta']}",
            metadata={
                "source": str(fuente),
                "file_type": "figure",
                "pagina": fig["pagina"],
                "ruta": fig["ruta"],
                "caption": fig["caption"],
            },
        )
        for fig in figuras
    ]


def _docs_tablas(tablas: list[dict], fuente: Path) -> list[Document]:
    """Convierte tablas exportadas en Documents para el RAG."""
    return [
        Document(
            page_content=f"Tabla:\n{Path(t['ruta_markdown']).read_text(encoding='utf-8')}",
            metadata={
                "source": str(fuente),
                "file_type": "table",
                "ruta_markdown": t["ruta_markdown"],
                "ruta_csv": t["ruta_csv"],
            },
        )
        for t in tablas
    ]


# ---------------------------------------------------------------------------
# Cargadores por formato
# ---------------------------------------------------------------------------

def _cargar_pdf(
    ruta: Path,
    extraer_figuras: bool,
    extraer_tablas: bool,
    carpeta_figuras,
    enriquecer_formulas: bool = True,
) -> list[Document]:
    """Carga un PDF con Docling y devuelve texto, figuras y tablas usando 2 pasadas si hay fórmulas."""
    try:
        from docling.chunking import HierarchicalChunker
        from docling.datamodel.document import TextItem

        _log.info("Iniciando pasada rápida de Docling...")
        # 1. Pasada Rápida (sin VLM de fórmulas, tablas solo si extraer_tablas=True)
        doc_fast = _get_docling_converter(False, extraer_tablas).convert(str(ruta)).document

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
            p = c.meta.doc_items[0].prov[0].page_no if c.meta.doc_items and c.meta.doc_items[0].prov else 1
            if p not in textos_por_pagina:
                textos_por_pagina[p] = []
            textos_por_pagina[p].append(c.text)

        # 2. Pasada Lenta (solo si se encontraron fórmulas y está activado)
        if paginas_con_formulas:
            paginas = sorted(paginas_con_formulas)
            _log.info(f"Fórmulas detectadas en {len(paginas)} páginas: {paginas}. Ejecutando VLM (esto tomará un tiempo)...")
            
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
                doc_lento = converter_vlm.convert(str(ruta), page_range=(inicio, fin)).document
                
                reemplazados = set()
                for c in chunker.chunk(doc_lento):
                    p = c.meta.doc_items[0].prov[0].page_no if c.meta.doc_items and c.meta.doc_items[0].prov else inicio
                    if p not in reemplazados:
                        textos_por_pagina[p] = []
                        reemplazados.add(p)
                    textos_por_pagina[p].append(c.text)

        # Armar el markdown final
        texto = ""
        for p in sorted(textos_por_pagina.keys()):
            texto += "\n\n".join(textos_por_pagina[p]) + "\n\n"
        
        texto = _limpiar_formulas(texto)
        
        meta = {"source": str(ruta), "file_type": "pdf", "backend": "docling"}
        docs: list[Document] = []

        destino = _carpeta_destino(ruta, carpeta_figuras)

        if extraer_figuras:
            figs = _extraer_figuras(ruta, doc_fast, destino)
            meta["figuras"] = figs
            docs.extend(_docs_figuras(figs, ruta))

        if extraer_tablas:
            tabs = _extraer_tablas(doc_fast, destino / "tablas")
            meta["tablas"] = tabs
            docs.extend(_docs_tablas(tabs, ruta))

        if texto:
            docs.append(Document(page_content=texto, metadata=meta))
        return docs

    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        _log.error("Error procesando %s con Docling: %s", ruta.name, exc)
        return []


def _cargar_jsonl(ruta: Path) -> list[Document]:
    """Carga un .jsonl con formato Natural Questions."""
    docs = []
    with open(ruta, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "question" in data and "answer" in data:
                resp = ", ".join(data["answer"]) if isinstance(data["answer"], list) else str(data["answer"])
                text = f"Q: {data['question']}\nA: {resp}"
            else:
                text = json.dumps(data, ensure_ascii=False)
            docs.append(Document(page_content=text, metadata={"source": str(ruta), "line": idx}))
    return docs


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def cargar_archivos(
    ruta_carpeta: Path | str,
    extraer_figuras: bool = False,
    extraer_tablas: bool = False,
    carpeta_figuras: Path | str | None = None,
    enriquecer_formulas: bool = True,
) -> list[Document]:
    """Carga archivos de una ruta como documentos.

    Formatos soportados: .txt, .md, .pdf, .jsonl
    """
    ruta_carpeta = Path(ruta_carpeta)
    if not ruta_carpeta.exists():
        raise FileNotFoundError(f"No existe la ruta de ingesta: {ruta_carpeta}")

    docs: list[Document] = []
    items = [ruta_carpeta] if ruta_carpeta.is_file() else ruta_carpeta.rglob("*")

    for item in items:
        if not item.is_file():
            continue

        ext = item.suffix.lower()
        if ext in (".txt", ".md"):
            docs.extend(TextLoader(str(item), encoding="utf-8").load())
        elif ext == ".pdf":
            docs.extend(_cargar_pdf(item, extraer_figuras, extraer_tablas, carpeta_figuras, enriquecer_formulas))
        elif ext == ".jsonl":
            docs.extend(_cargar_jsonl(item))

    return docs
