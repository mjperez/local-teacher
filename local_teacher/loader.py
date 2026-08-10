# Capa de carga: solo sabe leer archivos y convertirlos en documentos.

import json
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Tamaño mínimo (en puntos, sobre la página) para considerar una imagen como
# figura/diagrama relevante. Los fragmentos más pequeños suelen ser tiles
# (recortes de una figura mayor) o decoración, y solo añaden ruido.
UMBRAL_FIGURA_ANCHO = 60
UMBRAL_FIGURA_ALTO = 60

# Marker 2.0 requiere vLLM en Docker y GPU Ampere (RTX 30xx+). En la mayoría
# de equipos no está disponible, y probarlo por cada PDF es lento y ruidoso
# (arranca procesos/vLLM). Por eso es opt-in vía variable de entorno.
#   LOCAL_TEACHER_USE_MARKER=1  ->  intenta Marker antes que PyMuPDF
_USAR_MARKER = os.getenv("LOCAL_TEACHER_USE_MARKER", "").strip().lower() in {"1", "true", "yes", "si"}


def _extraer_contenido_pdf(ruta_pdf: Path) -> str | None:
    """Extrae el contenido de un PDF como texto.

    Usa Marker si está habilitado (variable LOCAL_TEACHER_USE_MARKER) y
    disponible; si no, usa PyMuPDF. Devuelve None si no se pudo extraer
    texto legible.
    """
    if _USAR_MARKER:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            model_dict = create_model_dict()
            converter = PdfConverter(artifact_dict=model_dict)
            rendered = converter(str(ruta_pdf))
            contenido = text_from_rendered(rendered)
            if contenido and contenido.strip():
                return contenido.strip()
        except Exception as exc:
            print(f"Error usando Marker para {ruta_pdf.name}: {exc}")

    try:
        import pymupdf

        doc = pymupdf.open(ruta_pdf)
        paginas = [pag.get_text() for pag in doc]
        doc.close()
        contenido = "\n\n".join(txt for txt in paginas if txt and txt.strip())
        if contenido.strip():
            return contenido.strip()
    except Exception as exc:
        print(f"Error extrayendo texto con PyMuPDF para {ruta_pdf.name}: {exc}")

    return None


def _detectar_regiones_figuras(pag, zoom: float = 1.5) -> list:
    """Detecta las regiones donde hay figuras/diagramas en una página.

    Método basado en píxeles, robusto frente a diagramas vectoriales, figuras
    con etiquetas de texto dentro, imágenes embebidas y figuras al inicio o
    final de página:

    1. Renderiza la página en gris.
    2. Enmascara (descarta) los píxeles que pertenecen a texto.
    3. Los píxeles oscuros restantes son "contenido gráfico".
    4. Agrupa las filas con densidad de gráficos en bandas y recorta cada
       banda a sus columnas → una región por figura.

    Esto elimina encabezados de página, marcos y letras capitales (todo eso es
    texto o trazos aislados de baja densidad). Devuelve ``pymupdf.Rect``.
    """
    import numpy as np
    import pymupdf

    mat = pymupdf.Matrix(zoom, zoom)
    pix = pag.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    # Máscara de texto: tapar la zona de cada span de texto.
    mascara = np.zeros_like(a, dtype=bool)
    for blk in pag.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                r = pymupdf.Rect(sp["bbox"]) * mat
                y0, y1 = max(0, int(r.y0)), min(pix.height, int(r.y1) + 1)
                x0, x1 = max(0, int(r.x0)), min(pix.width, int(r.x1) + 1)
                if y1 > y0 and x1 > x0:
                    mascara[y0:y1, x0:x1] = True

    # Píxel de figura = contenido oscuro que no es texto.
    fig = (a < 200) & (~mascara)

    # Densidad por fila; una fila "activa" tiene suficiente gráfico.
    densidad = fig.sum(axis=1)
    umbral_fila = int(pix.width * 0.04)
    activo = densidad >= umbral_fila

    # Unir bandas separadas por huecos pequeños (líneas sueltas de un dibujo).
    holgura = int(30 * zoom)
    bandas: list = []
    inicio = None
    for y, act in enumerate(activo):
        if act and inicio is None:
            inicio = y
        elif not act and inicio is not None and y - inicio > holgura + 1:
            bandas.append((inicio, y))
            inicio = None
    if inicio is not None:
        bandas.append((inicio, len(activo)))

    regiones: list = []
    for y0, y1 in bandas:
        if (y1 - y0) < 25 * zoom:  # demasiado bajo para ser una figura
            continue
        sub = fig[y0:y1, :]
        cols = np.where(sub.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])
        if (x1 - x0) < 25 * zoom:
            continue
        reg = pymupdf.Rect(x0 / zoom, y0 / zoom, (x1 + 1) / zoom, (y1 + 1) / zoom)
        reg &= pag.rect
        if reg.get_area() < 30 * 30:
            continue
        # Bandas decorativas de ancho completo (banners de capítulo, reglas,
        # fondos): tocan ambos bordes de la página. Las figuras reales quedan
        # dentro de los márgenes del texto.
        if reg.x0 <= pag.rect.x0 + 8 and reg.x1 >= pag.rect.x1 - 8:
            continue
        regiones.append(reg)

    return regiones


def _buscar_caption_pdf(pag, region) -> str:
    """Busca el pie de figura (caption) de una región.

    Entre todos los bloques de texto que quedan cerca (o solapan) la región,
    prioriza el que parece pie de figura (``Figure``, ``Figura``, ``Fig.``) y
    que esté más cerca. Si ninguno es un caption claro, usa el bloque de texto
    más cercano debajo, y si no, el más cercano encima.
    """
    import pymupdf

    def _texto(blk: dict) -> str:
        return " ".join(
            span["text"] for linea in blk["lines"] for span in linea["spans"]
        ).strip()

    def _parece_caption(t: str) -> bool:
        t = t.lower()
        return t.startswith(("figure", "figura", "fig.", "fig "))

    def _dist(blk: dict) -> float:
        b = blk["bbox"]
        # distancia vertical región-bloque (0 si se solapan)
        return max(0.0, b[1] - region.y1, region.y0 - b[3])

    bloques = [
        b
        for b in pag.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        if b.get("type") == 0
    ]
    if not bloques:
        return ""

    # 1) caption tipo figura más cercano (dentro de ~200pt o solapado)
    cercanos = sorted(bloques, key=_dist)
    for blk in cercanos:
        if _parece_caption(_texto(blk)) and _dist(blk) <= 200:
            return _texto(blk)

    # 2) bloque más cercano que quede debajo de la figura
    debajo = [b for b in bloques if b["bbox"][1] >= region.y1 - 5]
    if debajo:
        return _texto(min(debajo, key=_dist))

    # 3) bloque más cercano en general (arriba o debajo)
    return _texto(cercanos[0])


def _es_caption_figura(caption: str | None) -> bool:
    """True si el caption parece el de una figura ("Figure 1-2", "Fig.", ...)."""
    c = (caption or "").strip().lower()
    return c.startswith(("figure", "fig.", "fig ", "figura"))


def _exportar_figuras_pdf(ruta_pdf: Path, carpeta_destino: Path) -> list[dict]:
    """Exporta las figuras/diagramas de un PDF como PNG.

    Renderiza cada región detectada (grafismo vectorial + imágenes embebidas)
    tal como se ve en la página, a 2x de resolución. Devuelve una lista de
    diccionarios: ``{"pagina": int, "ruta": str, "caption": str | None}``.
    """
    import pymupdf

    figuras: list[dict] = []
    try:
        doc = pymupdf.open(ruta_pdf)
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        zoom = pymupdf.Matrix(2, 2)

        for num_pagina, pag in enumerate(doc, start=1):
            regiones = _detectar_regiones_figuras(pag)
            if not regiones:
                continue

            # Rangos verticales de los bloques de texto "sustanciales" de la
            # página: multilínea y casi tan anchos como la columna (párrafos o
            # pies de figura). Las anotaciones de los diagramas (e.g. "v → S´")
            # son de una línea y angostas, así que no cuentan.
            ancho_col = pag.rect.width - 2 * 63.0
            texto_y = [
                (b["bbox"][1], b["bbox"][3])
                for b in pag.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
                if b.get("type") == 0
                and (b["bbox"][3] - b["bbox"][1]) >= 20
                and (b["bbox"][2] - b["bbox"][0]) >= 0.6 * ancho_col
            ]

            # Descarta mini-diagramas embebidos dentro de un párrafo de texto
            # (p.ej. dibujitos inline); no son figuras independientes.
            regiones = [
                reg
                for reg in regiones
                if not any(
                    t0 - 2 <= reg.y0 and reg.y1 <= t1 + 2 for t0, t1 in texto_y
                )
            ]
            if not regiones:
                continue

            def _hay_texto_entre(y0: float, y1: float) -> bool:
                """True si algún bloque sustancial ocupa la banda abierta (y0, y1)."""
                margen = 5.0
                return any(t0 < y1 - margen and t1 > y0 + margen for t0, t1 in texto_y)

            # Fusiona en una sola región las partes que comparten caption y
            # que están separadas solo por espacio en blanco (reconstruye
            # figuras compuestas o troceadas, sin tragarse párrafos).
            con_caption = [(reg, _buscar_caption_pdf(pag, reg)) for reg in regiones]
            fusionadas: list = []
            usadas = [False] * len(con_caption)
            for i, (reg, cap) in enumerate(con_caption):
                if usadas[i]:
                    continue
                if not cap:
                    fusionadas.append((reg, cap))
                    usadas[i] = True
                    continue
                grupo = [reg]
                caja = reg
                for j in range(i + 1, len(con_caption)):
                    r2, c2 = con_caption[j]
                    # Une piezas con el mismo caption siempre que entre ellas
                    # no haya texto (solo blanco o grafismos de la propia
                    # figura). Así no se junta una figura con otra distinta
                    # que comparta caption ni se traga párrafos.
                    if (
                        not usadas[j]
                        and c2 == cap
                        and not _hay_texto_entre(caja.y1, r2.y0)
                    ):
                        grupo.append(r2)
                        caja |= r2
                        usadas[j] = True
                fusionadas.append((caja, cap))

            # Descarta ruido: bandas bajas de una sola línea (ecuaciones,
            # símbolos vectoriales) que no pertenezcan a una figura con
            # caption. Las figuras compuestas ya se fusionaron arriba.
            fusionadas = [
                (reg, cap)
                for reg, cap in fusionadas
                if reg.height >= 40 or _es_caption_figura(cap)
            ]

            for idx, (region, caption) in enumerate(fusionadas):
                pix = pag.get_pixmap(clip=region, matrix=zoom)
                nombre = f"fig_p{num_pagina:03d}_n{idx + 1:02d}.png"
                salida = carpeta_destino / nombre
                pix.save(str(salida))
                figuras.append(
                    {
                        "pagina": num_pagina,
                        "ruta": str(salida),
                        "caption": caption or None,
                    }
                )
        doc.close()
    except Exception as exc:
        print(f"Error exportando figuras de {ruta_pdf.name}: {exc}")

    return figuras


def _resolver_carpeta_figuras(ruta_pdf: Path, carpeta: Path | str | None) -> Path:
    """Decide dónde guardar las figuras de un PDF.

    Si se pasa ``carpeta`` se usa esa base; si no, una carpeta ``figuras/``
    junto al PDF. En ambos casos se crea un subdirectorio por cada PDF.
    """
    base = Path(carpeta) if carpeta is not None else ruta_pdf.parent / "figuras"
    return base / ruta_pdf.stem


def _cargar_jsonl(ruta: Path) -> list[Document]:
    """Carga un archivo .jsonl y formatea las preguntas de Natural Questions."""
    documentos: list[Document] = []
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
            documentos.append(
                Document(page_content=text, metadata={"source": str(ruta), "line": idx})
            )
    return documentos


def cargar_archivos(
    ruta_carpeta: Path | str,
    extraer_figuras: bool = False,
    carpeta_figuras: Path | str | None = None,
) -> list[Document]:
    """Carga los archivos de una carpeta como documentos.

    - ``.txt`` / ``.md``: texto plano.
    - ``.pdf``: texto con PyMuPDF (Marker si está disponible). Si
      ``extraer_figuras=True``, además exporta los diagramas/imágenes como PNG
      y los registra en el metadato ``figuras`` del documento.
    - ``.jsonl``: formato Natural Questions.
    """
    documentos: list[Document] = []
    ruta_carpeta = Path(ruta_carpeta)
    try:
        for item in ruta_carpeta.rglob("*"):
            if not item.is_file():
                continue

            extension = item.suffix.lower()
            if extension in [".txt", ".md"]:
                documentos.extend(TextLoader(str(item), encoding="utf-8").load())
            elif extension == ".pdf":
                texto = _extraer_contenido_pdf(item)
                if not texto:
                    continue
                metadata: dict = {"source": str(item), "file_type": "pdf"}
                if extraer_figuras:
                    destino = _resolver_carpeta_figuras(item, carpeta_figuras)
                    figuras = _exportar_figuras_pdf(item, destino)
                    if figuras:
                        metadata["figuras"] = figuras
                documentos.append(Document(page_content=texto, metadata=metadata))
            elif extension == ".jsonl":
                documentos.extend(_cargar_jsonl(item))
    except Exception as exc:
        print(f"Error cargando archivos: {exc}")
    return documentos
