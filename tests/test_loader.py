from pathlib import Path

from local_teacher.loader import cargar_archivos


def test_cargar_archivos_carga_txt(tmp_path: Path) -> None:
    (tmp_path / "notas.txt").write_text("Hola mundo", encoding="utf-8")

    documentos = cargar_archivos(tmp_path)

    assert len(documentos) == 1
    assert "Hola mundo" in documentos[0].page_content
    assert documentos[0].metadata["source"].endswith("notas.txt")


def test_cargar_archivos_omite_pdf_ilegible(tmp_path: Path) -> None:
    (tmp_path / "roto.pdf").write_bytes(b"%PDF-1.4\n%fake pdf")

    documentos = cargar_archivos(tmp_path)

    assert documentos == []
