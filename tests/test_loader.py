import pytest
from langchain_core.documents import Document
from local_teacher.ingestion.loader import _limpiar_formulas

def test_limpiar_formulas_basic():
    # 1. Bloques vacíos
    assert _limpiar_formulas("$$$$") == "[fórmula]"
    assert _limpiar_formulas("<!-- formula-not-decoded -->") == "[fórmula]"
    assert _limpiar_formulas("$$\n$$") == "[fórmula]"
    
    # 2. Espaciado extraño
    assert _limpiar_formulas(r"\f r a c") == r"\frac"
    assert _limpiar_formulas(r"\a l p h a") == r"\alpha"
    
    # 3. Múltiples corrupciones contiguas se colapsan
    assert _limpiar_formulas("$$$$ $$$$") == "[fórmula]"
