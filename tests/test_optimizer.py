import pytest
from langchain_core.language_models.chat_models import SimpleChatModel
from local_teacher.query.optimizer import (
    _extraer_json_o_campos,
    _limpiar_bloques_razonamiento,
    reescribir_consulta,
)


class MockLLM(SimpleChatModel):
    response_text: str

    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        return self.response_text

    @property
    def _llm_type(self) -> str:
        return "mock_llm"


def test_limpiar_bloques_razonamiento():
    texto = "<think>Analizando la pregunta del alumno...</think>¿Qué es un vector?"
    assert _limpiar_bloques_razonamiento(texto) == "¿Qué es un vector?"


def test_extraer_json_valido():
    json_str = '{"consulta": "definición de base de datos relacional", "capitulo": "3", "entidades": ["Base de datos", "SQL"], "intencion": "conceptual"}'
    res = _extraer_json_o_campos(json_str, "consulta original")

    assert res.consulta == "definición de base de datos relacional"
    assert res.capitulo == "3"
    assert "Base de datos" in res.entidades
    assert res.intencion == "conceptual"


def test_extraer_json_bloque_markdown():
    json_markdown = """```json
{
    "consulta": "arquitectura de microservicios",
    "capitulo": null,
    "entidades": ["Microservicios", "Docker"],
    "intencion": "conceptual"
}
```"""
    res = _extraer_json_o_campos(json_markdown, "consulta original")
    assert res.consulta == "arquitectura de microservicios"
    assert res.capitulo is None
    assert "Docker" in res.entidades


def test_extraer_formato_lineas_respaldo():
    lineas = """CONSULTA: algoritmos de ordenamiento rápido
CAPITULO: 5
ENTIDADES: QuickSort, MergeSort
INTENCION: ejercicio"""
    res = _extraer_json_o_campos(lineas, "consulta original")
    assert res.consulta == "algoritmos de ordenamiento rápido"
    assert res.capitulo == "5"
    assert "QuickSort" in res.entidades
    assert res.intencion == "ejercicio"


def test_extraer_json_entidades_diccionario():
    json_str = '{"consulta": "normas ISO en software", "capitulo": null, "entidades": [{"nombre": "ISO 12207", "desc": "estándar"}, {"name": "Ciclo de vida"}], "intencion": "conceptual"}'
    res = _extraer_json_o_campos(json_str, "consulta original")
    assert res.consulta == "normas ISO en software"
    assert "ISO 12207" in res.entidades
    assert "Ciclo de vida" in res.entidades


def test_reescribir_consulta_con_mock():
    mock_llm = MockLLM(
        response_text='{"consulta": "definición formal de autómatas finitos", "capitulo": null, "entidades": ["Autómata"], "intencion": "conceptual"}'
    )
    res = reescribir_consulta(mock_llm, "¿qué es un automata?")
    assert res.consulta == "definición formal de autómatas finitos"
    assert "Autómata" in res.entidades
