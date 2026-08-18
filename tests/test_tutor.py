import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import SimpleChatModel

from local_teacher.query.pipeline import PipelineConsulta
from local_teacher.query.critic import DecisionCritico
from local_teacher.query.optimizer import ConsultaEstructurada


class DummyRetriever(BaseRetriever):
    docs: list[Document] = []

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.docs


class FakeChatModel(SimpleChatModel):
    responses: list[str]
    _idx: int = 0

    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp

    @property
    def _llm_type(self) -> str:
        return "fake_chat_model"


def test_tutor_aprobado_primer_intento(monkeypatch):
    monkeypatch.setattr(
        "local_teacher.query.pipeline.reescribir_consulta",
        lambda llm, q, h: ConsultaEstructurada(consulta=q, capitulo=None, entidades=[]),
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.obtener_contexto_grafo",
        lambda entities, *args, **kwargs: ("(Grafo)", []),
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.recuperar_y_filtrar",
        lambda ret, q, c, e: [
            Document(page_content="Un actuador convierte energía en movimiento.")
        ],
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.evaluar_borrador",
        lambda llm, ctx, draft: DecisionCritico.APROBADO.value,
    )

    fake_llm = FakeChatModel(responses=["Un actuador genera movimiento mecánico [1]."])
    dummy_retriever = DummyRetriever()

    pipeline = PipelineConsulta(
        retriever=dummy_retriever,
        llm=fake_llm,
        busqueda_web_alternativa=False,
        usar_critico=True,
    )

    resultados = list(pipeline.ejecutar("¿Qué es un actuador?"))
    respuestas = [r["answer"] for r in resultados if "answer" in r]
    docs = [r["context_docs"] for r in resultados if "context_docs" in r]

    assert len(respuestas) > 0
    assert "actuador" in respuestas[0]
    assert len(docs) > 0
    assert len(docs[0]) == 1


def test_tutor_reintento_por_rechazo_del_critico(monkeypatch):
    monkeypatch.setattr(
        "local_teacher.query.pipeline.reescribir_consulta",
        lambda llm, q, h: ConsultaEstructurada(consulta=q, capitulo=None, entidades=[]),
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.obtener_contexto_grafo",
        lambda entities, *args, **kwargs: ("(Grafo)", []),
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.recuperar_y_filtrar",
        lambda ret, q, c, e: [Document(page_content="Concepto base.")],
    )

    evaluaciones = [DecisionCritico.RECHAZADO.value, DecisionCritico.APROBADO.value]
    eval_idx = 0

    def mock_evaluar(llm, ctx, draft):
        nonlocal eval_idx
        res = evaluaciones[eval_idx]
        eval_idx += 1
        return res

    monkeypatch.setattr("local_teacher.query.pipeline.evaluar_borrador", mock_evaluar)

    fake_llm = FakeChatModel(responses=[
        "Respuesta alucinada inventada",
        "Respuesta corregida y precisa basada en el texto [1].",
    ])
    dummy_retriever = DummyRetriever()

    pipeline = PipelineConsulta(
        retriever=dummy_retriever,
        llm=fake_llm,
        busqueda_web_alternativa=False,
        usar_critico=True,
    )

    resultados = list(pipeline.ejecutar("Pregunta de prueba"))
    respuestas = [r["answer"] for r in resultados if "answer" in r]

    assert any("[!]" in r for r in respuestas)
    assert any("Respuesta corregida" in r for r in respuestas)


def test_tutor_admite_ignorancia_sin_bucle(monkeypatch):
    monkeypatch.setattr(
        "local_teacher.query.pipeline.reescribir_consulta",
        lambda llm, q, h: ConsultaEstructurada(consulta=q, capitulo=None, entidades=[]),
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.obtener_contexto_grafo",
        lambda entities, *args, **kwargs: ("(Grafo)", []),
    )
    monkeypatch.setattr(
        "local_teacher.query.pipeline.recuperar_y_filtrar",
        lambda ret, q, c, e: [Document(page_content="Texto de prueba")],
    )

    fake_llm = FakeChatModel(responses=["Este tema no está cubierto en el material cargado."])
    dummy_retriever = DummyRetriever()

    pipeline = PipelineConsulta(
        retriever=dummy_retriever,
        llm=fake_llm,
        busqueda_web_alternativa=False,
        usar_critico=True,
    )

    resultados = list(pipeline.ejecutar("Pregunta fuera de contexto"))
    respuestas = [r["answer"] for r in resultados if "answer" in r]

    assert len(respuestas) == 1
    assert "no está cubierto" in respuestas[0].lower()


def test_tutor_cache_semantico_acierto():
    fake_cache = MagicMock()
    doc_cache = Document(page_content="Pregunta en cache", metadata={"respuesta": "Respuesta en cache."})
    fake_cache.similarity_search_with_score.return_value = [(doc_cache, 0.99)]

    fake_llm = FakeChatModel(responses=["No debería llamarse al LLM"])
    dummy_retriever = DummyRetriever()

    pipeline = PipelineConsulta(
        retriever=dummy_retriever,
        llm=fake_llm,
        cache_store=fake_cache,
        usar_critico=True,
    )

    resultados = list(pipeline.ejecutar("Pregunta exacta"))
    respuestas = [r["answer"] for r in resultados if "answer" in r]

    assert len(respuestas) == 1
    assert respuestas[0] == "Respuesta en cache."
