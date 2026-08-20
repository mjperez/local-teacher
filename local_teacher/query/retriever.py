import logging
from typing import Iterator, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from local_teacher.query.types import SemanticCacheProtocol

from local_teacher.query.pipeline import PipelineConsulta
from local_teacher.query.graph_search import obtener_contexto_grafo
from local_teacher.query.reranker import recuperar_y_filtrar, get_ranker
from local_teacher.query.prompts import formatear_documentos, crear_cadena_tutor
from local_teacher.query.critic import evaluar_borrador

_log = logging.getLogger(__name__)

# Re-exportaciones compatibles con el código existente
_obtener_contexto_grafo = obtener_contexto_grafo
_recuperar_y_filtrar = recuperar_y_filtrar
_formatear_documentos = formatear_documentos
_crear_cadena_tutor = crear_cadena_tutor
_evaluar_borrador = evaluar_borrador


def stream_consulta(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    consulta: str,
    chat_history: list = None,
    busqueda_web_alternativa: bool = False,
    web_filter: str = "",
    cache_store: Optional[SemanticCacheProtocol] = None,
    usar_critico: bool = True,
    llm_critic: Optional[BaseChatModel] = None,
    llm_fast: Optional[BaseChatModel] = None,
) -> Iterator[dict]:
    """Transmite en streaming la respuesta del tutor local con barra de progreso en consola."""
    pipeline = PipelineConsulta(
        retriever=retriever,
        llm=llm,
        busqueda_web_alternativa=busqueda_web_alternativa,
        web_filter=web_filter,
        cache_store=cache_store,
        usar_critico=usar_critico,
        llm_critic=llm_critic,
        llm_fast=llm_fast,
    )


    def _progreso_print(
        paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False
    ) -> None:

        import shutil
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        porcentaje = int((paso / total) * 100)
        barra = "█" * (porcentaje // 10) + "░" * (10 - (porcentaje // 10))
        texto = f"\r[{barra}] {porcentaje:3}% | {mensaje}"
        texto = texto + " " * max(0, terminal_width - len(texto) - 1)

        if saltar_linea:
            try:
                print(texto, flush=True)
                print() # Extra blank line
            except UnicodeEncodeError:
                print(texto.replace("█", "=").replace("░", "-"), flush=True)
                print()
        else:
            try:
                print(texto, end="", flush=True)
            except UnicodeEncodeError:
                print(texto.replace("█", "=").replace("░", "-"), end="", flush=True)

    yield from pipeline.ejecutar(consulta, chat_history, _progreso_print)


def procesar_consulta(*args, **kwargs) -> dict:
    """Ejecuta una consulta y retorna el diccionario consolidado con respuesta y documentos."""
    ans = ""
    docs = []
    for chunk in stream_consulta(*args, **kwargs):
        if "answer" in chunk:
            ans += chunk["answer"]
        if "context_docs" in chunk:
            docs = chunk["context_docs"]
    return {"answer": ans, "context_docs": docs}


def ejecutar_consulta(*args, **kwargs) -> dict:
    """Alias de procesar_consulta para compatibilidad de interfaz."""
    return procesar_consulta(*args, **kwargs)
