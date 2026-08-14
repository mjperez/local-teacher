import logging
import concurrent.futures
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Optional
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

class ConsultaReescrita(BaseModel):
    consulta: str = Field(
        description="La consulta reescrita y optimizada para vector search. Mantener SIEMPRE el idioma original."
    )
    capitulo: Optional[str] = Field(
        default=None,
        description="Si la consulta es sobre un capítulo o módulo en específico (ej. 'Resumen capitulo 4'), colocar 'capitulo 4'.",
    )
    entidades: List[str] = Field(
        default_factory=list,
        description="Conceptos nucleares, acrónimos o nombres propios mencionados en la pregunta.",
    )
    intencion: str = Field(
        default="conceptual",
        description="Intención de la pregunta: 'conceptual' (explicación teórica), 'ejercicio' (ayuda con un problema práctico), 'aclaracion' (duda rápida).",
    )

class ConsultaEstructurada:
    def __init__(self, consulta: str, capitulo: str | None, entidades: List[str], intencion: str = "conceptual"):
        self.consulta = consulta
        self.capitulo = capitulo
        self.entidades = entidades
        self.intencion = intencion

def reescribir_consulta(llm: BaseChatModel, consulta: str) -> ConsultaEstructurada:
    """1. Agente de Reformulación (Rewrite)"""
    prompt_reescritura = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un bibliotecario experto. Tu tarea es optimizar la pregunta del estudiante para buscarla en una base de datos vectorial e híbrida.\n"
                "Reglas:\n"
                "1. MANTÉN ESTRICTAMENTE el idioma original de la consulta (si está en español, déjala en español).\n"
                "2. Extrae el número de CAPÍTULO SÓLO si la pregunta es exclusivamente sobre ese capítulo (ej. 'Resume el capítulo 4').\n"
                "3. Extrae los CONCEPTOS NÚCLEO o NOMBRES PROPIOS de la pregunta, en una lista de strings.",
            ),
            ("human", "{input}"),
        ]
    )

    try:
        cadena_reescritura = prompt_reescritura | llm.with_structured_output(
            ConsultaReescrita
        )
        
        def _invocar():
            return cadena_reescritura.invoke({"input": consulta})
            
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_invocar)
        try:
            # 20 segundos máximo para optimizar
            respuesta = future.result(timeout=20)
            executor.shutdown(wait=False)
            return ConsultaEstructurada(
                consulta=respuesta.consulta,
                capitulo=respuesta.capitulo,
                entidades=respuesta.entidades,
                intencion=respuesta.intencion
            )
        except concurrent.futures.TimeoutError:
            _log.warning("[!] Timeout de 20s excedido al optimizar. Usando fallback.")
            executor.shutdown(wait=False)
            return ConsultaEstructurada(consulta=consulta, capitulo=None, entidades=[])
    except Exception as e:
        _log.error(f"[!] Falló la extracción estructurada: {e}. Usando fallback.")
        return ConsultaEstructurada(consulta=consulta, capitulo=None, entidades=[])
