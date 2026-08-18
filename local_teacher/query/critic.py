import re
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


class DecisionCritico(str, Enum):
    """Decisión binaria emitida por el evaluador de fidelidad."""
    APROBADO = "APROBADO"
    RECHAZADO = "RECHAZADO"


def evaluar_borrador(llm: BaseChatModel, texto_contexto: str, borrador: str) -> str:
    """Evalúa si una respuesta generada está respaldada fielmente por el contexto provisto."""
    prompt_critico = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un evaluador de fidelidad y precisión. Tu tarea es clasificar de forma binaria si la respuesta está sustentada en el contexto recuperado.\n"
                "Salida permitida: únicamente la palabra APROBADO o RECHAZADO.",
            ),
            (
                "human",
                "Contexto Original: Guido van Rossum.\nRespuesta Generada: Python fue inventado en 1991.\n¿Alucinó datos técnicos? Responde APROBADO o RECHAZADO:",
            ),
            ("ai", "RECHAZADO"),
            (
                "human",
                "Contexto Original: SICP enseña Scheme.\nRespuesta Generada: SICP es un libro de Scheme.\n¿Alucinó datos técnicos? Responde APROBADO o RECHAZADO:",
            ),
            ("ai", "APROBADO"),
            (
                "human",
                "Contexto Original:\n{context}\n\nRespuesta Generada:\n{draft}\n\n"
                "Reglas de evaluación:\n"
                "1. Si la respuesta admite honestamente que el material no contiene la información, responde APROBADO.\n"
                "2. Si la respuesta introduce hechos o detalles que no aparecen en el contexto, responde RECHAZADO.\n"
                "3. Si la respuesta responde de forma coherente usando exclusivamente el contexto, responde APROBADO.\n"
                "¿Apruebas la respuesta? Responde exactamente APROBADO o RECHAZADO.",
            ),
        ]
    )

    cadena_critico = prompt_critico | llm
    resultado = cadena_critico.invoke({"context": texto_contexto, "draft": borrador})

    contenido = (
        resultado.content if hasattr(resultado, "content") else str(resultado)
    ).strip()

    # Limpiar posibles bloques <think> de modelos de razonamiento
    contenido = re.sub(r"<think>.*?</think>", "", contenido, flags=re.DOTALL).strip().upper()

    if "RECHAZADO" in contenido or "REJECTED" in contenido:
        return DecisionCritico.RECHAZADO.value
    if "APROBADO" in contenido or "APPROVED" in contenido:
        return DecisionCritico.APROBADO.value

    return DecisionCritico.APROBADO.value
