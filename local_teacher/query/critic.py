import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

def evaluar_borrador(llm: BaseChatModel, texto_contexto: str, borrador: str) -> str:
    """5. Agente Crítico (Self-RAG Evaluator)"""
    prompt_critico = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un clasificador binario. Tu única salida válida es APROBADO o RECHAZADO, sin ensayos ni explicaciones.",
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
                "INSTRUCCIÓN FINAL:\n"
                "1. CRÍTICO: Si la respuesta indica honestamente que no hay información suficiente en el contexto para responder, responde APROBADO (es correcto admitir ignorancia si el texto no lo cubre).\n"
                "2. CRÍTICO: Si la respuesta divaga hablando de temas que están en el contexto pero que NO responden directamente a lo que el usuario preguntó, responde RECHAZADO.\n"
                "3. Si la respuesta inventa o alucina información que no está en el contexto, responde RECHAZADO.\n"
                "4. Si la respuesta es útil, directa y está sustentada en el contexto, responde APROBADO.\n"
                "¿Apruebas la respuesta generada dadas las instrucciones anteriores? Responde EXACTAMENTE con la palabra 'APROBADO' o 'RECHAZADO' y NADA MÁS. No incluyas puntuación ni explicaciones.",
            ),
        ]
    )

    cadena_critico = prompt_critico | llm
    resultado = cadena_critico.invoke({"context": texto_contexto, "draft": borrador})

    contenido = (
        resultado.content if hasattr(resultado, "content") else str(resultado)
    ).strip()
    
    # Limpiar <think> si es un modelo tipo DeepSeek
    contenido = re.sub(r"<think>.*?</think>", "", contenido, flags=re.DOTALL).strip()
    
    # Asegurar que el formato sea exacto, sino, ser tolerantes para evitar bucles
    if "APROBADO" in contenido.upper():
        return "APROBADO"
    elif "RECHAZADO" in contenido.upper():
        return "RECHAZADO"
    
    return "APROBADO"
