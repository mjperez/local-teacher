import logging
import concurrent.futures
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Optional
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

# Pool global para evitar overhead de instanciación
_optimizer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

import re

class ConsultaReescrita(BaseModel):
    consulta: str = Field(...)
    capitulo: Optional[str] = Field(default=None)
    entidades: List[str] = Field(default_factory=list)
    intencion: str = Field(default="conceptual")

class ConsultaEstructurada:
    def __init__(self, consulta: str, capitulo: str | None, entidades: List[str], intencion: str = "conceptual"):
        self.consulta = consulta
        self.capitulo = capitulo
        self.entidades = entidades
        self.intencion = intencion

def reescribir_consulta(llm: BaseChatModel, consulta: str, chat_history: list = None) -> ConsultaEstructurada:
    """1. Agente de Reformulación (Rewrite)"""
    prompt_reescritura = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un bibliotecario experto con mucha 'cachativa' (sentido común e intuición deductiva). Tu tarea es optimizar la pregunta del estudiante para buscarla en una base de datos vectorial e híbrida.\n"
                "Reglas:\n"
                "1. Usa el historial para deducir el contexto y transformar referencias vagas ('eso', 'esa clase', 'la wea') en una consulta técnica, precisa y autocontenida.\n"
                "2. Expande términos coloquiales, argot estudiantil, acrónimos o sinónimos a sus conceptos técnicos formales (ej. 'bd' -> 'base de datos', 'backend' -> 'programación backend').\n"
                "3. Si la pregunta menciona una FECHA coloquial o aproximada (ej. '17 de agosto', 'la última clase'), tradúcela también a posibles formatos numéricos de archivo (ej. '17-08' o '17-08-2026') si tienes esa información.\n"
                "4. Extrae un CAPÍTULO sólo si la pregunta se refiere explícitamente a un número de capítulo.\n"
                "5. MANTÉN ESTRICTAMENTE el idioma original de la consulta.\n"
                "RESPONDE ESTRICTAMENTE EN ESTE FORMATO TEXTUAL, SIN JSON NI EXPLICACIONES:\n"
                "CONSULTA: [consulta optimizada y expandida con sinónimos clave]\n"
                "CAPITULO: [capitulo o N/A]\n"
                "ENTIDADES: [entidad1, entidad2, etc]"
            ),
            ("human", "Historial de conversación:\n{historial}\n\nPregunta actual del usuario (traduciendo su cachativa a conceptos de búsqueda): {input}"),
        ]
    )

    historial_str = ""
    if chat_history:
        for msg in chat_history[-4:]:
            rol = "Usuario" if msg.type == "human" else "Asistente"
            historial_str += f"{rol}: {msg.content}\n"
    if not historial_str:
        historial_str = "No hay historial previo."

    try:
        cadena_reescritura = prompt_reescritura | llm
        
        def _invocar():
            res = cadena_reescritura.invoke({"input": consulta, "historial": historial_str})
            content = res.content if hasattr(res, "content") else str(res)
            # Quitar bloques <think> si es DeepSeek
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            # Parsear el formato textual
            consulta_opt = consulta
            capitulo_opt = None
            entidades_opt = []
            
            match_c = re.search(r"CONSULTA:\s*(.*)", content)
            if match_c: consulta_opt = match_c.group(1).strip()
                
            match_cap = re.search(r"CAPITULO:\s*(.*)", content)
            if match_cap and "N/A" not in match_cap.group(1).upper():
                capitulo_opt = match_cap.group(1).strip()
                
            match_ent = re.search(r"ENTIDADES:\s*(.*)", content)
            if match_ent:
                ents = match_ent.group(1).strip()
                if ents and "N/A" not in ents.upper():
                    entidades_opt = [e.strip() for e in ents.split(",")]
                    
            return ConsultaEstructurada(consulta=consulta_opt, capitulo=capitulo_opt, entidades=entidades_opt)
            
        future = _optimizer_executor.submit(_invocar)
        try:
            return future.result(timeout=45)
        except concurrent.futures.TimeoutError:
            _log.warning("[!] Timeout de 45s excedido al optimizar. Usando fallback.")
            return ConsultaEstructurada(consulta=consulta, capitulo=None, entidades=[])
    except Exception as e:
        _log.error(f"[!] Falló la extracción estructurada: {e}. Usando fallback.")
        return ConsultaEstructurada(consulta=consulta, capitulo=None, entidades=[])
