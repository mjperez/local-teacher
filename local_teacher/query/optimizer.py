import json
import logging
import os
import re
import time
import concurrent.futures
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

_log = logging.getLogger(__name__)

_optimizer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class ConsultaReescrita(BaseModel):
    """Modelo Pydantic para la salida estructurada del agente de reformulación."""
    consulta: str = Field(description="Consulta optimizada y enriquecida para búsqueda vectorial")
    capitulo: Optional[str] = Field(default=None, description="Número o nombre del capítulo si aplica")
    entidades: List[str] = Field(default_factory=list, description="Lista de conceptos y entidades extraídas")
    intencion: str = Field(default="conceptual", description="Intención pedagógica: conceptual, ejercicio o aclaracion")


class ConsultaEstructurada:
    """Clase interna para transportar la consulta procesada con sus filtros."""
    def __init__(
        self,
        consulta: str,
        capitulo: str | None = None,
        entidades: List[str] | None = None,
        intencion: str = "conceptual",
    ):
        self.consulta = consulta
        self.capitulo = capitulo
        self.entidades = entidades or []
        self.intencion = intencion


def _limpiar_bloques_razonamiento(texto: str) -> str:
    """Elimina etiquetas de razonamiento de modelos como DeepSeek R1."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()


def _normalizar_entidades(raw_entities) -> list[str]:
    resultado = []
    if isinstance(raw_entities, list):
        for e in raw_entities:
            if isinstance(e, str):
                resultado.append(e.strip())
            elif isinstance(e, dict):
                for k in ["nombre", "name", "entidad", "entity", "valor", "concepto"]:
                    if k in e and isinstance(e[k], str):
                        resultado.append(e[k].strip())
                        break
                else:
                    for v in e.values():
                        if isinstance(v, str):
                            resultado.append(v.strip())
                            break
            elif e is not None:
                resultado.append(str(e).strip())
    elif isinstance(raw_entities, str):
        resultado = [item.strip() for item in raw_entities.split(",") if item.strip()]
    return [r for r in resultado if r]


def _extraer_json_o_campos(texto: str, consulta_original: str) -> ConsultaEstructurada:
    """Parsea el resultado del LLM intentando JSON primero y formato clave-valor como respaldo."""
    limpio = _limpiar_bloques_razonamiento(texto)

    # 1. Intentar extraer bloque JSON con regex o parseo directo
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", limpio, re.DOTALL)
    contenido_json = json_match.group(1) if json_match else limpio

    try:
        data = json.loads(contenido_json)
        if isinstance(data, dict) and "consulta" in data:
            cap = data.get("capitulo")
            if cap and str(cap).upper() in ("N/A", "NONE", "NULL", ""):
                cap = None
            ents = _normalizar_entidades(data.get("entidades", []))
            return ConsultaEstructurada(
                consulta=str(data.get("consulta", consulta_original)).strip(),
                capitulo=str(cap).strip() if cap else None,
                entidades=ents,
                intencion=str(data.get("intencion", "conceptual")).strip(),
            )
    except Exception:
        pass

    # 2. Respaldo por formato de líneas de texto (CONSULTA / CAPITULO / ENTIDADES / INTENCION)
    consulta_opt = consulta_original
    capitulo_opt = None
    entidades_opt = []
    intencion_opt = "conceptual"

    match_c = re.search(r"CONSULTA:\s*(.*)", limpio, re.IGNORECASE)
    if match_c:
        val = match_c.group(1).strip()
        if val:
            consulta_opt = val

    match_cap = re.search(r"CAPITULO:\s*(.*)", limpio, re.IGNORECASE)
    if match_cap:
        val = match_cap.group(1).strip()
        if val and val.upper() not in ("N/A", "NONE", "NULL", ""):
            capitulo_opt = val

    match_ent = re.search(r"ENTIDADES:\s*(.*)", limpio, re.IGNORECASE)
    if match_ent:
        ents_str = match_ent.group(1).strip()
        if ents_str and ents_str.upper() not in ("N/A", "NONE", "NULL", ""):
            entidades_opt = [e.strip() for e in ents_str.split(",") if e.strip()]

    match_int = re.search(r"INTENCION:\s*(.*)", limpio, re.IGNORECASE)
    if match_int:
        val_int = match_int.group(1).strip().lower()
        if val_int in ("conceptual", "ejercicio", "aclaracion"):
            intencion_opt = val_int

    return ConsultaEstructurada(
        consulta=consulta_opt,
        capitulo=capitulo_opt,
        entidades=entidades_opt,
        intencion=intencion_opt,
    )


def reescribir_consulta(
    llm: BaseChatModel, consulta: str, chat_history: list = None
) -> ConsultaEstructurada:
    """Optimiza la consulta del estudiante para la búsqueda híbrida y recupera entidades clave."""
    prompt_reescritura = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un bibliotecario y tutor experto. Tu tarea es optimizar la pregunta del estudiante para buscar en una base de datos vectorial y de grafos.\n"
                "Instrucciones:\n"
                "1. Escribe una oración clara y descriptiva en lenguaje natural para la búsqueda. NUNCA generes código SQL, Cypher ni consultas técnicas de bases de datos.\n"
                "2. Expande términos coloquiales, acrónimos y modismos a conceptos formales (ej. 'bd' -> 'base de datos').\n"
                "3. Extrae un capítulo solo si se solicita de manera explícita.\n"
                "4. Extrae los nombres de entidades y conceptos técnicos más relevantes.\n"
                "5. Identifica la intención: 'conceptual' (teoría), 'ejercicio' (problema práctico) o 'aclaracion' (duda rápida).\n"
                "6. Mantén el idioma de la consulta.\n"
                "7. IMPORTANTE: NO escribas código SQL, no escribas SELECT, y no uses sintaxis de base de datos relacional. La salida debe ser JSON puro con texto en lenguaje natural.\n"
                "Responde en formato JSON estricto con las siguientes claves: 'consulta', 'capitulo', 'entidades', 'intencion'.",
            ),
            (
                "human",
                "Historial de conversación:\n{historial}\n\nPregunta actual del usuario: {input}",
            ),
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

        def _invocar() -> ConsultaEstructurada:
            res = cadena_reescritura.invoke(
                {"input": consulta, "historial": historial_str}
            )
            contenido = res.content if hasattr(res, "content") else str(res)
            return _extraer_json_o_campos(contenido, consulta)
        t0 = time.time()
        future = _optimizer_executor.submit(_invocar)
        try:
            try:
                _timeout = max(5, min(60, int(os.getenv("OPTIMIZER_TIMEOUT_SECS", "45"))))
            except (ValueError, TypeError):
                _log.warning("OPTIMIZER_TIMEOUT_SECS tiene un valor inválido; usando 45s por defecto.")
                _timeout = 45
            res_val = future.result(timeout=_timeout)
            _log.info(f"Optimización completada en {time.time() - t0:.2f}s.")
            return res_val
        except concurrent.futures.TimeoutError:
            t_elapsed = time.time() - t0
            model_name = getattr(llm, "model", getattr(llm, "model_name", "Desconocido"))
            _log.warning(f"Tiempo de espera agotado ({t_elapsed:.2f}s) al optimizar consulta con el modelo {model_name}. Usando consulta original.")
            return ConsultaEstructurada(consulta=consulta, capitulo=None, entidades=[])
    except Exception as e:
        _log.error("Fallo al reescribir la consulta: %s. Usando consulta original.", e)
        return ConsultaEstructurada(consulta=consulta, capitulo=None, entidades=[])
