import os
import json
import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

# Logger dedicado a métricas de latencia estructuradas
metrics_logger = logging.getLogger("latency_metrics")
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(LOGS_DIR, "latency_metrics.jsonl")

if not metrics_logger.handlers:
    fh = logging.FileHandler(METRICS_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(fh)


class QueryMetricsTracker:
    def __init__(
        self,
        consulta: str,
        llm: BaseChatModel,
        llm_critic: BaseChatModel,
        usar_critico: bool,
    ):
        self.t_inicio_total = time.time()
        self.consulta = consulta
        self.llm = llm
        self.llm_critic = llm_critic
        self.usar_critico = usar_critico

        self.meta = {
            "cache_hit": False,
            "final_status": "UNKNOWN",
            "optimized_query": "",
            "attempts": 0,
            "num_docs": 0,
            "context_size_chars": 0,
            "web_search_used": False,
        }

        self.latencias = {
            "Optimizacion_Reescritura": 0.0,
            "Recuperacion_Grafo": 0.0,
            "Recuperacion_Vectorial": 0.0,
            "Total": 0.0,
        }

    def set_meta(self, key: str, value: Any):
        self.meta[key] = value

    def add_latency(self, key: str, start_time: float):
        self.latencias[key] = time.time() - start_time

    def init_latency(self, key: str):
        self.latencias[key] = 0.0

    def finish_and_log(self, estado_final: str = "UNKNOWN"):
        self.meta["final_status"] = estado_final
        t_total = time.time() - self.t_inicio_total
        self.latencias["Total"] = round(t_total, 2)

        for k in self.latencias:
            self.latencias[k] = round(self.latencias[k], 2)

        nombre_llm = getattr(
            self.llm, "model", getattr(self.llm, "model_name", "desconocido")
        )
        nombre_critico = "desactivado"
        if self.usar_critico:
            llm_c = self.llm_critic or self.llm
            nombre_critico = getattr(
                llm_c, "model", getattr(llm_c, "model_name", "desconocido")
            )

        payload = {
            "timestamp": time.time(),
            "query": self.consulta,
            "llm_gen": nombre_llm,
            "llm_critic": nombre_critico,
        }
        payload.update(self.meta)
        payload["latencies_sec"] = self.latencias

        metrics_logger.info(json.dumps(payload, ensure_ascii=False))
