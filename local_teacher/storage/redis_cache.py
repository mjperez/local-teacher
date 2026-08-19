import os
import logging
from typing import List, Tuple, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_log = logging.getLogger(__name__)

class RedisCacheStore:
    """
    Caché Semántico usando Redis Stack (RediSearch) y langchain_community.vectorstores.Redis.
    """
    def __init__(self, embeddings: Embeddings, host: str = "localhost", port: int = 6379):
        self.redis_url = f"redis://{host}:{port}"
        self.embeddings = embeddings
        self.index_name = "local_teacher_cache"
        try:
            import redis
            r = redis.Redis.from_url(self.redis_url)
            r.ping()
            from langchain_community.vectorstores import Redis
            self.vectorstore = Redis(
                redis_url=self.redis_url,
                index_name=self.index_name,
                embedding=self.embeddings,
            )
            self.connected = True
            _log.info("[+] Conectado exitosamente a Redis Stack para Semantic Cache.")
        except Exception as e:
            _log.error(f"[-] No se pudo conectar a Redis Stack (¿Tienes Redis Stack en ejecución?): {e}")
            self.connected = False

    def similarity_search_with_score(self, query: str, k: int = 1) -> List[Tuple[Document, float]]:
        if not self.connected:
            return []
        try:
            # En Langchain Redis, un score más bajo significa más cercanía (distancia Euclideana/Coseno)
            # Adaptaremos el score de similitud restándolo de 1 para ser congruente si es distancia
            # Depende de la métrica por defecto, usualmente es L2 o Cosine distance (0 es idéntico).
            # Para simular "Acierto" que espera el retriever (score > 0.95), invertimos.
            resultados = self.vectorstore.similarity_search_with_score(query, k=k)
            procesados = []
            for doc, distance in resultados:
                # distance de 0 = 1.0 (exact match)
                score_similitud = max(0.0, 1.0 - distance)
                procesados.append((doc, score_similitud))
            return procesados
        except Exception as e:
            _log.warning(f"Error consultando la caché semántica: {e}")
            return []

    def add_texts(self, texts: List[str], metadatas: List[dict]) -> None:
        if not self.connected:
            return
        try:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        except Exception as e:
            _log.warning(f"Error guardando en caché semántica: {e}")

    def clear(self) -> None:
        if not self.connected:
            return
        
        # Guardrail de seguridad: requerir confirmación explícita para evitar pérdida
        # accidental de datos en instancias de Redis compartidas.
        if os.getenv("REDIS_ALLOW_DROP", "false").lower() != "true":
            _log.error(
                "[!] Operación de borrado de caché semántica denegada.\n"
                "    La eliminación de índices en Redis está bloqueada por seguridad.\n"
                "    Configura 'REDIS_ALLOW_DROP=true' en tu entorno si estás seguro de que\n"
                "    el servidor Redis es de uso exclusivo para esta aplicación."
            )
            return

        # Guardrail adicional: solo borrar índices que pertenezcan a esta app.
        if not self.index_name.startswith("local_teacher_"):
            _log.error(
                "[!] Abortando clear(): el index_name '%s' no tiene el prefijo 'local_teacher_'. "
                "Verifica la configuración antes de continuar.",
                self.index_name,
            )
            return
        try:
            from langchain_community.vectorstores import Redis
            _log.warning(
                "[!] Eliminando índice Redis '%s' en %s. Esta operación es irreversible.",
                self.index_name,
                self.redis_url,
            )
            Redis.drop_index(index_name=self.index_name, delete_documents=True, redis_url=self.redis_url)
            _log.info("[*] Índice de Caché Semántico (Redis) eliminado correctamente.")
        except Exception as e:
            _log.warning(f"Error al limpiar caché semántica: {e}")

def get_semantic_cache_store(embeddings: Embeddings) -> RedisCacheStore:
    """Inicializa la conexión a Redis Cache."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return RedisCacheStore(embeddings=embeddings, host=host, port=port)
