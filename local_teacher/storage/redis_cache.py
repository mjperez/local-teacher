import os
import hashlib
import logging
from typing import List, Tuple, Any

import redis

_log = logging.getLogger(__name__)

class DummyDoc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

class RedisCacheStore:
    """
    Implementación simple de Caché L1 (Exact Match) usando Redis estándar.
    """
    def __init__(self, host: str = "localhost", port: int = 6379):
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            # Ping para verificar conexión rápida
            self.client.ping()
            _log.info("[+] Conectado exitosamente a Redis Cache.")
        except Exception as e:
            _log.error(f"[-] No se pudo conectar a Redis: {e}")
            self.client = None

    def _hash_query(self, query: str) -> str:
        # Normalizamos un poco para aumentar aciertos (sin tildes, minúsculas, sin espacios extra)
        # Nota: Idealmente para Caché Semántico real en Redis se requiere el módulo RediSearch y redis/redis-stack
        import unicodedata
        q_norm = unicodedata.normalize('NFKD', query).encode('ASCII', 'ignore').decode('utf-8')
        q_norm = q_norm.lower().strip()
        return f"cache:{hashlib.md5(q_norm.encode('utf-8')).hexdigest()}"

    def similarity_search_with_score(self, query: str, k: int = 1) -> List[Tuple[Any, float]]:
        if not self.client:
            return []
            
        key = self._hash_query(query)
        respuesta = self.client.get(key)
        
        if respuesta:
            # Simulamos el formato de Qdrant (Documento, Score)
            # Retornamos un score de 1.0 (Acierto perfecto)
            doc = DummyDoc(page_content=respuesta, metadata={"respuesta": respuesta})
            return [(doc, 1.0)]
            
        return []

    def add_texts(self, texts: List[str], metadatas: List[dict]) -> None:
        if not self.client:
            return
            
        for i, text in enumerate(texts):
            key = self._hash_query(text)
            respuesta = metadatas[i].get("respuesta", "")
            # Guardamos la respuesta con un TTL de 7 días (604800 segundos)
            if respuesta:
                self.client.setex(key, 604800, respuesta)

def get_semantic_cache_store() -> RedisCacheStore:
    """Inicializa la conexión a Redis Cache."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return RedisCacheStore(host=host, port=port)
