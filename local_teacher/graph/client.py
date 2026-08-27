import os
import logging
from typing import Any, Optional
try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    GraphDatabase = None
    Driver = Any

_log = logging.getLogger(__name__)

_DEFAULT_MEMGRAPH_URI = "bolt://localhost:7687"


class MemgraphClient:
    """Cliente para interactuar con Memgraph MAGE mediante el protocolo Bolt."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or os.getenv("MEMGRAPH_URI", _DEFAULT_MEMGRAPH_URI)
        self.username = username if username is not None else os.getenv("MEMGRAPH_USER", "")
        self.password = password if password is not None else os.getenv("MEMGRAPH_PASSWORD", "")
        
        auth = (self.username, self.password) if (self.username or self.password) else None
        self._driver: Optional[Driver] = None
        self._auth = auth

    def get_driver(self) -> Driver:
        """Obtiene o inicializa el driver de conexión Bolt."""
        if self._driver is None:
            if GraphDatabase is None:
                raise ImportError(
                    "El paquete 'neo4j' no está instalado en el entorno. "
                    "Instálalo con: pip install neo4j"
                )
            _log.info("Conectando a Memgraph en %s...", self.uri)
            self._driver = GraphDatabase.driver(self.uri, auth=self._auth)
        return self._driver

    def ensure_schema(self) -> None:
        """Crea índices y restricciones únicas de forma declarativa si no existen."""
        driver = self.get_driver()
        queries = [
            "CREATE CONSTRAINT ON (e:Entity) ASSERT e.name IS UNIQUE;",
            "CREATE INDEX ON :Entity(name);",
        ]
        with driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    # En Memgraph, si la restricción ya existe puede levantar advertencia
                    _log.debug("Esquema verificado (%s): %s", q, e)

    def execute_query(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Ejecuta una consulta Cypher de lectura y retorna los registros como lista de diccionarios."""
        driver = self.get_driver()
        with driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Optional[dict[str, Any]] = None) -> Any:
        """Ejecuta una consulta Cypher dentro de una transacción de escritura."""
        driver = self.get_driver()
        with driver.session() as session:
            return session.execute_write(lambda tx: tx.run(query, parameters or {}).consume())

    def clear_graph(self) -> None:
        """Elimina todos los nodos y relaciones del grafo."""
        _log.info("Limpiando Grafo de Conocimiento en Memgraph...")
        self.execute_write("MATCH (n) DETACH DELETE n")

    def run_louvain(self) -> None:
        """Ejecuta el algoritmo Louvain de MAGE para asignar IDs de comunidad a los nodos."""
        _log.info("Ejecutando detección de comunidades (Louvain) con MAGE...")
        
        # MAGE < 1.3 usa louvain.get(), MAGE >= 1.3 usa community_detection.get()
        procedures = [
            "CALL community_detection.get() YIELD node, community_id SET node.community = community_id",
            "CALL louvain.get() YIELD node, community_id SET node.community = community_id",
        ]
        
        for query in procedures:
            try:
                self.execute_write(query)
                _log.info("Comunidades Louvain asignadas exitosamente a los nodos del grafo.")
                return
            except Exception:
                continue
        
        _log.warning("No se pudo ejecutar Louvain: ni community_detection.get() ni louvain.get() están disponibles en MAGE.")

    def health_check(self) -> bool:
        """Verifica la conectividad con el servidor Memgraph."""
        try:
            res = self.execute_query("RETURN 1 AS ok")
            return len(res) > 0 and res[0].get("ok") == 1
        except Exception as e:
            _log.error("Error en healthcheck de Memgraph: %s", e)
            return False

    def close(self) -> None:
        """Cierra el driver de conexiones."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None


_global_client: Optional[MemgraphClient] = None


def get_memgraph_client(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> MemgraphClient:
    """Retorna una instancia singleton de MemgraphClient."""
    global _global_client
    if _global_client is None:
        _global_client = MemgraphClient(uri=uri, username=username, password=password)
        import atexit
        atexit.register(_global_client.close)
    return _global_client


def close_memgraph_client() -> None:
    """Cierra la conexión global de Memgraph si está activa."""
    global _global_client
    if _global_client is not None:
        _global_client.close()
        _global_client = None
