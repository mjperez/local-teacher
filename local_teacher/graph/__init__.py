"""Módulo de conexión y operaciones sobre el Grafo de Conocimiento (Memgraph MAGE)."""

from local_teacher.graph.client import MemgraphClient, get_memgraph_client

__all__ = ["MemgraphClient", "get_memgraph_client"]
