import logging
from typing import Optional
from local_teacher.graph import MemgraphClient, get_memgraph_client

_log = logging.getLogger(__name__)


def obtener_contexto_grafo(
    entidades_filtro: list[str],
    client: Optional[MemgraphClient] = None,
    limit: int = 15,
) -> tuple[str, list[str]]:
    """
    Consulta el Grafo de Conocimiento en Memgraph MAGE y expande la búsqueda con entidades relacionadas.
    
    Aprovecha la búsqueda bidireccional ponderada y devuelve el contexto estructurado y las palabras clave.
    """
    contexto_grafo = "(No se detectaron entidades o no hay grafo disponible)"
    palabras_clave_grafo: list[str] = []

    if not entidades_filtro:
        return contexto_grafo, palabras_clave_grafo

    # Limpiar y normalizar entidades
    entidades_limpias = [e.strip() for e in entidades_filtro if e and len(e.strip()) > 1]
    if not entidades_limpias:
        return contexto_grafo, palabras_clave_grafo

    try:
        memgraph = client or get_memgraph_client()

        query = """
        MATCH (a:Entity)-[r:Rel]-(b:Entity)
        WHERE ANY(ent IN $entidades WHERE toLower(a.name) CONTAINS toLower(ent))
        RETURN a.name AS source, 
               COALESCE(r.type, 'CO_OCCURS_WITH') AS rel, 
               b.name AS target, 
               COALESCE(r.weight, 1) AS weight,
               b.community AS community
        ORDER BY weight DESC
        LIMIT $limit
        """
        
        records = memgraph.execute_query(
            query,
            {"entidades": entidades_limpias, "limit": limit},
        )

        if not records:
            return contexto_grafo, palabras_clave_grafo

        conexiones = []
        nodos_vistos = set()

        for rec in records:
            source = rec.get("source")
            rel = rec.get("rel")
            target = rec.get("target")
            weight = rec.get("weight", 1)

            if source:
                nodos_vistos.add(source)
            if target:
                nodos_vistos.add(target)

            if source and target:
                if weight and weight > 1:
                    conexiones.append(f"- {source} [{rel} (fuerza: {weight})] {target}")
                else:
                    conexiones.append(f"- {source} [{rel}] {target}")

        if conexiones:
            # Eliminar posibles duplicados preservando el orden por peso
            conexiones_unicas = list(dict.fromkeys(conexiones))
            contexto_grafo = "\n".join(conexiones_unicas)
            palabras_clave_grafo = list(nodos_vistos)
            _log.info(
                "Se inyectaron %d conexiones de Memgraph al contexto.",
                len(conexiones_unicas),
            )

        return contexto_grafo, palabras_clave_grafo

    except Exception as e:
        _log.error("Error al consultar el Grafo de Conocimiento en Memgraph: %s", e)
        return contexto_grafo, palabras_clave_grafo
