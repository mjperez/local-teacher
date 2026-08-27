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
    contexto_grafo = ""
    palabras_clave_grafo: list[str] = []

    if not entidades_filtro:
        return contexto_grafo, palabras_clave_grafo

    # Limpiar y normalizar entidades
    entidades_limpias = [e.strip() for e in entidades_filtro if e and len(e.strip()) > 1]
    if not entidades_limpias:
        return contexto_grafo, palabras_clave_grafo

    try:
        memgraph = client or get_memgraph_client()

        # 2-hop search con bono por coincidencia de comunidad
        query = """
        MATCH (a:Entity)-[r1:Rel]-(b:Entity)-[r2:Rel]-(c:Entity)
        WHERE ANY(ent IN $entidades WHERE toLower(a.name) CONTAINS toLower(ent))
          AND a <> c
        WITH a, r1, b, r2, c,
             (COALESCE(r1.weight, 1) + COALESCE(r2.weight, 1)) * (CASE WHEN a.community = c.community AND a.community IS NOT NULL THEN 1.5 ELSE 1.0 END) AS total_weight
        ORDER BY total_weight DESC
        LIMIT $limit
        RETURN a.name AS source, 
               COALESCE(r1.type, 'RELATED_TO') AS rel1, 
               b.name AS intermediate, 
               COALESCE(r2.type, 'RELATED_TO') AS rel2, 
               c.name AS target, 
               total_weight
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
            rel1 = rec.get("rel1")
            inter = rec.get("intermediate")
            rel2 = rec.get("rel2")
            target = rec.get("target")
            weight = rec.get("total_weight", 1)

            if source:
                nodos_vistos.add(source)
            if inter:
                nodos_vistos.add(inter)
            if target:
                nodos_vistos.add(target)

            if source and inter and target:
                conexiones.append(f"- {source} [{rel1}] {inter} [{rel2}] {target} (peso: {weight:.1f})")

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
