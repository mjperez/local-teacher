import logging
from pathlib import Path
import kuzu

_log = logging.getLogger(__name__)


def obtener_contexto_grafo(
    entidades_filtro: list[str], db_path: str | Path = "./local_teacher_kuzu"
) -> tuple[str, list[str]]:
    """Consulta el Grafo de Conocimiento (Kùzu) y expande la búsqueda con entidades relacionadas."""
    contexto_grafo = "(No se detectaron entidades o no hay grafo disponible)"
    palabras_clave_grafo: list[str] = []

    if not entidades_filtro:
        return contexto_grafo, palabras_clave_grafo

    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        return contexto_grafo, palabras_clave_grafo

    try:
        db = kuzu.Database(str(db_path_obj))
        conn = kuzu.Connection(db)

        try:
            conn.execute("MATCH (n:Entity) RETURN n LIMIT 1")
        except RuntimeError:
            return contexto_grafo, palabras_clave_grafo

        conexiones = []
        nodos_encontrados = set()

        for entidad in entidades_filtro:
            res = conn.execute(
                "MATCH (n:Entity) WHERE n.name CONTAINS $ent RETURN n.name",
                parameters={"ent": entidad},
            )
            while res.has_next():
                nodo = res.get_next()[0]
                nodos_encontrados.add(nodo)

        for nodo in nodos_encontrados:
            palabras_clave_grafo.append(nodo)

            # Relaciones salientes
            out_res = conn.execute(
                "MATCH (a:Entity {name: $n})-[r:Rel]->(b:Entity) RETURN a.name, r.type, b.name",
                parameters={"n": nodo},
            )
            while out_res.has_next():
                origen, rel, destino = out_res.get_next()
                palabras_clave_grafo.append(destino)
                conexiones.append(f"- {origen} [{rel}] {destino}")

            # Relaciones entrantes
            in_res = conn.execute(
                "MATCH (a:Entity)-[r:Rel]->(b:Entity {name: $n}) RETURN a.name, r.type, b.name",
                parameters={"n": nodo},
            )
            while in_res.has_next():
                origen, rel, destino = in_res.get_next()
                palabras_clave_grafo.append(origen)
                conexiones.append(f"- {origen} [{rel}] {destino}")

        conexiones_unicas = list(set(conexiones))
        if conexiones_unicas:
            contexto_grafo = "\n".join(conexiones_unicas)
            _log.info(
                "Se inyectaron %d conexiones del Grafo de Conocimiento al contexto.",
                len(conexiones_unicas),
            )

        return contexto_grafo, list(set(palabras_clave_grafo))

    except Exception as e:
        _log.error("Error al consultar la base de datos de grafos Kùzu: %s", e)
        return contexto_grafo, palabras_clave_grafo
