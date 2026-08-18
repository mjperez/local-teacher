import sqlite3
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


class StateManager:
    """Gestor de estado persistente con SQLite para reanudar tareas de ingesta."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            root_dir = Path(__file__).parent.parent.parent
            data_dir = root_dir / "data"
            data_dir.mkdir(exist_ok=True)
            self.db_path = data_dir / "checkpoint.db"
        else:
            self.db_path = Path(db_path)
            
        self._init_db()

    def _obtener_conexion(self) -> sqlite3.Connection:
        """Crea una conexión con timeout extendido y modo WAL activado."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Inicializa las tablas necesarias si no existen."""
        try:
            with self._obtener_conexion() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processed_files (
                        file_hash TEXT PRIMARY KEY
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS qdrant_batches (
                        batch_id INTEGER PRIMARY KEY
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graph_chunks (
                        chunk_index INTEGER PRIMARY KEY
                    )
                """)
                conn.commit()
        except Exception as e:
            _log.error("Error al inicializar la base de datos de checkpoints SQLite: %s", e)

    # Hashes de Archivos
    def get_processed_files(self) -> set[str]:
        """Obtiene el conjunto de hashes de archivos ya procesados."""
        try:
            with self._obtener_conexion() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_hash FROM processed_files")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            _log.warning("No se pudieron leer los archivos procesados: %s", e)
            return set()

    def mark_file_processed(self, file_hash: str) -> None:
        """Registra un archivo como completado mediante su hash SHA-256."""
        try:
            with self._obtener_conexion() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO processed_files (file_hash) VALUES (?)",
                    (file_hash,),
                )
                conn.commit()
        except Exception as e:
            _log.warning("Error al guardar hash del archivo procesado: %s", e)

    # Lotes de Qdrant
    def get_qdrant_batches(self) -> set[int]:
        """Obtiene el conjunto de lotes indexados en Qdrant."""
        try:
            with self._obtener_conexion() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT batch_id FROM qdrant_batches")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            _log.warning("No se pudieron leer los lotes de Qdrant: %s", e)
            return set()

    def mark_qdrant_batch(self, batch_id: int) -> None:
        """Registra un lote de Qdrant como completado."""
        try:
            with self._obtener_conexion() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO qdrant_batches (batch_id) VALUES (?)",
                    (batch_id,),
                )
                conn.commit()
        except Exception as e:
            _log.warning("Error al guardar lote de Qdrant: %s", e)

    # Fragmentos del Grafo
    def get_graph_chunks(self) -> set[int]:
        """Obtiene los índices de fragmentos ya procesados en el grafo."""
        try:
            with self._obtener_conexion() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_index FROM graph_chunks")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            _log.warning("No se pudieron leer los fragmentos del grafo: %s", e)
            return set()

    def mark_graph_chunk(self, chunk_index: int) -> None:
        """Registra un fragmento del grafo como completado."""
        try:
            with self._obtener_conexion() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO graph_chunks (chunk_index) VALUES (?)",
                    (chunk_index,),
                )
                conn.commit()
        except Exception as e:
            _log.warning("Error al guardar fragmento del grafo: %s", e)

    def clear(self) -> None:
        """Limpia todos los checkpoints registrados eliminando las filas de las tablas."""
        try:
            with self._obtener_conexion() as conn:
                conn.execute("DELETE FROM processed_files")
                conn.execute("DELETE FROM qdrant_batches")
                conn.execute("DELETE FROM graph_chunks")
                conn.commit()
        except Exception as e:
            _log.warning("Error al limpiar checkpoints: %s", e)


_global_state_manager = StateManager()


def get_state_manager() -> StateManager:
    """Devuelve la instancia global del gestor de estado."""
    return _global_state_manager
