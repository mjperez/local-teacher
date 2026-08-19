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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS meta_data (
                        key TEXT PRIMARY KEY,
                        value REAL
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
                conn.execute("DELETE FROM meta_data WHERE key='cumulative_time'")
                conn.commit()
        except Exception as e:
            _log.warning("Error al limpiar checkpoints: %s", e)

    # Tiempo Acumulado
    def get_cumulative_time(self) -> float:
        """Obtiene el tiempo parcial acumulado de sesiones anteriores."""
        try:
            with self._obtener_conexion() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM meta_data WHERE key='cumulative_time'")
                row = cursor.fetchone()
                return row[0] if row else 0.0
        except Exception as e:
            _log.warning("No se pudo leer el tiempo acumulado: %s", e)
            return 0.0

    def add_cumulative_time(self, seconds: float) -> None:
        """Suma tiempo al contador acumulado (para cuando se presiona Ctrl+C)."""
        try:
            with self._obtener_conexion() as conn:
                conn.execute("""
                    INSERT INTO meta_data (key, value) VALUES ('cumulative_time', ?)
                    ON CONFLICT(key) DO UPDATE SET value=value+?
                """, (seconds, seconds))
                conn.commit()
        except Exception as e:
            _log.warning("Error al guardar tiempo acumulado: %s", e)

    def reset_cumulative_time(self) -> None:
        """Reinicia el tiempo acumulado (al finalizar exitosamente)."""
        try:
            with self._obtener_conexion() as conn:
                # Upsert: garantiza que la fila exista aunque haya sido borrada por clear().
                conn.execute("""
                    INSERT INTO meta_data (key, value) VALUES ('cumulative_time', 0.0)
                    ON CONFLICT(key) DO UPDATE SET value=0.0
                """)
                conn.commit()
        except Exception as e:
            _log.warning("Error al resetear tiempo acumulado: %s", e)


_global_state_manager = None


def get_state_manager() -> StateManager:
    """Devuelve la instancia global (singleton) del gestor de estado de producción."""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = StateManager()
    return _global_state_manager


def create_state_manager(db_path: Path | str) -> StateManager:
    """Crea una instancia AISLADA del gestor de estado.

    Usar exclusivamente en tests y benchmarks para evitar mutar el singleton
    global de producción.
    """
    return StateManager(db_path=db_path)


def override_state_manager(instance: StateManager) -> None:
    """Reemplaza el singleton global con una instancia específica.

    Usar en tests/benchmarks ANTES de llamar a cualquier módulo de ingesta
    para garantizar que todos usen la BD de checkpoints aislada.
    Llamar a `get_state_manager()` sin argumentos restaura el comportamiento
    normal al siguiente ciclo si la instancia se pone a None externamente.
    """
    global _global_state_manager
    _global_state_manager = instance
