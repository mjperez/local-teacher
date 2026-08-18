import pytest
from pathlib import Path
from local_teacher.ingestion.state_manager import StateManager


def test_state_manager_flujo_completo(tmp_path: Path):
    db_file = tmp_path / "test_checkpoint.db"
    sm = StateManager(db_path=db_file)

    # 1. Hashes de archivos
    assert len(sm.get_processed_files()) == 0
    sm.mark_file_processed("hash_abc123")
    sm.mark_file_processed("hash_xyz789")
    sm.mark_file_processed("hash_abc123")  # Duplicado debe ser ignorado

    archivos = sm.get_processed_files()
    assert len(archivos) == 2
    assert "hash_abc123" in archivos
    assert "hash_xyz789" in archivos

    # 2. Lotes de Qdrant
    assert len(sm.get_qdrant_batches()) == 0
    sm.mark_qdrant_batch(1)
    sm.mark_qdrant_batch(2)
    sm.mark_qdrant_batch(1)

    lotes = sm.get_qdrant_batches()
    assert len(lotes) == 2
    assert 1 in lotes
    assert 2 in lotes

    # 3. Fragmentos del Grafo
    assert len(sm.get_graph_chunks()) == 0
    sm.mark_graph_chunk(10)
    sm.mark_graph_chunk(20)

    chunks = sm.get_graph_chunks()
    assert len(chunks) == 2
    assert 10 in chunks
    assert 20 in chunks

    # 4. Limpieza (clear)
    sm.clear()
    assert len(sm.get_processed_files()) == 0
    assert len(sm.get_qdrant_batches()) == 0
    assert len(sm.get_graph_chunks()) == 0
