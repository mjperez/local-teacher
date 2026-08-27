from unittest.mock import MagicMock
from local_teacher.graph.client import MemgraphClient
from local_teacher.query.graph_search import obtener_contexto_grafo
from local_teacher.ingestion.graph_builder import build_knowledge_graph
from langchain_core.documents import Document


def test_memgraph_client_query():
    client = MemgraphClient(uri="bolt://localhost:7687")
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = [
        MagicMock(data=lambda: {"source": "Sensor", "rel": "CO_OCCURS_WITH", "target": "Actuador", "weight": 3, "community": 1})
    ]
    mock_session.run.return_value = mock_result
    mock_driver.session.return_value.__enter__.return_value = mock_session
    client._driver = mock_driver

    res = client.execute_query("MATCH (a)-[r]->(b) RETURN a, r, b")
    assert len(res) == 1
    assert res[0]["source"] == "Sensor"
    assert res[0]["target"] == "Actuador"


def test_obtener_contexto_grafo_con_memgraph():
    mock_client = MagicMock()
    mock_client.execute_query.return_value = [
        {"source": "Sensor", "rel1": "USES_TOOL", "intermediate": "Arduino", "rel2": "CO_OCCURS_WITH", "target": "Actuador", "total_weight": 6.0},
        {"source": "Sensor", "rel1": "CO_OCCURS_WITH", "intermediate": "Placa", "rel2": "IMPLEMENTS", "target": "Microcontrolador", "total_weight": 3.0},
    ]

    contexto, keywords = obtener_contexto_grafo(["sensor"], client=mock_client)

    assert "Sensor [USES_TOOL] Arduino [CO_OCCURS_WITH] Actuador (peso: 6.0)" in contexto
    assert "Sensor [CO_OCCURS_WITH] Placa [IMPLEMENTS] Microcontrolador (peso: 3.0)" in contexto
    assert "Sensor" in keywords
    assert "Arduino" in keywords
    assert "Actuador" in keywords
    assert "Placa" in keywords
    assert "Microcontrolador" in keywords


def test_obtener_contexto_grafo_vacio():
    contexto, keywords = obtener_contexto_grafo([])
    assert contexto == ""
    assert keywords == []
