import pytest
from unittest.mock import MagicMock
import youtube_transcript_api
from local_teacher.ingestion.loaders.youtube import extraer_transcripcion_youtube


def test_extraer_transcripcion_url_invalida():
    assert extraer_transcripcion_youtube("https://sitio-invalido.com/video") is None


def test_extraer_transcripcion_youtube_mock(monkeypatch):
    mock_item = MagicMock()
    mock_item.text = "Hola mundo de prueba"

    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = [mock_item]

    mock_list = MagicMock()
    mock_list.find_transcript.return_value = mock_transcript

    class MockApi:
        @classmethod
        def list_transcripts(cls, video_id):
            return mock_list

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", MockApi)

    resultado = extraer_transcripcion_youtube("https://www.youtube.com/watch?v=srozR_Vw97k")
    assert resultado is not None
    assert "Hola mundo de prueba" in resultado
