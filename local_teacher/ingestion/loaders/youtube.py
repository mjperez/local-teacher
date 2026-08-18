import logging
import re

_log = logging.getLogger(__name__)

def extraer_transcripcion_youtube(url: str) -> str | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not match:
            return None
        video_id = match.group(1)
        
        if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            
        try:
            transcript = transcript_list.find_transcript(['es', 'en'])
        except Exception:
            transcript = next(iter(transcript_list))
            
        texto_list = transcript.fetch()
        
        if texto_list and hasattr(texto_list[0], 'text'):
            texto = " ".join([t.text for t in texto_list])
        else:
            texto = " ".join([t['text'] for t in texto_list])
            
        return texto
    except Exception as e:
        _log.warning(f"No se pudo extraer transcripción de YouTube para {url}: {e}")
        return None
