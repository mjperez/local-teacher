import logging
from pathlib import Path
from langchain_core.documents import Document

_log = logging.getLogger(__name__)

def cargar_video_local(ruta: Path, curso: str | None = None) -> list[Document]:
    import whisper
    import cv2
    import easyocr
    from difflib import SequenceMatcher
    
    docs = []
    _log.info(f"Procesando video local: {ruta.name}")
    
    # 1. Extraer audio temporal
    audio_path = ruta.with_suffix(".wav")
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", str(ruta), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Transcribir audio
        _log.info("Transcribiendo audio del video...")
        model = whisper.load_model("base")
        transcription = model.transcribe(str(audio_path), language="es")
        texto_audio = transcription["text"]
        
        meta_audio = {"fuente": str(ruta), "tipo_archivo": "video_audio"}
        if curso:
            meta_audio["curso"] = curso
        docs.append(Document(page_content=f"Transcripción de audio ({ruta.name}):\n{texto_audio}", metadata=meta_audio))
    except Exception as e:
        _log.warning(f"Error procesando audio de {ruta.name}: {e}")
    finally:
        if audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass
            
    # 3. Procesar video (frames) para OCR
    try:
        _log.info("Procesando frames del video (OCR)...")
        reader = easyocr.Reader(['es', 'en'], gpu=True)
        cap = cv2.VideoCapture(str(ruta))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30
            
        # 1 frame cada 30 segundos
        frame_interval = int(fps * 30)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        texto_visual = []
        last_text = ""
        frame_count = 0
        
        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
                
            result = reader.readtext(frame, detail=0)
            text = " ".join(result)
            
            if text.strip():
                similitud = SequenceMatcher(None, text, last_text).ratio()
                if similitud < 0.8:
                    minuto = (frame_count / fps) / 60
                    texto_visual.append(f"[Minuto {minuto:.1f}] Texto en pantalla: {text}")
                    last_text = text
                        
            frame_count += frame_interval
            
        cap.release()
        
        if texto_visual:
            meta_visual = {"fuente": str(ruta), "tipo_archivo": "video_visual"}
            if curso:
                meta_visual["curso"] = curso
            contenido_visual = "\n".join(texto_visual)
            docs.append(Document(page_content=f"Contenido visual (diapositivas) de {ruta.name}:\n{contenido_visual}", metadata=meta_visual))
            
    except Exception as e:
        _log.warning(f"Error procesando video visualmente {ruta.name}: {e}")
        
    return docs
