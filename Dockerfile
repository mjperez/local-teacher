# Usamos Python 3.11 slim como base
FROM python:3.11-slim

# Evitamos que Python escriba archivos .pyc y forzamos stdout sin buffer
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalamos dependencias del sistema requeridas por OpenCV (EasyOCR), Whisper (ffmpeg) y Docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos requerimientos primero para cachear la capa de instalación
COPY requirements.txt .

# Instalamos dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto de forma explícita
COPY local_teacher/ local_teacher/
COPY teacher.sh .
COPY teacher.bat .
COPY evals/ evals/

# Comando por defecto (se puede sobreescribir al hacer docker run)
ENTRYPOINT ["python", "-m", "local_teacher.cli"]
CMD ["--help"]
