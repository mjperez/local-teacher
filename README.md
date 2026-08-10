# Local Teacher

Sistema básico de RAG (Retrieval-Augmented Generation) para procesar documentos (`.txt`, `.md`, `.jsonl`, `.pdf`) y hacer consultas con modelos locales (Ollama) o APIs (OpenAI). Proyecto de aprendizaje.

## Instalación

> Requiere **Python 3.11**.

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Levanta Qdrant con Docker Compose:

```bash
docker compose up -d
```

> Opcional: crea un `.env` en la raíz con `OPENAI_API_KEY`, `OLLAMA_HOST` o `QDRANT_URL`.

## Uso

Ejecuta el CLI con `python -m local_teacher.cli`.

### Ingestar documentos y hacer una consulta

Con Ollama (primero descarga los modelos, por ejemplo `ollama pull nomic-embed-text`):

```bash
python -m local_teacher.cli --ingest test_docs --query "Tu pregunta aquí" --provider ollama --ollama-llm qwen3.5:4b
```

Con OpenAI:

```bash
python -m local_teacher.cli --ingest test_docs --query "Tu pregunta aquí" --provider openai
```

### Ingestar PDFs

Los PDFs se procesan con Marker si está instalado (mejor estructura para tablas y diagramas). Si no, se usa PyPDF como fallback.

```bash
python -m local_teacher.cli --ingest ./docs --query "Tu pregunta aquí" --provider openai
```

### Consultar sin re-ingestar

Si los documentos ya están indexados, omite `--ingest`:

```bash
python -m local_teacher.cli --query "Tu pregunta aquí" --provider ollama
```

## Roadmap

- [ ] **Modo Tutor (Próximamente)**: capa pedagógica para responder como profesor (explicaciones, resúmenes y quizzes).
