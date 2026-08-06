# Local Teacher

Sistema básico de RAG (Retrieval-Augmented Generation) modular para procesar documentos (.txt, .md, .jsonl) y realizar consultas utilizando modelos locales (Ollama) o APIs (OpenAI). Proyecto de aprendizaje.

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Levanta la base de datos vectorial (Qdrant) con Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

*(Opcional: configura un archivo `.env` en la raíz con tus variables como `OPENAI_API_KEY` u `OLLAMA_HOST` si es necesario).*

## Uso

El script principal es `cli.py`.

### Ingestar documentos y hacer una consulta

Usando Ollama (requiere descargar los modelos antes, ej. `ollama pull nomic-embed-text`):
```bash
python local_teacher/cli.py --ingest test_docs --query "Tu pregunta aquí" --provider ollama --ollama-llm qwen3.5:4b
```

Usando OpenAI:
```bash
python local_teacher/cli.py --ingest test_docs --query "Tu pregunta aquí" --provider openai
```

### Consultar sin re-ingestar
Si los documentos ya están en la base de datos, omite el parámetro `--ingest`:
```bash
python local_teacher/cli.py --query "Tu pregunta aquí" --provider ollama
```

## Roadmap

- [ ] **Modo Tutor (Próximamente)**: Implementación de una capa pedagógica para que el sistema no solo responda de forma directa, sino que actúe como un profesor (con modos de explicación detallada, resúmenes y quizzes).
