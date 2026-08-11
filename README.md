# Local Teacher

Tutor educativo **local-first** basado en RAG (Retrieval-Augmented Generation).

El objetivo es ingerir material educativo, conservar su estructura y permitir
preguntas sobre el contenido mediante un tutor LLM. El MVP prioriza texto y
PDFs con figuras y tablas relacionadas con su documento y página.

El procesamiento, los embeddings y Qdrant funcionan localmente. Ollama es el
proveedor principal; OpenAI, Claude y Gemini se conectan mediante API keys
opcionales.

> Audio y vídeo quedan fuera del MVP. Se incorporarán más adelante usando el
> mismo contrato de documentos y metadatos.

> **Nota:** La interfaz actual es por línea de comandos (CLI) y soporta respuestas generadas en tiempo real (*streaming*), con fórmulas matemáticas optimizadas para lectura en consola de texto plano. Eventualmente se desarrollará una interfaz web.

## Documentación

- [Alcance](docs/scope.md)
- [Requisitos](docs/requirements.md)
- [Roadmap](docs/roadmap.md)
- [Arquitectura](docs/rag-structure.md)

## Requisitos

- Python 3.11+
- Docker + Docker Compose
- GPU opcional pero recomendada para Docling y Ollama

## Instalación

```bash
pip install -r requirements.txt
```

Levanta Qdrant y Ollama:

```bash
docker compose up -d
```

Descarga modelos locales, por ejemplo:

```bash
docker exec -it local_teacher_ollama ollama pull llama3.2
docker exec -it local_teacher_ollama ollama pull nomic-embed-text
```

## Uso

Ingestar y consultar con Ollama:

```bash
python -m local_teacher.cli \
  --ingest test_docs \
  --query "¿Qué es un sistema inercial?" \
  --provider ollama \
  --ollama-llm llama3.2
```

Ingestar figuras y tablas:

```bash
python -m local_teacher.cli \
  --ingest test_docs \
  --figuras --tablas \
  --recreate \
  --provider ollama
```

Consultar sin re-ingestar:

```bash
python -m local_teacher.cli \
  --query "¿Qué es un sistema inercial?" \
  --provider ollama
```

Con OpenAI (requiere `OPENAI_API_KEY`):

```bash
python -m local_teacher.cli \
  --ingest test_docs \
  --query "¿Qué es un sistema inercial?" \
  --provider openai
```

## Ingesta programática

```python
from local_teacher.loader import cargar_archivos

docs = cargar_archivos("test_docs", extraer_figuras=True, extraer_tablas=True)
```

## Cómo funciona la extracción

- Los PDFs se procesan con **Docling**.
- Docling extrae texto, figuras y tablas en una sola pasada.
- Las figuras se filtran por caption (`Figure N-M`, `Figura N`, etc.) para
  evitar íconos y etiquetas sueltas.
- Cada figura y tabla se convierte en un documento recuperable con metadata de
  fuente.
- Se genera `captions.json` como manifest legible para auditoría.

## Embeddings y multimodalidad

El MVP usa embeddings textuales locales. Las figuras se recuperan mediante
caption y contexto textual. CLIP u otros embeddings visuales se evaluarán en
una fase posterior.

## Objetivo del MVP

Un tutor LLM utiliza la colección local de Qdrant para responder preguntas
sobre el material, indicar cuándo no hay contexto suficiente y mostrar las
fuentes utilizadas.
