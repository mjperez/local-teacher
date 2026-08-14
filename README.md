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

Levanta Qdrant, Redis y Ollama:

```bash
docker compose up -d
```

> **Nota:** El archivo `docker-compose.yml` incluye un contenedor `ollama-init` que se encargará automáticamente de descargar los modelos `deepseek-r1:8b` y `nomic-embed-text` la primera vez que levantes el entorno. Ya no es necesario ejecutar `ollama pull` manualmente.

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

### Opciones completas del CLI

El script principal soporta los siguientes argumentos:

**Operaciones principales:**
- `--ingest <ruta>`: Archivo o directorio a ingerir.
- `--query <texto>`: Pregunta a realizar al tutor.
- `--recreate`: Recrea la colección en Qdrant (útil si los datos están desactualizados).
- `--graph`: Extrae y construye un Grafo de Conocimiento para usar GraphRAG.

**Extracción y procesamiento:**
- `--figuras`: Extrae imágenes (PNG).
- `--tablas`: Extrae tablas (a CSV y Markdown).
- `--no-formulas`: Desactiva el uso del VLM de Docling para decodificar fórmulas (acelera el proceso).

**Generación y Modelos:**
- `--web-fallback`: Permite que el sistema consulte a la web (Tavily) si la respuesta no está en el contexto local.
- `--provider <nombre>`: Proveedor LLM (`ollama`, `openai`, `claude`, `gemini`).
- `--ollama-llm <modelo>`: Nombre del modelo LLM de Ollama (ej. `llama3.2`).
- `--ollama-embed <modelo>`: Nombre del modelo de embedding (ej. `nomic-embed-text`).
- `--ollama-host <url>`: URL del servidor de Ollama.

## Ingesta programática

```python
from local_teacher.ingestion.loader import cargar_archivos

docs = cargar_archivos(
    "test_docs",
    extraer_figuras=True,
    extraer_tablas=True,
    enriquecer_formulas=True # Por defecto usa el VLM de Docling
)
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

## Estructura del Proyecto (Domain-Driven)

El proyecto está organizado por dominios funcionales para facilitar su escalabilidad:

- **`ingestion/`**: Extrae datos de documentos (`loader.py`), los divide (`chunker.py`) y construye grafos (`graph_builder.py`).
- **`query/`**: Orquesta la interacción con el usuario (`retriever.py`), reescribe preguntas (`optimizer.py`) y evalúa alucinaciones (`critic.py`).
- **`storage/`**: Gestiona las conexiones a bases de datos vectoriales (`qdrant_store.py`) y cachés exactos L1 (`redis_cache.py`).
- **`evaluation/`**: Herramientas para generar datasets de prueba de forma sintética (`generate_dataset.py`) y evaluar el rendimiento del RAG (`eval.py`).

## Tecnologías de Almacenamiento

- **Qdrant**: Base de datos vectorial persistente utilizada para búsquedas semánticas (Embedding search).
- **Redis**: Caché L1 volátil para "Exact Match" (MD5) de consultas repetidas, acelerando las respuestas al saltarse el procesamiento del LLM.

## Objetivo del MVP

Un tutor LLM utiliza la colección local de Qdrant para responder preguntas
sobre el material, indicar cuándo no hay contexto suficiente, soportar Timeouts asíncronos y mostrar las
fuentes utilizadas, todo de manera 100% local y modular.
