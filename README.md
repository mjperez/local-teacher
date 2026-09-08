# Local Teacher

Tutor educativo **local-first** basado en RAG (Retrieval-Augmented Generation) con soporte GraphRAG.

El sistema procesa material didáctico estructurado, lo indexa en bases vectoriales y de grafos locales, y responde consultas interactivas mediante LLM. Todo el procesamiento se ejecuta en tu equipo: ningún documento sale hacia servicios externos.

---

## Características

- **Ingesta de documentos con Docling**: extracción de texto, tablas (Markdown/CSV) e imágenes con leyendas desde PDF, DOCX, PPTX, Markdown y JSONL. Normalización de fórmulas matemáticas opcional (VLM).
- **Búsqueda híbrida y reranking**: búsqueda densa combinada con vectores dispersos BM25 en Qdrant, refinada con un reranker local (**FlashRank**, `ms-marco-MiniLM-L-12-v2`).
- **Grafo de conocimiento (GraphRAG)**: extracción de entidades con **GLiNER**, búsqueda semántica de 2 saltos y detección de comunidades sobre **Memgraph MAGE** (Louvain).
- **Control de fidelidad y abstención segura**: crítico interno Self-RAG (`granite3-guardian:2b`) que valida cada respuesta contra el contexto recuperado. Si el tema no está en el material, el tutor lo dice explícitamente en lugar de alucinar.
- **Optimización de consultas**: reescritura de la pregunta con un modelo rápido auxiliar antes de la recuperación.
- **Búsqueda web condicionada**: si la información local no basta, puede recurrir a DuckDuckGo (opcional, con filtro configurable, ej. `--web 'site:edu'`).
- **Caché semántica en Redis**: respuestas repetidas se sirven sin re-ejecutar el pipeline.
- **Modos de ejecución**: ajusta calidad vs. velocidad con `--mode exact | fast | ultra-fast`.

---

## Requisitos

- Python 3.11 o superior
- Docker y Docker Compose
- GPU opcional para acelerar Docling y Ollama

## Instalación

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Configura las variables de entorno:

   ```bash
   cp .env.example .env
   ```

3. Levanta la infraestructura (Qdrant, Redis, Memgraph MAGE, Memgraph Lab y Ollama):

   ```bash
   docker compose up -d
   ```

   El contenedor `ollama-init` descarga automáticamente los modelos de embeddings y generación definidos en la configuración. Memgraph Lab queda disponible en `http://localhost:3000` para explorar el grafo.

---

## Uso

### Ingesta de documentos

Procesa e indexa un archivo o carpeta (documentos + grafo de conocimiento):

```bash
python -m local_teacher.cli ingest test_docs --workers 4
```

### Consultas

```bash
python -m local_teacher.cli query "¿Qué es un sistema inercial?"
```

### Opciones principales

| Flag | Descripción |
|---|---|
| `--mode {exact,fast,ultra-fast}` | `exact` usa `deepseek-r1:8b` + crítico, `fast` usa `llama3.2`, `ultra-fast` omite el crítico. |
| `--no-figures` / `--no-tables` | Desactiva la extracción de imágenes o tablas (activadas por defecto). |
| `--with-formulas` | Habilita el modelo VLM de Docling para fórmulas (más lento). |
| `--no-graph` | Omite la construcción del grafo de conocimiento. |
| `--no-critic` | Desactiva el supervisor Self-RAG. |
| `--no-cache` | Ignora la caché semántica de Redis y recalcula la respuesta. |
| `--web [filtro]` | Habilita búsqueda web opcional, p. ej. `--web 'site:edu'`. |
| `--recreate` | Recrea la colección en Qdrant desde cero. |
| `--workers N` | Procesos concurrentes para la extracción de documentos. |
| `--provider` | Proveedor LLM (`ollama`, `openai`). |
| `--embed-provider` | Motor de embeddings: `fastembed` (ONNX, por defecto) u `ollama`. |
| `--ollama-llm` / `--ollama-critic-llm` / `--ollama-fast-llm` | Modelos de generación, crítico y optimización de consultas. |
| `--ollama-host` | URL del servidor Ollama. |

---

## Pruebas y evaluaciones

### Tests unitarios

```bash
pytest tests/
```

### Evaluaciones de fidelidad (groundedness)

Los scripts de benchmark y evaluación están en `evals/`. Por ejemplo, para evaluar al crítico interno:

```bash
python evals/eval_guardian.py --limit 3
```

---

## Arquitectura

```text
local-teacher/
├── local_teacher/
│   ├── ingestion/       # Loaders (Docling), Chunker y Graph Builder (GLiNER)
│   ├── query/           # Pipeline RAG: Retriever, Optimizer, Critic, Graph Search y Tutor
│   ├── graph/           # Cliente de Memgraph (MAGE / Louvain)
│   ├── storage/         # Qdrant Store y Redis Cache
│   ├── factory.py       # Configuración de LLM y embeddings
│   ├── metadata.py      # Tipos de metadatos
│   ├── metrics.py       # Métricas del pipeline
│   ├── repl.py          # Sesión interactiva de chat
│   └── cli.py           # Interfaz de línea de comandos
├── docs/                # Documentación técnica y guías
├── evals/               # Benchmarks y evaluación de fidelidad
├── tests/               # Suite de tests con pytest
├── docker-compose.yml   # Qdrant, Redis, Memgraph MAGE, Memgraph Lab y Ollama
└── requirements.txt     # Dependencias de Python
```

### Flujo de una consulta

1. **Optimización**: un modelo rápido reescribe la pregunta del usuario.
2. **Recuperación híbrida**: búsqueda densa + BM25 en Qdrant, más búsqueda semántica de 2 saltos en el grafo de conocimiento.
3. **Reranking**: FlashRank reordena los candidatos.
4. **Generación**: el LLM principal responde con streaming.
5. **Crítica**: el supervisor Self-RAG valida la respuesta contra el contexto; si no es fiel o el tema no existe en el material, se abstiene con un mensaje de fallback.
