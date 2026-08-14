# Local Teacher

Tutor educativo local-first basado en RAG (Retrieval-Augmented Generation).

El sistema procesa material didáctico estructurado, indexa su contenido en bases vectoriales locales y permite realizar consultas interactivas mediante modelos LLM. El procesamiento se ejecuta en el equipo del usuario sin enviar datos a servicios externos.

---

## Características Principales

- **Procesamiento de Documentos con Docling**: Extracción de texto, tablas en Markdown/CSV y figuras con leyendas desde archivos PDF, DOCX, PPTX, Markdown y JSONL. Normalización de fórmulas matemáticas en dos pasadas.
- **Búsqueda Híbrida y Reranking**: Combinación de búsqueda vectorial densa con vectores dispersos BM25 en Qdrant, refinada mediante un reranker local con **FlashRank** (`ms-marco-MiniLM-L-12-v2`).
- **Control de Fidelidad y Abstención**: Clasificador binario Self-RAG para evitar alucinaciones y umbral de score mínimo para indicar explícitamente cuándo el tema consultado no está en el material.
- **Grafo de Conocimiento (GraphRAG)**: Extracción de tripletas entidad-relación y enriquecimiento de contexto mediante grafos conceptuales.
- **Modo Tutor Interactivo**: CLI multi-turno con memoria conversacional para responder dudas de seguimiento y presentación detallada de fuentes citadas.
- **Optimización de Hardware**: Descarga automática de modelos de embeddings de la memoria tras vectorizar (`keep_alive=0`) y soporte de caché semántico en Redis.

---

## Requisitos

- Python 3.11 o superior
- Docker y Docker Compose
- Soporte GPU opcional para acelerar Docling y Ollama

---

## Instalación y Configuración

1. Instala las dependencias del proyecto:
   ```bash
   pip install -r requirements.txt
   ```

2. Configura las variables de entorno:
   ```bash
   cp .env.example .env
   ```

3. Inicia los servicios de infraestructura (Qdrant, Redis y Ollama):
   ```bash
   docker compose up -d
   ```
   El contenedor `ollama-init` descargará de forma automática los modelos `deepseek-r1:8b` y `nomic-embed-text`.

---

## Uso de la CLI

### Ingesta de Documentos
Para procesar e indexar documentos ubicados en una carpeta:

```bash
python -m local_teacher.cli \
  --ingest test_docs \
  --figuras --tablas \
  --recreate
```

### Consultas Interactivas
Para interactuar con el tutor en consola:

```bash
python -m local_teacher.cli --query "¿Qué es un sistema inercial?"
```

El comando abrirá un diálogo interactivo donde puedes escribir preguntas de seguimiento manteniendo el contexto de la conversación.

### Opciones de la Línea de Comandos

- `--ingest <ruta>`: Ruta al archivo o carpeta a procesar.
- `--query <texto>`: Pregunta inicial para el tutor.
- `--figuras`: Extrae imágenes y diagramas a disco en formato PNG.
- `--tablas`: Extrae tablas a formatos Markdown y CSV.
- `--no-formulas`: Omite el modelo de enriquecimiento de fórmulas para acelerar la carga.
- `--recreate`: Recrea la colección en Qdrant desde cero.
- `--graph`: Genera el Grafo de Conocimiento a partir de los documentos.
- `--web-fallback`: Habilita búsqueda en internet si la información no existe localmente.
- `--provider <nombre>`: Proveedor LLM (`ollama`, `openai`).
- `--ollama-llm <modelo>`: Nombre del modelo de generación (por defecto `deepseek-r1:8b` o `llama3.2`).
- `--ollama-embed <modelo>`: Modelo de embeddings (por defecto `nomic-embed-text`).
- `--ollama-host <url>`: URL del servidor Ollama (por defecto `http://127.0.0.1:11434`).

---

## Pruebas Automatizadas

Ejecuta la suite de pruebas unitarias e integración con `pytest`:

```bash
pytest tests/
```

---

## Estructura del Código

```text
local-teacher/
├── local_teacher/
│   ├── ingestion/       # Loader (Docling), Chunker y Graph Builder
│   ├── query/           # Retriever, Optimizer, Critic y Tutor
│   ├── storage/         # Qdrant Store y Redis Cache
│   ├── evaluation/      # Generación sintética y evaluación RAG
│   ├── factory.py       # Configuración de LLM y Embeddings
│   ├── metadata.py      # Definición de tipos de metadatos
│   └── cli.py           # Interfaz de línea de comandos interactiva
├── docs/                # Documentación técnica y guías
├── tests/               # Suite de tests con pytest
├── docker-compose.yml   # Orquestación de Qdrant, Redis y Ollama
├── requirements.txt     # Dependencias de Python
└── README.md            # Este documento
```
