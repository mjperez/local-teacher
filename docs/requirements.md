# Requisitos del sistema

## Requisitos funcionales

### RF-01 — Ingesta de documentos

El sistema debe aceptar un archivo o directorio y procesar únicamente extensiones soportadas, informando de forma clara los archivos omitidos o fallidos.

### RF-02 — Extracción de PDF

El sistema debe extraer texto de PDFs digitales y conservar la página de origen. Los backends avanzados de extracción serán opcionales.

### RF-03 — Figuras y diagramas

Cuando Docling detecte una figura con caption, el sistema debe asociarla con:

- documento de origen;
- página;
- caption;
- capítulo o sección cercana;
- ruta de la imagen exportada.

El manifest `captions.json` puede generarse para auditoría, pero la asociación operativa debe almacenarse en los metadatos indexados.

### RF-04 — Tablas

El sistema podrá extraer tablas como Markdown y CSV. Las tablas solo se indexarán si la extracción supera las validaciones mínimas de estructura.

### RF-05 — Normalización y fragmentación

El sistema debe convertir el contenido en documentos y fragmentos con metadata heredada. Los fragmentos deben conservar la trazabilidad hasta el documento original.

### RF-06 — Embeddings

El sistema debe permitir seleccionar un modelo de embeddings local. Para el MVP se priorizan embeddings textuales locales; las figuras se recuperan inicialmente mediante caption y contexto textual.

Los embeddings visuales tipo CLIP quedan fuera del MVP y deberán incorporarse como una capacidad separada, no mezclada con el espacio vectorial textual sin una estrategia explícita.

### RF-07 — Almacenamiento

El sistema debe guardar y consultar los fragmentos en Qdrant local, con colección configurable y opción explícita de recrear el índice.

### RF-08 — Recuperación

El sistema debe recuperar los fragmentos más relevantes para una pregunta y conservar sus metadatos de origen. El número de resultados debe ser configurable.

### RF-09 — Tutor

El tutor debe generar respuestas basadas en el contexto recuperado. Si el material no contiene información suficiente, debe indicarlo y no inventar una respuesta.

El tutor debe poder operar, como mínimo, en estos modos:

- explicar;
- resumir;
- comparar;
- generar preguntas de repaso.

### RF-10 — Fuentes

Cada respuesta debe mostrar las fuentes recuperadas, incluyendo como mínimo documento y página cuando estén disponibles. Para figuras y tablas debe mostrar también el tipo y la ruta o identificador correspondiente.

### RF-11 — Proveedores de modelos

El sistema debe soportar una interfaz común para:

- Ollama, como proveedor local principal;
- OpenAI, mediante API key opcional;
- Anthropic/Claude, mediante API key opcional;
- Google Gemini, mediante API key opcional.

La configuración del proveedor y de los modelos debe poder realizarse mediante variables de entorno y, cuando corresponda, argumentos de CLI.

### RF-12 — Interfaz inicial

La primera interfaz será una CLI con operaciones separadas para:

- ingerir;
- consultar;
- recrear la colección;
- seleccionar proveedor y backend de extracción.

## Requisitos no funcionales

### RNF-01 — Local-first

El flujo principal debe funcionar sin enviar documentos a servicios externos. Las APIs externas serán opt-in. La verificación del MVP debe ejecutarse con red externa bloqueada y Ollama/Qdrant accesibles solo localmente.

### RNF-02 — Instalación mínima

Las dependencias obligatorias deben ser las mínimas necesarias para el flujo local. Docling, Marker, proveedores externos y funcionalidades futuras deben ser opcionales cuando sea técnicamente posible.

### RNF-03 — Trazabilidad

Los datos indexados deben permitir responder de dónde proviene cada fragmento recuperado.

### RNF-04 — Reproducibilidad

La instalación y ejecución deben estar documentadas para Windows y Python 3.11+.

### RNF-05 — Errores explícitos

Los errores de conexión, credenciales, extracción e indexación deben producir mensajes accionables. No se debe presentar una ingesta parcial como exitosa sin advertencia.

### RNF-06 — Idempotencia futura

La ingesta repetida del mismo archivo debe evitar duplicados mediante identificadores estables derivados del documento, página, tipo de contenido y fragmento.

### RNF-07 — Testabilidad

Las capas de carga, fragmentación, recuperación y tutor deben poder probarse sin requerir siempre Ollama, APIs externas o un Qdrant real.

## Criterios de aceptación del MVP

- Se puede instalar el proyecto en una máquina Windows con Python 3.11+.
- Qdrant local se inicia con Docker Compose.
- Ollama proporciona el modelo de chat y el embedding local.
- Un PDF educativo se ingiere conservando páginas y figuras con caption.
- Una consulta recupera contexto relevante.
- El tutor responde usando ese contexto.
- La respuesta incluye fuentes.
- Una pregunta fuera del material produce una respuesta de insuficiencia de contexto.
- La suite de tests pasa sin credenciales externas.
