# Alcance del proyecto

## Propósito

`local-teacher` es un tutor basado en RAG (Retrieval-Augmented Generation) para estudiar material educativo. El sistema ingiere documentos, conserva su estructura y permite hacer preguntas sobre el material mediante un LLM.

El proyecto sigue una estrategia **local-first**: el procesamiento, los embeddings, el índice vectorial y el uso principal deben poder ejecutarse en la máquina del usuario. Las APIs externas son adaptadores opcionales, no un requisito del MVP.

## Objetivo del MVP

El MVP termina cuando un usuario puede:

1. Ingerir uno o varios documentos educativos.
2. Indexar el contenido en Qdrant local.
3. Hacer una pregunta desde la CLI.
4. Recibir una respuesta del tutor basada únicamente en el material recuperado.
5. Ver las fuentes utilizadas: documento, página, sección y, cuando corresponda, figura o tabla.

## Alcance incluido

### Tipos de contenido

- Texto plano: `.txt`.
- Markdown: `.md` y `.markdown`.
- JSONL con registros textuales.
- PDF, especialmente libros y apuntes educativos.
- Figuras y diagramas contenidos en PDFs cuando Docling los asocia con un caption. La asociación por relación espacial sin caption queda para una fase posterior.
- Tablas de PDF como contenido textual estructurado cuando su extracción sea confiable.
- Un loader unificado con Docling como backend PDF principal.

### Capacidades

- Ingesta local desde archivos y directorios.
- Extracción de texto de PDF.
- Conservación de documento, página, capítulo/sección y tipo de contenido.
- Indexación vectorial en Qdrant.
- Recuperación semántica de fragmentos.
- Recuperación de texto asociado a figuras y tablas.
- Respuestas de tutor: explicación, resumen, comparación y preguntas de repaso.
- Ollama como proveedor principal.
- Adaptadores opcionales para OpenAI, Claude y Gemini.
- CLI como primera interfaz.

## Fuera del MVP

- Audio y vídeo.
- Transcripción de audio o vídeo.
- Aplicación móvil (la interfaz web está contemplada en el roadmap futuro).
- Sincronización entre máquinas.
- Multiusuario, autenticación y permisos.
- Despliegue en nube como requisito.
- Entrenamiento o fine-tuning de modelos.
- Búsqueda visual avanzada mediante CLIP.
- Generación automática de material didáctico complejo.

Audio y vídeo podrán añadirse en una fase posterior reutilizando el mismo contrato de documentos, fragmentos y metadatos.

## Principios del proyecto

1. **Local primero:** no depender de una API para la demo mínima.
2. **Minimalismo:** incorporar solo componentes necesarios para ingerir, recuperar y responder.
3. **Trazabilidad:** toda respuesta debe poder relacionarse con sus fuentes.
4. **Separación de responsabilidades:** carga, normalización, fragmentación, embeddings, almacenamiento, recuperación y tutor deben ser capas independientes.
5. **Degradación clara:** si una extracción o modelo opcional no está disponible, el sistema debe indicar el problema sin ocultar una ingesta incompleta.
6. **Evolución multimodal:** texto y figuras primero; audio y vídeo después.

## Criterio de finalización del MVP

El MVP se considera listo cuando el flujo completo funciona en una máquina local con:

- Python 3.11+.
- Ollama ejecutándose localmente.
- Qdrant ejecutándose localmente.
- Un PDF educativo real con al menos una figura.
- Una pregunta cuya respuesta dependa del texto o de una figura del PDF.
- Fuentes mostradas en la respuesta.
- Pruebas automatizadas para carga, chunking, recuperación y tutor.
