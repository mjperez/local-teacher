# Roadmap técnico

## Fase 0 — Contrato y documentación

- [x] Definir propósito y alcance.
- [x] Definir requisitos funcionales y no funcionales.
- [ ] Definir el esquema común de metadata.
- [ ] Alinear README, arquitectura y comandos reales.

## Fase 1 — MVP textual local

- [x] Cargar TXT, Markdown, JSONL y PDF.
- [x] Fragmentar texto y conservar metadata básica.
- [x] Persistir vectores en Qdrant.
- [x] Consultar mediante CLI.
- [x] Hacer configurable el modelo de embeddings local.
- [x] Añadir tutor con prompt pedagógico.
- [ ] Mostrar fuentes en cada respuesta.
- [ ] Añadir pruebas unitarias del flujo completo.

## Fase 2 — PDFs educativos trazables

- [ ] Asociar cada fragmento con página y sección.
- [ ] Asociar figuras con documento, página, sección y caption.
- [ ] Mantener `captions.json` como manifest de auditoría.
- [x] Unificar extracción de texto, figuras y tablas con Docling.
- [x] Indexar tablas estructuradas cuando sean confiables.
- [ ] Añadir deduplicación e ingesta idempotente.
- [ ] Validar con varios libros reales.

## Fase 3 — Proveedores opcionales

- [ ] Configuración centralizada de modelos y proveedores.
- [ ] OpenAI mediante API key opcional.
- [ ] Anthropic/Claude mediante API key opcional.
- [ ] Google Gemini mediante API key opcional.
- [ ] Separar dependencias locales, opcionales y de desarrollo.

## Fase 4 — Multimodalidad visual

- [ ] Definir estrategia de embeddings visuales.
- [ ] Evaluar CLIP frente a otros modelos locales.
- [ ] Decidir si se usa una colección visual separada o búsqueda híbrida.
- [ ] Permitir consultas visuales o recuperación directa de imágenes.
- [ ] Mantener la relación imagen-documento-página-sección.

## Fase 5 — Audio y vídeo

- [ ] Transcripción local.
- [ ] Segmentación temporal.
- [ ] Metadata de minuto/segundo y fuente.
- [ ] Indexación de transcripciones.
- [ ] Recuperación multimodal unificada.

## Fase 6 — Experiencia de tutor

- [ ] Modos explicar, resumir, comparar y repasar.
- [ ] Preguntas de seguimiento.
- [ ] Evaluación de respuestas y quizzes.
- [ ] Historial local de sesiones.
- [ ] Interfaz web opcional (Streamlit/Gradio), priorizando un buen renderizado de Markdown y fórmulas LaTeX.
