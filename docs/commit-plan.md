# Commit plan para el MVP RAG

## Commit 1: estructura base del proyecto  [completado]

- Crear el paquete `local_teacher/`.  [hecho]
- Dejar `config.py`, `schemas.py`, `logging_utils.py` y `hardware.py` como soporte común.  [hecho]
- Crear `docs/rag-structure.md` y `docs/24h-checklist.md`.  [hecho]
- Objetivo: tener la arquitectura clara antes de programar lógica.  [hecho]

## Commit 2: carga de archivos

- Implementar `loader.py` para leer `.txt` y `.md`.
- Validar archivos vacíos o no soportados.
- Objetivo: convertir archivos del usuario en texto usable.

## Commit 3: fragmentación de texto

- Implementar `chunker.py` con tamaño y solapamiento.
- Conservar metadata de origen por chunk.
- Objetivo: preparar el texto para recuperación semántica.

## Commit 4: persistencia vectorial

- Implementar `storage/qdrant_store.py`.
- Conectar con Qdrant en Docker.
- Guardar y consultar chunks.
- Objetivo: tener índice persistente y búsquedas básicas.

## Commit 5: recuperación de contexto

- Implementar `retriever.py`.
- Hacer que devuelva los chunks más relevantes para una pregunta.
- Objetivo: separar búsqueda de generación.

## Commit 6: proveedor local

- Implementar `providers/base.py` y `providers/local.py`.
- Resolver embeddings y chat local.
- Objetivo: poder correr el MVP sin API externa.

## Commit 7: tutor pedagógico

- Implementar `prompts.py` y `tutor.py`.
- Responder como profesor: explicar, resumir, preguntar de vuelta.
- Objetivo: transformar contexto recuperado en aprendizaje útil.

## Commit 8: CLI mínima

- Implementar `cli.py`.
- Comandos para indexar y consultar.
- Objetivo: probar el flujo end-to-end sin UI.

## Commit 9: modo API externo

- Implementar `providers/openai.py`.
- Conectar selección local/API desde `config.py`.
- Objetivo: cambiar de backend sin reescribir la app.

## Commit 10: evaluación y cierre

- Implementar `evaluation.py`.
- Agregar quiz, examen y feedback.
- Documentar uso básico.
- Objetivo: cerrar la demo del MVP.

## Regla práctica

- Si un commit toca ingestión, no mezclarlo con prompts o API.
- Si un commit toca storage, no mezclarlo con evaluación.
- Si dudas, haz commits más pequeños, no más grandes.
