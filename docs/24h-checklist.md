# Checklist 24h para el MVP RAG

## 1. 30 min: fija el alcance

- Solo `txt` y `md`.
- Solo local primero.
- Solo CLI para la primera demo.

## 2. 1 h: define la ruta de datos

- [loader.py](../local_teacher/loader.py)
- [chunker.py](../local_teacher/chunker.py)
- [storage/qdrant_store.py](../local_teacher/storage/qdrant_store.py)
- [retriever.py](../local_teacher/retriever.py)

## 3. 2 h: conecta Qdrant

- Confirma que Docker expone `localhost:6333`.
- Haz que el storage guarde y lea chunks.
- Verifica que puedas recuperar 3 a 5 resultados por consulta.

## 4. 2 h: implementa ingestión mínima

- Lee un archivo.
- Divide en chunks.
- Guarda en Qdrant.
- Repite con varios archivos.

## 5. 2 h: conecta embeddings

- Si local te complica, usa API temporalmente para avanzar.
- Si local funciona, mejor.
- No intentes optimizar todavía.

## 6. 2 h: arma el tutor

- [tutor.py](../local_teacher/tutor.py)
- Dale el contexto recuperado.
- Haz que responda como profesor: explicar, resumir, preguntar de vuelta.

## 7. 2 h: crea la CLI mínima

- Comando para indexar.
- Comando para preguntar.
- Comando para limpiar o reiniciar índice si lo necesitas.

## 8. 2 h: agrega modo local/API

- [providers/local.py](../local_teacher/providers/local.py)
- [providers/openai.py](../local_teacher/providers/openai.py)
- [config.py](../local_teacher/config.py)

## 9. 2 h: endurece

- Manejo de errores.
- Mensajes claros.
- Validación de archivos vacíos o no soportados.

## 10. 2 h: prueba end-to-end

- Indexa un archivo real.
- Haz una pregunta real.
- Revisa si la respuesta usa contexto correcto.
- Ajusta prompt y chunking si hace falta.

## 11. Últimas 3 h: pulido y demo

- [evaluation.py](../local_teacher/evaluation.py) si te da tiempo.
- Documenta cómo correrlo.
- Prepara una demo corta.

## Regla de oro

- Si algo no aporta a "indexar, preguntar, responder", se pospone.
- Si te trabas más de 30 minutos en un punto, cambia a API externa o simplifica el componente.
