import os
import sys

from local_teacher.factory import obtener_modelos
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.query import retriever as ret

llm, emb = obtener_modelos("ollama")
retriever = get_qdrant_retriever(emb)

# We will just patch the critic to print what it receives and what it outputs
original_eval = ret._evaluar_borrador

def patched_eval(llm, ctx, draft):
    print("\n\n>>> CRITIC IS EVALUATING THIS DRAFT:")
    print(draft)
    print("\n>>> CONTEXT PASSED TO CRITIC:")
    print(ctx[:500] + "... (truncated)")
    res = original_eval(llm, ctx, draft)
    print("\n>>> CRITIC DECISION:", res)
    return res

ret._evaluar_borrador = patched_eval

res = ret.ejecutar_consulta(retriever, llm, "¿Qué es un actuador y un sensor?", busqueda_web_alternativa=False)
print("\n\nFINAL RESULT:")
print(res)
