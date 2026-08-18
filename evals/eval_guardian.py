import argparse
import json
import logging
import os
import sys

# Permitir importaciones relativas desde la carpeta raíz o la carpeta evals
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from local_teacher.factory import obtener_modelos, obtener_llm_critico
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.query.retriever import ejecutar_consulta

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)

# Phrases that indicate a correct refusal (the tutor admitted it doesn't know)
REFUSAL_PHRASES = [
    "no está cubierto en el material",
    "no he encontrado información",
    "no poseo información",
    "require_web_search",
]

def is_correct_refusal(response: str) -> bool:
    """Check if the response is a correct refusal to answer."""
    response_lower = response.strip().lower()
    return any(phrase in response_lower for phrase in REFUSAL_PHRASES)

def evaluate_groundedness(guardian_llm, context: str, response: str) -> dict:
    # Use Granite Guardian's native hallucination detection format
    user_prompt = """You are a safety agent trying to help determine whether the assistant message is a hallucination based on the context provided by the user.

<context>
{context}
</context>

<assistant_message>
{response}
</assistant_message>

Is the assistant message a hallucination based on the context? Respond with only Yes or No."""

    prompt = ChatPromptTemplate.from_messages([
        ("user", user_prompt)
    ])
    
    chain = prompt | guardian_llm | StrOutputParser()
    result = chain.invoke({"context": context, "response": response})
    
    raw = result.strip().lower()
    
    # Guardian answers "Yes" = IS a hallucination, "No" = is NOT a hallucination
    # We invert the logic for our labels: grounded = Yes in our output
    if "no" in raw:
        label = "Grounded"
    elif "yes" in raw:
        label = "Hallucination"
    else:
        label = "Unknown"
    
    return {"raw": result.strip(), "label": label}

def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo supervisor")
    parser.add_argument("--limit", type=int, default=3, help="Límite de preguntas a evaluar")
    args = parser.parse_args()

    print("[*] Iniciando Evaluación de Groundedness con Granite Guardian...")
    llm, emb = obtener_modelos("ollama")
    retriever = get_qdrant_retriever(emb)
    
    guardian_llm = obtener_llm_critico("ollama")
    
    results_summary = {"Grounded": 0, "Hallucination": 0, "Correct Refusal": 0, "Unknown": 0}
    
    print(f"[*] Evaluando respuestas para el dataset de prueba ({args.limit} preguntas)...\n")
    count = 0
    with open("test_eval_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if count >= args.limit:
                break
            item = json.loads(line)
            q = item["question"]
            
            print(f"{'='*60}")
            print(f"Q: {q}")
            print(f"{'='*60}")
            try:
                res = ejecutar_consulta(retriever, llm, q, usar_critico=False)
                ans = res["answer"]
                docs = res["context_docs"]
                
                context_str = "\n\n".join([d.page_content for d in docs])
                
                print(f"\n  [Contexto] {len(docs)} docs, {len(context_str)} chars")
                if context_str:
                    print(f"  {context_str[:200]}...")
                
                print("\n  [Respuesta]")
                print(f"  {ans[:200]}...")
                
                # Check if this is a correct refusal first
                if is_correct_refusal(ans):
                    results_summary["Correct Refusal"] += 1
                    print("\n  -> Resultado: ✅ Correct Refusal (el tutor admitió que no tiene la info)")
                elif not context_str.strip():
                    # No context but tutor answered anyway = hallucination
                    results_summary["Hallucination"] += 1
                    print("\n  -> Resultado: ❌ Hallucination (respondió sin contexto disponible)")
                else:
                    print("\n  -> Evaluando fidelidad con Guardian...")
                    eval_result = evaluate_groundedness(guardian_llm, context_str, ans)
                    
                    label = eval_result["label"]
                    results_summary[label] += 1
                    
                    icon = {"Grounded": "✅", "Hallucination": "❌", "Unknown": "❓"}
                    print(f"  -> Resultado: {icon.get(label, '❓')} {label} (raw: {eval_result['raw'][:80]})")
                
                count += 1
            except Exception as e:
                print(f"Error procesando pregunta: {e}")

    print(f"\n{'='*60}")
    print("RESUMEN DE EVALUACIÓN")
    print(f"{'='*60}")
    total = sum(results_summary.values())
    for label, cnt in results_summary.items():
        if cnt > 0:
            pct = (cnt / total * 100) if total > 0 else 0
            icon = {"Grounded": "✅", "Hallucination": "❌", "Correct Refusal": "✅", "Unknown": "❓"}
            print(f"  {icon.get(label, '')} {label}: {cnt}/{total} ({pct:.0f}%)")

if __name__ == "__main__":
    main()
