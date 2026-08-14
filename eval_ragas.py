import json
import logging
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from local_teacher.factory import obtener_modelos
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.query.retriever import ejecutar_consulta

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)

def main():
    print("[*] Iniciando Evaluación Estandarizada Ragas...")
    llm, emb = obtener_modelos("ollama")
    retriever = get_qdrant_retriever(emb)
    
    preguntas = []
    golden_answers = []
    respuestas_generadas = []
    contextos_recuperados = []
    
    print("[*] Generando respuestas para el dataset de prueba (3 preguntas)...")
    count = 0
    with open("test_eval_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if count >= 3: break
            item = json.loads(line)
            q = item["question"]
            ga = item["golden_answer"]
            
            print(f"\n- Q: {q}")
            try:
                res = ejecutar_consulta(retriever, llm, q, usar_critico=False)
                ans = res["answer"]
                docs = res["context_docs"]
                
                preguntas.append(q)
                golden_answers.append(ga)
                respuestas_generadas.append(ans)
                contextos_recuperados.append([d.page_content for d in docs])
                count += 1
            except Exception as e:
                print(f"Error procesando pregunta: {e}")
            
    data = {
        "question": preguntas,
        "answer": respuestas_generadas,
        "contexts": contextos_recuperados,
        "ground_truth": golden_answers
    }
    
    dataset = Dataset.from_dict(data)
    
    print("\n[*] Calculando métricas de Ragas (faithfulness, answer_relevancy, context_recall)...")
    print("    (Puede demorar si Ollama corre en CPU o GPU lenta)")
    
    metrics = [faithfulness, answer_relevancy, context_recall]
    
    try:
        # Suprimimos logs excesivos de httpx
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        resultado = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=emb
        )
        print("\n=== Resultados Ragas ===")
        print(resultado)
    except Exception as e:
        print(f"\n[!] Error al ejecutar la evaluación de Ragas: {e}")

if __name__ == "__main__":
    main()
