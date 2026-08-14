import argparse
import json
import logging
import os
import re
import sys
import time

# Permite ejecutar con "python -m local_teacher.eval" o "python local_teacher/eval.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from local_teacher.factory import obtener_modelos
from local_teacher.query.retriever import ejecutar_consulta
from local_teacher.storage.qdrant_store import get_qdrant_store

# Desactivar logs ruidosos para la evaluación
logging.getLogger("httpx").setLevel(logging.WARNING)


def compute_cosine_similarity(vec1, vec2):
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Suite de Evaluación RAG (3 Niveles)")
    parser.add_argument(
        "--dataset", required=True, help="Ruta al archivo .jsonl del dataset dorado"
    )
    parser.add_argument(
        "--provider", default=os.getenv("LOCAL_TEACHER_PROVIDER", "ollama")
    )
    parser.add_argument("--ollama-llm", default=os.getenv("OLLAMA_LLM", "deepseek-r1:8b"))
    parser.add_argument(
        "--ollama-embed", default=os.getenv("OLLAMA_EMBED", "nomic-embed-text")
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )

    args = parser.parse_args()

    print("[*] Iniciando Evaluación RAG...")
    llm, embeddings = obtener_modelos(
        args.provider,
        ollama_llm=args.ollama_llm,
        ollama_embed=args.ollama_embed,
        ollama_host=args.ollama_host,
    )

    vectorstore = get_qdrant_store(embeddings)

    with open(args.dataset, "r", encoding="utf-8") as f:
        lineas = [l.strip() for l in f if l.strip()]

    total_q = len(lineas)
    stats = {"l1_pass": 0, "l2_pass": 0, "l3_pass": 0, "total": total_q}

    for i, linea in enumerate(lineas, 1):
        data = json.loads(linea)
        q = data["question"]
        golden = data["golden_answer"]
        expected_kws = data.get("expected_keywords", [])
        forbidden = data.get("forbidden_words", [])

        print(f"\n--- Pregunta {i}/{total_q}: {q} ---")

        res = ejecutar_consulta(vectorstore, llm, q, transmitir=False)

        # Eliminar barras de progreso ruidosas que dejó ejecutar_consulta
        print("\n")

        answer = res["answer"]
        docs = res.get("context_docs", [])

        print(f"Respuesta generada:\n> {answer[:200]}...")

        # Nivel 1: Código / Reglas Duras
        l1_pass = True
        for fb in forbidden:
            if re.search(r"\b" + re.escape(fb.lower()) + r"\b", answer.lower()):
                l1_pass = False
                print(f"   [Nivel 1] FALLÓ: Contiene palabra prohibida '{fb}'")
                break

        if l1_pass:
            stats["l1_pass"] += 1
            print("   [Nivel 1] PASÓ (Reglas estructurales)")

        # Nivel 2: Heurística (Keywords en respuesta)
        l2_pass = False
        if not expected_kws:
            l2_pass = True
        else:
            kws_found = sum(1 for kw in expected_kws if kw.lower() in answer.lower())
            # Requisito: al menos el 50% de las expected keywords
            if kws_found / len(expected_kws) >= 0.5:
                l2_pass = True
            else:
                print(
                    f"   [Nivel 2] FALLÓ: Se esperaban {len(expected_kws)} keywords, se encontraron {kws_found}."
                )

        if l2_pass:
            stats["l2_pass"] += 1
            print("   [Nivel 2] PASÓ (Heurística de keywords)")

        # Nivel 3: Semántica (Similitud Coseno > 0.85)
        ans_vec = embeddings.embed_query(answer)
        gold_vec = embeddings.embed_query(golden)

        sim = compute_cosine_similarity(ans_vec, gold_vec)
        if sim >= 0.85:
            stats["l3_pass"] += 1
            print(f"   [Nivel 3] PASÓ (Similitud: {sim:.3f} >= 0.85)")
        else:
            print(f"   [Nivel 3] FALLÓ (Similitud: {sim:.3f} < 0.85)")

    print("\n" + "=" * 50)
    print("REPORTE DE EVALUACIÓN FINAL")
    print("=" * 50)
    print(f"Total evaluadas: {total_q}")
    print(
        f"Nivel 1 (Reglas/Estructura)  : {stats['l1_pass']}/{total_q} ({(stats['l1_pass']/total_q)*100:.1f}%)"
    )
    print(
        f"Nivel 2 (Heurística/Keywords): {stats['l2_pass']}/{total_q} ({(stats['l2_pass']/total_q)*100:.1f}%)"
    )
    print(
        f"Nivel 3 (Semántica/Coseno)   : {stats['l3_pass']}/{total_q} ({(stats['l3_pass']/total_q)*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
