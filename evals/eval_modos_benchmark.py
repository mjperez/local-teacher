import os
import sys
import shutil
import time
import json
import logging
from pathlib import Path

# Permitir importaciones relativas desde la carpeta raíz o la carpeta evals
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_teacher.factory import obtener_modelos, obtener_llm_critico
from local_teacher.ingestion.state_manager import StateManager
from local_teacher.ingestion.loader import cargar_archivos
from local_teacher.ingestion.chunker import dividir_texto
from local_teacher.ingestion.graph_builder import build_knowledge_graph
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.storage.redis_cache import get_semantic_cache_store
from local_teacher.query.pipeline import PipelineConsulta

# Configurar logging para reducir ruido
logging.basicConfig(level=logging.WARNING)

BENCHMARK_COLLECTION = "benchmark_eval_coleccion"
BENCHMARK_KUZU_DIR = "./benchmark_eval_kuzu"


def benchmark_modos():
    print("=" * 80)
    print("BENCHMARK COMPARATIVO DE MODOS: ULTRA-FAST vs FAST vs EXACT")
    print(f"Coleccion: {BENCHMARK_COLLECTION} | Grafo: {BENCHMARK_KUZU_DIR}")
    print("=" * 80)

    # Inicializar embeddings compartidos
    _, embeddings = obtener_modelos("ollama", ollama_llm="llama3.2", ollama_embed="granite-embedding:278m")
    llm_critic = obtener_llm_critico("ollama", ollama_critic_llm="granite3-guardian:2b")
    retriever = get_qdrant_retriever(embeddings=embeddings, collection_name=BENCHMARK_COLLECTION)

    # Cargar los LLMs para cada modo
    print("[*] Inicializando modelos de lenguaje...")
    llm_llama, _ = obtener_modelos("ollama", ollama_llm="llama3.2", ollama_embed="granite-embedding:278m")
    llm_deepseek, _ = obtener_modelos("ollama", ollama_llm="deepseek-r1:8b", ollama_embed="granite-embedding:278m")
    print("[+] Modelos listos: llama3.2 y deepseek-r1:8b")

    modos_config = [
        {
            "nombre": "ultra-fast",
            "descripcion": "Llama 3.2 (3B) sin supervisor critico",
            "llm": llm_llama,
            "usar_critico": False,
            "llm_critic": None,
        },
        {
            "nombre": "fast",
            "descripcion": "Llama 3.2 (3B) con supervisor Granite Guardian (2B)",
            "llm": llm_llama,
            "usar_critico": True,
            "llm_critic": llm_critic,
        },
        {
            "nombre": "exact",
            "descripcion": "DeepSeek-R1 (8B) con supervisor Granite Guardian (2B)",
            "llm": llm_deepseek,
            "usar_critico": True,
            "llm_critic": llm_critic,
        },
    ]

    preguntas_test = [
        {
            "id": "Q1_Conceptual",
            "pregunta": "¿Que es un actuador y que es un sensor segun el material de IoT?",
            "espera_contenido": True,
        },
        {
            "id": "Q2_Modismo_ISO",
            "pregunta": "¿que onda con el estandar ISO 12207 en el ciclo de vida del software?",
            "espera_contenido": True,
        },
        {
            "id": "Q3_Fuera_Dominio",
            "pregunta": "¿Cual es la receta para preparar una cazuela de vacuno?",
            "espera_contenido": False,
        },
    ]

    resultados_comparativos = []

    for modo in modos_config:
        print("\n" + "=" * 80)
        print(f"EVALUANDO MODO: [{modo['nombre'].upper()}] - {modo['descripcion']}")
        print("=" * 80)

        for p in preguntas_test:
            print(f"\n---> Consulta [{p['id']}]: \"{p['pregunta']}\"")

            pipeline = PipelineConsulta(
                retriever=retriever,
                llm=modo["llm"],
                busqueda_web_alternativa=False,
                usar_critico=modo["usar_critico"],
                llm_critic=modo["llm_critic"],
                kuzu_path=BENCHMARK_KUZU_DIR,
            )

            t0 = time.time()
            res_gen = pipeline.ejecutar(p["pregunta"])

            respuesta_acumulada = ""
            docs_recuperados = []

            for chunk in res_gen:
                if "answer" in chunk:
                    respuesta_acumulada += chunk["answer"]
                if "context_docs" in chunk:
                    docs_recuperados = chunk["context_docs"]

            latencia = time.time() - t0

            # Evaluación de calidad
            if p["espera_contenido"]:
                aprobado = len(docs_recuperados) > 0 and len(respuesta_acumulada) > 30 and "no est" not in respuesta_acumulada.lower()[:50]
            else:
                aprobado = any(frase in respuesta_acumulada.lower() for frase in ["no est", "no he encontrado", "no poseo", "no se encuentra"])

            estado = "[APROBADO]" if aprobado else "[FALLIDO]"
            print(f"     Latencia: {latencia:.2f}s | Docs: {len(docs_recuperados)} | Estado: {estado}")
            preview = respuesta_acumulada.replace("\n", " ")[:200]
            print(f"     Resumen: {preview}...")

            resultados_comparativos.append({
                "modo": modo["nombre"],
                "pregunta_id": p["id"],
                "pregunta": p["pregunta"],
                "latencia_seg": round(latencia, 2),
                "docs_count": len(docs_recuperados),
                "longitud_chars": len(respuesta_acumulada),
                "aprobado": aprobado,
                "respuesta_texto": respuesta_acumulada,
            })

    # Imprimir Tabla Resumen
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA DE RENDIMIENTO POR MODO")
    print("=" * 80)
    print(f"{'Modo':<12} | {'Pregunta':<16} | {'Latencia (s)':<12} | {'Docs':<6} | {'Estado':<10}")
    print("-" * 65)
    for r in resultados_comparativos:
        print(f"{r['modo']:<12} | {r['pregunta_id']:<16} | {r['latencia_seg']:<12.2f} | {r['docs_count']:<6} | {'OK' if r['aprobado'] else 'FALLO':<10}")

    with open("benchmark_modos_comparativo.json", "w", encoding="utf-8") as f:
        json.dump(resultados_comparativos, f, ensure_ascii=False, indent=2)
    print("\nResultados completos guardados en benchmark_modos_comparativo.json")


if __name__ == "__main__":
    benchmark_modos()
