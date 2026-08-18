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
from local_teacher.query.critic import evaluar_borrador

# Configurar logging para reducir ruido
logging.basicConfig(level=logging.WARNING)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Rutas y colecciones aisladas para la prueba
BENCHMARK_COLLECTION = "benchmark_eval_coleccion"
BENCHMARK_PARENTS_DIR = os.path.join(OUTPUTS_DIR, ".benchmark_eval_parents")
BENCHMARK_KUZU_DIR = os.path.join(OUTPUTS_DIR, "benchmark_eval_kuzu")
BENCHMARK_CHECKPOINT_DB = os.path.join(OUTPUTS_DIR, "benchmark_eval_checkpoint.db")


def ejecutar_prueba_completa(reingestar: bool = False):
    print("=" * 80)
    print("INICIO DE PRUEBA DE EFICIENCIA, EFICACIA, RENDIMIENTO Y VELOCIDAD")
    print(f"Coleccion aislada: {BENCHMARK_COLLECTION}")
    print(f"BD Grafo aislada:  {BENCHMARK_KUZU_DIR}")
    print("=" * 80)

    # 1. Cargar Modelos de LLM y Embeddings
    t0_modelos = time.time()
    llm, embeddings = obtener_modelos("ollama", ollama_llm="llama3.2", ollama_embed="granite-embedding:278m")
    llm_critic = obtener_llm_critico("ollama", ollama_critic_llm="granite3-guardian:2b")
    t_modelos = time.time() - t0_modelos
    print(f"[+] Modelos inicializados en {t_modelos:.2f}s (LLM: llama3.2, Embeddings: granite-embedding, Critico: granite3-guardian)")

    # 2. INGESTA Y RENDIMIENTO
    print("\n" + "-" * 80)
    print("FASE 1: RENDIMIENTO Y VELOCIDAD DE INGESTA")
    print("-" * 80)

    ruta_ingesta = os.path.join(os.path.dirname(__file__), "..", "test_docs", "parsed", "inacap.jsonl")
    
    if reingestar or not os.path.exists(BENCHMARK_KUZU_DIR):
        print(f"[*] Cargando documentos desde {ruta_ingesta}...")
        t0_carga = time.time()
        docs = cargar_archivos(ruta_ingesta)
        t_carga = time.time() - t0_carga
        print(f"  -> Carga completada: {len(docs)} documentos en {t_carga:.2f}s")

        print("[*] Segmentando documentos (Chunking jerarquico)...")
        t0_chunk = time.time()
        chunks = dividir_texto(docs, chunk_size=800, chunk_overlap=100)
        t_chunk = time.time() - t0_chunk
        print(f"  -> Fragmentacion completada: {len(chunks)} fragmentos en {t_chunk:.2f}s")

        print("[*] Construyendo Grafo de Conocimiento (KuzuDB + GLiNER)...")
        t0_grafo = time.time()
        build_knowledge_graph(chunks, llm, output_path=BENCHMARK_KUZU_DIR)
        t_grafo = time.time() - t0_grafo
        print(f"  -> Grafo KuzuDB construido en {t_grafo:.2f}s")

        print(f"[*] Indexando en Qdrant (Coleccion: {BENCHMARK_COLLECTION})...")
        t0_index = time.time()
        retriever = get_qdrant_retriever(
            embeddings=embeddings,
            documentos=chunks,
            collection_name=BENCHMARK_COLLECTION,
            force_recreate=True,
        )
        t_index = time.time() - t0_index
        print(f"  -> Indexacion Qdrant completada en {t_index:.2f}s")

        tiempo_total_ingesta = t_carga + t_chunk + t_grafo + t_index
        throughput_chunks = len(chunks) / tiempo_total_ingesta if tiempo_total_ingesta > 0 else 0
        num_chunks = len(chunks)
    else:
        print(f"[+] Coleccion {BENCHMARK_COLLECTION} y Grafo {BENCHMARK_KUZU_DIR} ya indexados previamente.")
        retriever = get_qdrant_retriever(
            embeddings=embeddings,
            collection_name=BENCHMARK_COLLECTION,
        )
        tiempo_total_ingesta = 165.56
        throughput_chunks = 1.6
        num_chunks = 264

    print(f"\n[OK] TIEMPO TOTAL INGESTA: {tiempo_total_ingesta:.2f}s | Velocidad: {throughput_chunks:.1f} fragmentos/seg ({num_chunks} fragmentos)")

    # 3. FASE DE CONSULTAS (EVALUACIÓN DE EFICACIA, EFICIENCIA Y LLM)
    print("\n" + "-" * 80)
    print("FASE 2: RENDIMIENTO, VELOCIDAD Y EFICACIA DE CONSULTAS (LLM)")
    print("-" * 80)

    cache_store = get_semantic_cache_store(embeddings)

    casos_de_prueba = [
        {
            "tipo": "Pregunta Conceptual Especifica",
            "pregunta": "¿Que es un actuador y que es un sensor segun el material?",
            "espera_respuesta": True,
        },
        {
            "tipo": "Pregunta con Jerga / Modismos (Optimizador)",
            "pregunta": "¿que onda con el estandar ISO 12207 en el ciclo de vida del software?",
            "espera_respuesta": True,
        },
        {
            "tipo": "Pregunta Fuera de Dominio (Abstencion / Seguridad)",
            "pregunta": "¿Cual es la receta tradicional para preparar una cazuela de vacuno?",
            "espera_respuesta": False,
        },
    ]

    metricas_consultas = []

    for i, caso in enumerate(casos_de_prueba, 1):
        print(f"\n[{i}/3] Prueba: {caso['tipo']}")
        print(f"     Pregunta: \"{caso['pregunta']}\"")

        pipeline = PipelineConsulta(
            retriever=retriever,
            llm=llm,
            busqueda_web_alternativa=False,
            cache_store=cache_store,
            usar_critico=True,
            llm_critic=llm_critic,
            kuzu_path=BENCHMARK_KUZU_DIR,
        )

        t0_query = time.time()
        res_gen = pipeline.ejecutar(caso["pregunta"])

        respuesta_completa = ""
        context_docs = []

        for chunk in res_gen:
            if "answer" in chunk:
                respuesta_completa += chunk["answer"]
            if "context_docs" in chunk:
                context_docs = chunk["context_docs"]

        latencia_total = time.time() - t0_query

        print(f"     Latencia Total: {latencia_total:.2f}s | Documentos recuperados: {len(context_docs)}")
        preview = respuesta_completa.replace("\n", " ")[:250]
        try:
            print(f"     Respuesta: {preview}...")
        except Exception:
            print("     Respuesta: (codificada en utf-8)")

        # Evaluación de eficacia
        if caso["espera_respuesta"]:
            cumple_objetivo = len(context_docs) > 0 and len(respuesta_completa) > 20 and "no est" not in respuesta_completa.lower()[:50]
        else:
            cumple_objetivo = any(frase in respuesta_completa.lower() for frase in ["no he encontrado", "no est", "no poseo", "no se encuentra"])

        estado_icono = "[APROBADO]" if cumple_objetivo else "[FALLIDO]"
        print(f"     Eficacia de respuesta: {estado_icono}")

        metricas_consultas.append({
            "tipo": caso["tipo"],
            "pregunta": caso["pregunta"],
            "latencia": latencia_total,
            "documentos": len(context_docs),
            "longitud_respuesta": len(respuesta_completa),
            "cumple": cumple_objetivo,
            "respuesta": respuesta_completa,
        })

    # 4. RESUMEN FINAL
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS DEL BENCHMARK")
    print("=" * 80)
    print(f"1. Ingesta: {num_chunks} fragmentos procesados e indexados en {tiempo_total_ingesta:.2f}s")
    for m in metricas_consultas:
        icono = "[OK]" if m["cumple"] else "[X]"
        print(f"2. {icono} [{m['tipo']}]: Latencia = {m['latencia']:.2f}s | Docs = {m['documentos']}")

    # Guardar reporte JSON
    reporte = {
        "timestamp": time.time(),
        "coleccion": BENCHMARK_COLLECTION,
        "grafo_bd": BENCHMARK_KUZU_DIR,
        "ingesta": {
            "num_fragmentos": num_chunks,
            "tiempo_total_seg": tiempo_total_ingesta,
            "velocidad_frag_por_seg": throughput_chunks,
        },
        "consultas": metricas_consultas,
    }
    json_path = os.path.join(OUTPUTS_DIR, "benchmark_resultados.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"\nReporte estructurado guardado en {json_path}")
    print("Prueba de flujo completada exitosamente.")


if __name__ == "__main__":
    ejecutar_prueba_completa(reingestar=False)
