import time
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.query.pipeline import PipelineConsulta
from local_teacher.query.optimizer import reescribir_consulta, ConsultaEstructurada
from local_teacher.query.prompts import crear_cadena_tutor, formatear_documentos

BENCHMARK_COLLECTION = "benchmark_eval_coleccion"
BENCHMARK_KUZU_DIR = "./benchmark_eval_kuzu"

def test_diagnostico():
    print("=" * 80)
    print("DIAGNOSTICO Y EXPERIMENTO DE OPTIMIZACION DEL MODO EXACT")
    print("=" * 80)

    # 1. Modelos con keep_alive=300 (segundos) y num_ctx equilibrado
    llm_fast = ChatOllama(model="llama3.2", temperature=0, base_url="http://127.0.0.1:11434", num_ctx=4096, keep_alive=300)
    llm_deepseek = ChatOllama(model="deepseek-r1:8b", temperature=0, base_url="http://127.0.0.1:11434", num_ctx=4096, keep_alive=300)
    embeddings = OllamaEmbeddings(model="granite-embedding:278m", base_url="http://127.0.0.1:11434", keep_alive=300)
    llm_critic = ChatOllama(model="granite3-guardian:2b", temperature=0, base_url="http://127.0.0.1:11434", num_ctx=4096, keep_alive=300)

    consulta_prueba = "¿que onda con el estandar ISO 12207 en el ciclo de vida del software?"

    print("\n--- 1. Prueba de Reescritura / Optimizacion de Consulta ---")
    t0 = time.time()
    res_llama = reescribir_consulta(llm_fast, consulta_prueba)
    t_llama = time.time() - t0
    print(f"[+] Llama 3.2 reescritura en {t_llama:.2f}s:")
    print(f"    Consulta optimizada: {res_llama.consulta}")
    print(f"    Entidades: {res_llama.entidades}")

    # 2. Pipeline con DeepSeek-R1 (8B) para tutoría + Llama 3.2 para optimizador
    print("\n--- 2. Ejecucion con Pipeline Hibrido (Llama 3.2 Optimizador + DeepSeek-R1 Tutor + Granite Critic) ---")
    retriever = get_qdrant_retriever(embeddings=embeddings, collection_name=BENCHMARK_COLLECTION)

    pipeline = PipelineConsulta(
        retriever=retriever,
        llm=llm_deepseek,
        busqueda_web_alternativa=False,
        usar_critico=True,
        llm_critic=llm_critic,
        kuzu_path=BENCHMARK_KUZU_DIR,
    )

    t0_q = time.time()
    res_gen = pipeline.ejecutar(consulta_prueba)
    resp = ""
    docs = []
    for c in res_gen:
        if "answer" in c:
            resp += c["answer"]
        if "context_docs" in c:
            docs = c["context_docs"]
    lat_total = time.time() - t0_q

    print(f"\n[OK] LATENCIA TOTAL MODO EXACT CON KEEP_ALIVE Y OPTIMIZADOR: {lat_total:.2f}s (vs 101.18s inicial)")
    print(f"     Documentos recuperados: {len(docs)}")
    preview = resp.replace("\n", " ")[:250]
    print(f"     Respuesta:\n     {preview}...")

if __name__ == "__main__":
    test_diagnostico()
