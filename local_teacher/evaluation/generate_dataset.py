import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time

# Permite ejecutar con "python -m local_teacher.generate_dataset" o "python local_teacher/generate_dataset.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from local_teacher.factory import obtener_modelos
from local_teacher.storage.qdrant_store import get_qdrant_store

# Desactivar logs ruidosos
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def extract_json_from_text(text: str) -> dict:
    """Extrae de manera robusta un objeto JSON desde una respuesta del LLM."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generador Sintético de Dataset de Evaluación RAG")
    parser.add_argument("--dataset", required=True, help="Ruta al archivo .jsonl de destino")
    parser.add_argument("--provider", default=os.getenv("LOCAL_TEACHER_PROVIDER", "ollama"))
    parser.add_argument("--ollama-llm", default=os.getenv("OLLAMA_LLM", "deepseek-r1:8b"))
    parser.add_argument("--ollama-embed", default=os.getenv("OLLAMA_EMBED", "nomic-embed-text"))
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    
    args = parser.parse_args()
    
    print("[*] Iniciando Generador de Dataset Sintético...")
    print(f"[*] Usando modelo LLM: {args.ollama_llm}")
    
    # 1. Cargar hashes procesados
    processed_hashes = set()
    if os.path.exists(args.dataset):
        with open(args.dataset, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "chunk_hash" in data:
                        processed_hashes.add(data["chunk_hash"])
                except Exception:
                    pass
    
    print(f"[*] Se encontraron {len(processed_hashes)} fragmentos ya procesados en '{args.dataset}'.")
    
    # 2. Conectar a Qdrant
    print("[*] Conectando a Ollama y Qdrant...")
    llm, embeddings = obtener_modelos(
        args.provider,
        ollama_llm=args.ollama_llm,
        ollama_embed=args.ollama_embed,
        ollama_host=args.ollama_host,
    )
    vectorstore = get_qdrant_store(embeddings)
    client = vectorstore.client
    collection = vectorstore.collection_name
    
    # 3. Obtener todos los puntos (scroll)
    print("[*] Recuperando fragmentos documentales de la base de datos vectorial...")
    try:
        all_points = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if offset is None:
                break
    except Exception as e:
        print(f"[-] Error al conectar con Qdrant: {e}")
        return
        
    print(f"[*] Total de fragmentos almacenados: {len(all_points)}")
    
    # 4. Filtrar fragmentos nuevos usando su hash
    new_chunks = []
    for p in all_points:
        # Langchain QdrantVectorStore guarda el texto en payload.page_content por defecto
        content = p.payload.get("page_content", "")
        if not content:
            metadata = p.payload.get("metadata", {})
            content = metadata.get("page_content", "")
            
        if content:
            h = get_chunk_hash(content)
            if h not in processed_hashes:
                new_chunks.append((h, content))
                
    print(f"[*] Fragmentos nuevos pendientes por procesar: {len(new_chunks)}")
    if not new_chunks:
        print("\n[✔] No hay fragmentos nuevos. ¡El dataset dorado está completamente actualizado!")
        return
        
    # 5. Preparar Prompt Generator
    prompt_gen = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en evaluar sistemas RAG (Retrieval-Augmented Generation). "
                   "Tu tarea es leer el fragmento de texto proporcionado e inventar UNA pregunta difícil que un alumno podría hacer. "
                   "Luego debes generar la respuesta ideal basada en el texto.\n\n"
                   "DEBES devolver ÚNICAMENTE un objeto JSON válido con esta estructura, sin bloques de código extra ni introducciones:\n"
                   "{{\n"
                   '  "question": "Pregunta de un alumno que se puede responder con el texto",\n'
                   '  "golden_answer": "La respuesta ideal y detallada, basada puramente en el texto",\n'
                   '  "expected_keywords": ["palabra_importante1", "concepto_clave"],\n'
                   '  "forbidden_words": ["palabra_fuera_de_contexto", "alucinacion_relacionada"]\n'
                   "}}\n"
                   "Para `forbidden_words`, incluye palabras que un LLM malo podría alucinar o conceptos equivocados."),
        ("human", "Fragmento de texto:\n{context}")
    ])
    
    chain = prompt_gen | llm
    
    print("\n[*] Iniciando generación de preguntas con LLM...")
    # 6. Invocar LLM e ir guardando en streaming (append)
    with open(args.dataset, "a", encoding="utf-8") as f:
        for i, (chunk_hash, content) in enumerate(new_chunks, 1):
            print(f"\n--- Procesando fragmento {i}/{len(new_chunks)} ---")
            try:
                response = chain.invoke({"context": content})
                raw_text = response.content if hasattr(response, "content") else str(response)
                
                # Extraer JSON de deepseek (que a veces añade bloques <think>)
                clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
                data = extract_json_from_text(clean_text)
                
                if "question" in data and "golden_answer" in data:
                    # Añadir el hash para tracking
                    data["chunk_hash"] = chunk_hash
                    # Guardar en archivo
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[+] Pregunta generada: {data['question']}")
                    print(f"    Keywords esperadas: {data.get('expected_keywords', [])}")
                else:
                    print("[-] El modelo no devolvió un JSON con la estructura correcta. Saltando fragmento.")
                    print(f"Output crudo: {clean_text[:150]}...")
            except Exception as e:
                print(f"[-] Error procesando fragmento: {e}")
            
            # Pequeña pausa para no quemar el servidor Ollama local
            time.sleep(1)
            
    print("\n[✔] ¡Generación de dataset completada exitosamente!")

if __name__ == "__main__":
    main()
