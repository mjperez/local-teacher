import argparse
import sys
from dotenv import load_dotenv

from chunker import dividir_texto
from factory import obtener_modelos
from loader import cargar_archivos
from retriever import ejecutar_query
from storage.qdrant_store import get_qdrant_store


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="CLI Modular RAG (Agnóstico a LLM)")
    parser.add_argument("--ingest", help="Directorio de archivos TXT", type=str)
    parser.add_argument("--query", help="Pregunta al RAG", type=str)
    # Nuevo argumento para seleccionar proveedor
    parser.add_argument("--provider", help="Proveedor de modelos (openai o ollama)", type=str, default="openai", choices=["openai", "ollama"])
    parser.add_argument("--ollama-llm", help="Modelo LLM para Ollama", type=str, default="llama3")
    parser.add_argument("--ollama-embed", help="Modelo de embeddings para Ollama", type=str, default="nomic-embed-text")
    args = parser.parse_args()

    if not args.ingest and not args.query:
        print("[-] Error: Define --ingest y/o --query")
        sys.exit(1)

    # 1. Iniciar fábrica de modelos
    try:
        llm, embeddings = obtener_modelos(
            args.provider, 
            ollama_llm=args.ollama_llm, 
            ollama_embed=args.ollama_embed
        )
        print(f"[*] Proveedor configurado: {args.provider.upper()}")
    except ValueError as e:
        print(e)
        sys.exit(1)

    vectorstore = None

    # 2. Ingesta
    if args.ingest:
        docs = cargar_archivos(args.ingest)
        if docs:
            chunks = dividir_texto(docs)
            vectorstore = get_qdrant_store(embeddings=embeddings, documentos=chunks)
            print(f"[+] Ingesta completada ({len(chunks)} fragmentos).")

    # 3. Consulta
    if args.query:
        if not vectorstore:
            vectorstore = get_qdrant_store(embeddings=embeddings)

        resultado = ejecutar_query(vectorstore, llm, args.query)
        
        print("\n--- RESPUESTA ---")
        print(resultado["answer"])

if __name__ == "__main__":
    main()