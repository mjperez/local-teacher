import argparse
import sys

from dotenv import load_dotenv

from local_teacher.chunker import dividir_texto
from local_teacher.factory import obtener_modelos
from local_teacher.loader import cargar_archivos
from local_teacher.retriever import ejecutar_query
from local_teacher.storage.qdrant_store import get_qdrant_store


def main() -> None:
    """Flujo mínimo de RAG: ingestar, indexar y responder."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI simple de RAG")
    parser.add_argument("--ingest", help="Directorio con archivos .txt, .md, .jsonl o .pdf", type=str)
    parser.add_argument(
        "--figuras",
        help="Exportar los diagramas/imágenes de los PDFs como PNG (en test_docs/figuras)",
        action="store_true",
    )
    parser.add_argument("--query", help="Pregunta para el RAG", type=str)
    parser.add_argument("--provider", help="Proveedor de modelos", type=str, default="openai", choices=["openai", "ollama"])
    parser.add_argument("--ollama-llm", help="Modelo LLM para Ollama", type=str, default="llama3")
    parser.add_argument("--ollama-embed", help="Modelo de embeddings para Ollama", type=str, default="nomic-embed-text")
    args = parser.parse_args()

    if not args.ingest and not args.query:
        print("[-] Error: define --ingest y/o --query")
        sys.exit(1)

    try:
        llm, embeddings = obtener_modelos(
            args.provider,
            ollama_llm=args.ollama_llm,
            ollama_embed=args.ollama_embed,
        )
    except (ValueError, SystemError) as exc:
        print(exc)
        sys.exit(1)

    vectorstore = None

    if args.ingest:
        documentos = cargar_archivos(args.ingest, extraer_figuras=args.figuras)
        if documentos:
            chunks = dividir_texto(documentos)
            vectorstore = get_qdrant_store(embeddings=embeddings, documentos=chunks)
            print(f"[+] Ingesta completada ({len(chunks)} fragmentos).")

    if args.query:
        if vectorstore is None:
            vectorstore = get_qdrant_store(embeddings=embeddings)

        resultado = ejecutar_query(vectorstore, llm, args.query)
        print("\n--- RESPUESTA ---")
        print(resultado["answer"])


if __name__ == "__main__":
    main()