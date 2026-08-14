import argparse
import logging
import os
import sys

# Permite ejecutar con "python local_teacher/cli.py" directamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from local_teacher.ingestion.chunker import dividir_texto
from local_teacher.factory import obtener_modelos
from local_teacher.ingestion.loader import cargar_archivos, guardar_cache_jsonl
from local_teacher.query.retriever import ejecutar_consulta
from local_teacher.storage.qdrant_store import get_qdrant_store
from local_teacher.storage.redis_cache import get_semantic_cache_store
from local_teacher.ingestion.graph_builder import build_knowledge_graph

logging.basicConfig(
    filename="local_teacher.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    """Flujo mínimo de RAG: ingestar, indexar y responder."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI simple de local-teacher")
    parser.add_argument("--ingest", help="Archivo o directorio a ingerir")
    parser.add_argument("--query", help="Pregunta para el tutor")
    parser.add_argument("--figuras", action="store_true", help="Extraer imágenes PNG")
    parser.add_argument("--tablas", action="store_true", help="Extraer tablas a CSV/MD")
    parser.add_argument(
        "--no-formulas", action="store_true", help="Desactivar VLM de Docling"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recrear colección Qdrant"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Extraer y construir Grafo de Conocimiento (GraphRAG)",
    )
    parser.add_argument(
        "--web-fallback",
        action="store_true",
        help="Permitir consultar la web si la respuesta no está en los documentos locales",
    )

    # Modelos
    parser.add_argument(
        "--provider", default=os.getenv("LOCAL_TEACHER_PROVIDER", "ollama")
    )
    parser.add_argument("--ollama-llm", default=os.getenv("OLLAMA_LLM", "llama3.2"))
    parser.add_argument(
        "--ollama-embed", default=os.getenv("OLLAMA_EMBED", "nomic-embed-text")
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )

    args = parser.parse_args()

    if not args.ingest and not args.query:
        print("[-] Error: define --ingest y/o --query")
        sys.exit(1)

    llm, embeddings = obtener_modelos(
        args.provider,
        ollama_llm=args.ollama_llm,
        ollama_embed=args.ollama_embed,
        ollama_host=args.ollama_host,
    )

    vectorstore = None

    if args.ingest:
        print(f"[*] Iniciando carga de documentos desde: {args.ingest}")
        print(
            "[*] (Si es la primera vez que se procesa un PDF, Docling podría descargar modelos y tardar varios minutos...)"
        )
        docs = cargar_archivos(
            args.ingest,
            extraer_figuras=args.figuras,
            extraer_tablas=args.tablas,
            enriquecer_formulas=not args.no_formulas,
        )
        if docs:
            guardar_cache_jsonl(docs, args.ingest)

            print("[*] Dividiendo texto en fragmentos (chunking)...")
            chunks = dividir_texto(docs)

            if args.graph:
                print(
                    "[*] Construyendo Grafo de Conocimiento (esto puede tomar mucho tiempo)..."
                )
                build_knowledge_graph(chunks, llm)

            print(
                "[*] Generando embeddings y guardando en Qdrant (esto puede tomar un tiempo)..."
            )
            vectorstore = get_qdrant_store(
                embeddings, chunks, force_recreate=args.recreate
            )
            print(f"[+] Ingesta completada ({len(chunks)} fragmentos).")

    if args.query:
        print("[*] Conectando a Qdrant...")
        vectorstore = vectorstore or get_qdrant_store(embeddings)
        cache_store = get_semantic_cache_store()
        print("[*] Ejecutando búsqueda y generación...")
        res = ejecutar_consulta(
            vectorstore,
            llm,
            args.query,
            transmitir=True,
            busqueda_web_alternativa=args.web_fallback,
            cache_store=cache_store,
        )
        for chunk in res:
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print()


if __name__ == "__main__":
    main()
