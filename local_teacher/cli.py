import argparse
import logging
import os
import sys

# Permite ejecutar con "python local_teacher/cli.py" directamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from local_teacher.factory import obtener_modelos, obtener_llm_critico
from local_teacher.ingestion.chunker import dividir_texto
from local_teacher.ingestion.graph_builder import build_knowledge_graph
from local_teacher.ingestion.loader import cargar_archivos, guardar_cache_jsonl
from local_teacher.query.retriever import stream_consulta
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.storage.redis_cache import get_semantic_cache_store

logging.basicConfig(
    filename="local_teacher.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.INFO)


def _print_sources(docs):
    if not docs:
        return
    print("\n\n---\nFuentes citadas:")
    
    grupos = {}
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        fuente = meta.get("fuente", "Desconocida")
        fuente_nombre = os.path.basename(fuente)
        pagina = meta.get("pagina", "N/A")
        seccion = meta.get("ruta_seccion", "N/A")
        tipo = meta.get("tipo_archivo", "texto")
        ruta_recurso = meta.get("ruta_recurso", "N/A")

        clave = (fuente_nombre, pagina, seccion, tipo, ruta_recurso)
        if clave not in grupos:
            grupos[clave] = []
        grupos[clave].append(i)

    for clave, indices in grupos.items():
        fuente_nombre, pagina, seccion, tipo, ruta_recurso = clave
        inds_str = ", ".join(f"[{i}]" for i in indices)
        info = f"{inds_str} Archivo: {fuente_nombre} | Pag: {pagina} | Sección: {seccion}"
        if tipo in ("figura", "tabla"):
            info += f" | Tipo: {tipo} | Ruta: {ruta_recurso}"
        print(info)
        
    print("---")


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
    parser.add_argument(
        "--critic",
        action="store_true",
        help="Activa el supervisor interno (Self-RAG). Más preciso pero más lento. Por defecto desactivado.",
    )

    # Modelos
    parser.add_argument(
        "--provider", default=os.getenv("LOCAL_TEACHER_PROVIDER", "ollama")
    )
    parser.add_argument("--ollama-llm", default=os.getenv("OLLAMA_LLM", "llama3.2"))
    parser.add_argument("--ollama-critic-llm", default=os.getenv("OLLAMA_CRITIC_LLM", "granite3-guardian:2b"))
    parser.add_argument(
        "--ollama-embed", default=os.getenv("OLLAMA_EMBED", "granite-embedding:278m")
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
    
    llm_critic = obtener_llm_critico(
        args.provider,
        ollama_critic_llm=args.ollama_critic_llm,
        ollama_host=args.ollama_host,
    )

    retriever = None

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
            retriever = get_qdrant_retriever(
                embeddings, chunks, force_recreate=args.recreate
            )
            print(f"[+] Ingesta completada ({len(chunks)} fragmentos).")

    if args.query:
        print("[*] Conectando a Qdrant...")
        retriever = retriever or get_qdrant_retriever(embeddings)
        cache_store = get_semantic_cache_store()

        chat_history = []

        consulta_actual = args.query
        while True:
            print("\n[Tutor Local]: Procesando tu pregunta...")
            res_gen = stream_consulta(
                retriever,
                llm,
                consulta_actual,
                chat_history=chat_history,
                busqueda_web_alternativa=args.web_fallback,
                cache_store=cache_store,
                usar_critico=args.critic,
                llm_critic=llm_critic,
            )

            respuesta_final = ""
            context_docs = []

            for chunk in res_gen:
                if "answer" in chunk:
                    respuesta_final += chunk["answer"]
                    print(chunk["answer"], end="", flush=True)
                if "context_docs" in chunk:
                    context_docs = chunk["context_docs"]

            _print_sources(context_docs)

            # Guardar en memoria
            chat_history.append(HumanMessage(content=consulta_actual))
            chat_history.append(AIMessage(content=respuesta_final))

            # Loop
            try:
                print("\n")
                consulta_actual = input(
                    "Haz una pregunta de seguimiento (o 'salir' para terminar): "
                )
                if consulta_actual.strip().lower() in ("salir", "exit", "quit"):
                    break
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
