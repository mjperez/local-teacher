import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from local_teacher.factory import obtener_llm_critico, obtener_modelos
from local_teacher.ingestion.chunker import dividir_texto
from local_teacher.ingestion.graph_builder import build_knowledge_graph
from local_teacher.ingestion.loader import cargar_archivos, guardar_cache_jsonl
from local_teacher.query.retriever import stream_consulta
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.storage.redis_cache import get_semantic_cache_store



# Permite ejecutar con "python local_teacher/cli.py" directamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


logging.basicConfig(
    filename="local_teacher.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("fastembed").setLevel(logging.ERROR)
logging.getLogger("multipart").setLevel(logging.WARNING)

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


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
        info = (
            f"{inds_str} Archivo: {fuente_nombre} | Pag: {pagina} | Sección: {seccion}"
        )
        if tipo in ("figura", "tabla"):
            info += f" | Tipo: {tipo} | Ruta: {ruta_recurso}"
        print(info)

    print("---")


class LocalTeacherApp:
    def __init__(self, args):
        self.args = args
        self.llm, self.embeddings = obtener_modelos(
            args.provider,
            ollama_llm=args.ollama_llm,
            ollama_embed=args.ollama_embed,
            ollama_host=args.ollama_host,
        )
        self.llm_critic = obtener_llm_critico(
            args.provider,
            ollama_critic_llm=args.ollama_critic_llm,
            ollama_host=args.ollama_host,
        )
        self.retriever = None
        self.cache_store = None
        self.chat_history = []

    def ingest(self):
        print(f"[*] Iniciando carga de documentos desde: {self.args.ingest}")

        if self.args.recreate:
            import shutil

            # Limpiar Qdrant LocalFileStore
            parent_store = "./.local_teacher_parents"
            if os.path.exists(parent_store):
                shutil.rmtree(parent_store)

            if self.args.graph:
                kuzu_path = "./local_teacher_kuzu"
                if os.path.exists(kuzu_path):
                    if os.path.isdir(kuzu_path):
                        shutil.rmtree(kuzu_path)
                    else:
                        os.remove(kuzu_path)
                if os.path.exists(kuzu_path + ".wal"):
                    os.remove(kuzu_path + ".wal")
                if os.path.exists("./.graph_checkpoint.json"):
                    os.remove("./.graph_checkpoint.json")
                print("[*] Base de datos de grafos limpiada por --recreate.")

            try:
                import redis

                r = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                )
                r.flushdb()
                print("[*] Caché semántico (Redis) limpiado por --recreate.")
            except Exception:
                pass

            if os.path.exists("./.loader_checkpoint.json"):
                os.remove("./.loader_checkpoint.json")
            cache_file = Path(self.args.ingest).with_suffix(".jsonl")
            if cache_file.exists():
                cache_file.unlink()
            print("[*] Caché de documentos locales limpiado por --recreate.")

        print(
            "[*] (Si es la primera vez que se procesa un PDF, Docling podría descargar modelos y tardar varios minutos...)"
        )
        docs = cargar_archivos(
            self.args.ingest,
            extraer_figuras=self.args.figures,
            extraer_tablas=self.args.tables,
            enriquecer_formulas=self.args.with_formulas,
            max_workers=self.args.workers,
        )
        if docs:
            guardar_cache_jsonl(docs, self.args.ingest)

            print("[*] Dividiendo texto en fragmentos (chunking)...")
            chunks = dividir_texto(docs)

            if self.args.graph:
                build_knowledge_graph(chunks, self.llm)

            print(
                "[*] Generando embeddings y guardando en Qdrant (esto puede tomar un tiempo)..."
            )
            self.retriever = get_qdrant_retriever(
                self.embeddings, chunks, force_recreate=self.args.recreate
            )
            print(f"[+] Ingesta completada ({len(chunks)} fragmentos).")

    def run_chat(self):
        print("[*] Conectando a Qdrant...")
        self.retriever = self.retriever or get_qdrant_retriever(self.embeddings)
        self.cache_store = get_semantic_cache_store(self.embeddings)

        consulta_actual = self.args.query
        while True:
            print("\n[Tutor Local]: Procesando tu pregunta...")
            res_gen = stream_consulta(
                self.retriever,
                self.llm,
                consulta_actual,
                chat_history=self.chat_history,
                busqueda_web_alternativa=self.args.web_fallback,
                cache_store=self.cache_store,
                usar_critico=self.args.critic,
                llm_critic=self.llm_critic,
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
            self.chat_history.append(HumanMessage(content=consulta_actual))
            self.chat_history.append(AIMessage(content=respuesta_final))

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


def main() -> None:
    """Flujo mínimo de RAG: ingestar, indexar y responder."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*loading weights.*")
    warnings.filterwarnings("ignore", module="gliner.*")
    warnings.filterwarnings("ignore", module="huggingface_hub.*")
    warnings.filterwarnings("ignore", category=UserWarning)

    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI simple de local-teacher")
    parser.add_argument("--ingest", help="Archivo o directorio a ingerir")
    parser.add_argument("--query", help="Pregunta para el tutor")
    parser.add_argument("--figures", action="store_true", help="Extract PNG images")
    parser.add_argument(
        "--tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract tables to CSV/MD. Enabled by default.",
    )
    parser.add_argument(
        "--with-formulas", action="store_true", help="Enable Docling VLM for formulas"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate Qdrant collection"
    )
    parser.add_argument(
        "--graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract and build Knowledge Graph (GraphRAG). Enabled by default.",
    )
    parser.add_argument(
        "--web-fallback",
        action="store_true",
        help="Allow web fallback if local documents don't have the answer",
    )
    parser.add_argument(
        "--critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable internal supervisor (Self-RAG). Enabled by default. Use --no-critic to disable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de procesos concurrentes para la extracción de documentos (por defecto: 1).",
    )

    parser.add_argument(
        "--mode",
        choices=["fast", "exact", "ultra-fast"],
        default="exact",
        help="Execution mode. 'exact' uses deepseek-r1:8b, 'fast' uses llama3.2, 'ultra-fast' uses llama3.2 without the internal critic.",
    )

    # Modelos
    parser.add_argument(
        "--provider", default=os.getenv("LOCAL_TEACHER_PROVIDER", "ollama")
    )
    parser.add_argument(
        "--ollama-llm", default=os.getenv("OLLAMA_LLM", "deepseek-r1:8b")
    )
    parser.add_argument(
        "--ollama-critic-llm",
        default=os.getenv("OLLAMA_CRITIC_LLM", "granite3-guardian:2b"),
    )
    parser.add_argument(
        "--ollama-embed", default=os.getenv("OLLAMA_EMBED", "granite-embedding:278m")
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )

    args = parser.parse_args()

    # Sobreescribir modelo y flags según el modo seleccionado
    if args.mode == "fast":
        args.ollama_llm = "llama3.2"
    elif args.mode == "ultra-fast":
        args.ollama_llm = "llama3.2"
        args.critic = False
    elif args.mode == "exact":
        args.ollama_llm = "deepseek-r1:8b"

    if not args.ingest and not args.query:
        print("[-] Error: define --ingest y/o --query")
        sys.exit(1)

    app = LocalTeacherApp(args)
    if args.ingest:
        app.ingest()

    if args.query:
        app.run_chat()


if __name__ == "__main__":
    main()
