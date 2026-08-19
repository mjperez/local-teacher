import os
import sys
import warnings
import time
import shutil
import re
import argparse
import logging
from typing import Protocol, List, Tuple
from pathlib import Path

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*loading weights.*")
    warnings.filterwarnings("ignore", module="gliner.*")
    warnings.filterwarnings("ignore", module="huggingface_hub.*")
    warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from local_teacher.factory import obtener_llm_critico, obtener_modelos  # noqa: E402
from local_teacher.ingestion.chunker import dividir_texto  # noqa: E402
from local_teacher.ingestion.graph_builder import build_knowledge_graph  # noqa: E402
from local_teacher.ingestion.loader import (
    cargar_archivos,
    guardar_cache_jsonl,
)  # noqa: E402
from local_teacher.query.retriever import stream_consulta  # noqa: E402
from local_teacher.storage.qdrant_store import get_qdrant_retriever  # noqa: E402
from local_teacher.storage.redis_cache import get_semantic_cache_store  # noqa: E402
from local_teacher.ingestion.state_manager import get_state_manager  # noqa: E402
from local_teacher.ingestion.loader import get_cache_path  # noqa: E402

# Permite ejecutar con "python local_teacher/cli.py" directamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "local_teacher.log")

logging.basicConfig(
    filename=LOG_FILE,
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

        if not pagina or str(pagina).strip() == "":
            pagina = "N/A"

        seccion_str = str(seccion).strip()
        if not seccion_str:
            seccion = "N/A"
        else:
            seccion = re.sub(r"^#+\s*", "", seccion_str).strip()

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
        self.llm_fast = None
        if args.mode == "exact":
            self.llm_fast, _ = obtener_modelos(
                args.provider,
                ollama_llm="llama3.2",
                ollama_embed=args.ollama_embed,
                ollama_host=args.ollama_host,
                num_ctx=4096,
            )
        self.retriever = None
        self.cache_store = None
        self.chat_history = []

    def ingest(self):
        t0_ingest = time.time()
        print(f"[*] Iniciando carga de documentos desde: {self.args.ingest}")

        if self.args.recreate:
            # Limpiar Qdrant LocalFileStore
            parent_store = "./.local_teacher_parents"
            if os.path.exists(parent_store):
                shutil.rmtree(parent_store)

            if self.args.graph:
                kuzu_dir = Path("./local_teacher_kuzu").resolve()
                repo_root = Path(__file__).resolve().parent.parent
                try:
                    kuzu_dir.relative_to(repo_root)
                except ValueError:
                    print(
                        "[-] Error de seguridad: kuzu_path está fuera del directorio del proyecto."
                    )
                else:
                    for suffix in ["", ".wal", ".tmp", ".lck", ".lock"]:
                        f = Path(str(kuzu_dir) + suffix)
                        if f.exists() and not f.is_symlink():
                            if f.is_dir():
                                shutil.rmtree(f)
                            else:
                                f.unlink()
                    print("[*] Base de datos de grafos limpiada por --recreate.")

            try:

                cache_store = get_semantic_cache_store(self.embeddings)
                if cache_store.clear():
                    print("[*] Caché semántico (Redis) limpiado por --recreate.")
                else:
                    print(
                        "[-] Limpieza de caché semántico omitida o bloqueada por guardrails."
                    )
            except Exception as e:
                print(f"[-] Error al limpiar caché semántico: {e}")

            try:
                sm = get_state_manager()
                sm.clear()
                print(
                    "[*] Base de datos de estado (checkpoints) limpiada por --recreate."
                )
            except Exception as e:
                print(f"[-] Error al limpiar StateManager: {e}")

            try:
                cache_file = get_cache_path(Path(self.args.ingest))
                if cache_file.exists():
                    cache_file.unlink()
                print("[*] Caché de documentos locales limpiado por --recreate.")
            except Exception as e:
                print(f"[-] Error al limpiar caché de documentos locales: {e}")

            # Compatibilidad hacia atrás: limpiar checkpoints del formato anterior
            # si aún existen en el directorio de trabajo.
            for legacy_file in [".graph_checkpoint.json", ".loader_checkpoint.json"]:
                legacy_path = Path(legacy_file)
                if legacy_path.exists():
                    try:
                        legacy_path.unlink()
                        print(
                            f"[*] Archivo de checkpoint legacy '{legacy_file}' eliminado."
                        )
                    except Exception as e:
                        print(
                            f"[-] Error al eliminar checkpoint legacy '{legacy_file}': {e}"
                        )

        print(
            "[*] (Si es la primera vez que se procesa un PDF, Docling podría descargar modelos y tardar varios minutos...)"
        )

        try:
            if getattr(self.args, "workers", None) is not None:
                os.environ["INGEST_WORKERS"] = str(self.args.workers)

            # Exponer el número de workers via env var para que document.py pueda
            # escalar num_threads de Docling y evitar sobresuscripción de CPU.
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
                t_total = time.time() - t0_ingest

                sm = get_state_manager()
                tiempo_historico = sm.get_cumulative_time()
                t_real = t_total + tiempo_historico

                print(
                    f"[+] Ingesta completada ({len(chunks)} fragmentos) en {t_real:.2f} segundos (Sesión actual: {t_total:.2f}s)."
                )
                sm.reset_cumulative_time()

        except KeyboardInterrupt:
            t_parcial = time.time() - t0_ingest
            try:
                sm = get_state_manager()
                sm.add_cumulative_time(t_parcial)
                print("\n\n[!] Ingesta interrumpida por el usuario.")
                print(
                    f"[!] {t_parcial:.2f}s guardados en la BD para sumar mañana automáticamente."
                )
            except Exception as e:
                print(f"\n[!] Error al guardar tiempo parcial: {e}")
            sys.exit(0)

    def run_chat(self):
        print("[*] Conectando a Qdrant...")
        self.retriever = self.retriever or get_qdrant_retriever(self.embeddings)
        self.cache_store = (
            get_semantic_cache_store(self.embeddings)
            if not getattr(self.args, "no_cache", False)
            else None
        )

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
                llm_fast=self.llm_fast,
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
    # Env vars were set at the top if run as __main__

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
        "--no-cache",
        action="store_true",
        help="Ignorar caché semántico (siempre re-calcular respuesta)",
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
