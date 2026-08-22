import os
import sys
import warnings
import argparse
from dotenv import load_dotenv

# Permite ejecutar con "python local_teacher/cli.py" directamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

from local_teacher.logging_config import setup_logging
from local_teacher.app import LocalTeacherApp

def main() -> None:
    """Flujo mínimo de RAG: ingestar, indexar y responder."""
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(description="CLI simple de local-teacher")
    parser.add_argument("--ingest", help="Archivo o directorio a ingerir")
    parser.add_argument("--query", help="Pregunta para el tutor")
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract PNG images. Enabled by default.",
    )
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
        "--web",
        nargs="?",
        const="",
        default=False,
        help="Habilita búsqueda web. Opcionalmente especifica un filtro (ej: --web 'site:edu')",
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
        "--ollama-fast-llm",
        default=os.getenv("OLLAMA_FAST_LLM", "llama3.2"),
        help="Modelo rápido auxiliar para optimización de consultas en modo exact (por defecto: llama3.2)",
    )
    parser.add_argument(
        "--embed-provider",
        choices=["fastembed", "ollama"],
        default=os.getenv("LOCAL_TEACHER_EMBED_PROVIDER", "fastembed"),
        help="Motor de embeddings: 'fastembed' (ONNX/C++ ultrarrápido) o 'ollama' (servidor Ollama).",
    )
    parser.add_argument(
        "--fastembed-model",
        default=os.getenv("FASTEMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        help="Modelo FastEmbed a utilizar (por defecto: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).",
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
