import os
import sys
import time
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from local_teacher.factory import obtener_llm_critico, obtener_modelos
from local_teacher.ingestion.chunker import dividir_texto
from local_teacher.ingestion.graph_builder import build_knowledge_graph
from local_teacher.ingestion.loader import (
    cargar_archivos,
    guardar_cache_jsonl,
    get_cache_path
)
from local_teacher.storage.qdrant_store import get_qdrant_retriever
from local_teacher.storage.redis_cache import get_semantic_cache_store
from local_teacher.ingestion.state_manager import get_state_manager
from local_teacher.query.pipeline import PipelineConsulta
from local_teacher.query.prompts import _print_sources


def _progreso_print(
    paso: int, total: int = 4, mensaje: str = "", saltar_linea: bool = False
) -> None:
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    porcentaje = int((paso / total) * 100)
    barra = "█" * (porcentaje // 10) + "░" * (10 - (porcentaje // 10))
    texto = f"\r[{barra}] {porcentaje:3}% | {mensaje}"
    texto = texto + " " * max(0, terminal_width - len(texto) - 1)

    if saltar_linea:
        try:
            print(texto, flush=True)
            print() # Extra blank line
        except UnicodeEncodeError:
            print(texto.replace("█", "=").replace("░", "-"), flush=True)
            print()
    else:
        try:
            print(texto, end="", flush=True)
        except UnicodeEncodeError:
            print(texto.replace("█", "=").replace("░", "-"), end="", flush=True)


class LocalTeacherApp:
    def __init__(self, args):
        self.args = args
        self.llm, self.embeddings = obtener_modelos(
            args.provider,
            ollama_llm=args.ollama_llm,
            ollama_embed=args.ollama_embed,
            ollama_host=args.ollama_host,
            embed_provider=getattr(args, "embed_provider", "fastembed"),
            fastembed_model=getattr(args, "fastembed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        )
        self.llm_critic = obtener_llm_critico(
            args.provider,
            ollama_critic_llm=args.ollama_critic_llm,
            ollama_host=args.ollama_host,
        )
        self.llm_fast = None
        if args.mode == "exact":
            fast_model = getattr(args, "ollama_fast_llm", "llama3.2")
            self.llm_fast, _ = obtener_modelos(
                args.provider,
                ollama_llm=fast_model,
                ollama_embed=args.ollama_embed,
                ollama_host=args.ollama_host,
                num_ctx=4096,
            )
        self.retriever = None
        self.cache_store = None
        self.chat_history = []
        self.busqueda_web_alternativa = getattr(args, "web", False) is not False
        self.web_filter = args.web if isinstance(getattr(args, "web", False), str) else ""
        
        embed_display = (
            f"FastEmbed ({args.fastembed_model.split('/')[-1]})"
            if getattr(args, "embed_provider", "fastembed") == "fastembed"
            else f"Ollama ({args.ollama_embed})"
        )
        print(
            f"[*] Modelos inicializados (LLM: {args.ollama_llm}, Embed: {embed_display}, Supervisor: {args.ollama_critic_llm})"
        )

    def ingest(self):
        t0_ingest = time.time()
        print(f"[*] Iniciando carga de documentos desde: {self.args.ingest}")

        if self.args.recreate:
            # Limpiar Qdrant LocalFileStore
            parent_store = "./.local_teacher_parents"
            if os.path.exists(parent_store):
                shutil.rmtree(parent_store)

            if self.args.graph:
                try:
                    from local_teacher.graph import get_memgraph_client

                    mg = get_memgraph_client()
                    mg.clear_graph()
                    print("[*] Base de datos de grafos (Memgraph) limpiada por --recreate.")
                except Exception as e:
                    print(f"[-] Error al limpiar Grafo de Conocimiento en Memgraph: {e}")

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
            else:
                print("[-] No se encontraron o extrajeron documentos. Inicializando el recuperador con la base de datos existente.")
                self.retriever = get_qdrant_retriever(self.embeddings, force_recreate=False)

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
            pipeline = PipelineConsulta(
                retriever=self.retriever,
                llm=self.llm,
                busqueda_web_alternativa=self.busqueda_web_alternativa,
                web_filter=self.web_filter,
                cache_store=self.cache_store,
                usar_critico=self.args.critic,
                llm_critic=self.llm_critic,
                llm_fast=self.llm_fast,
            )
            res_gen = pipeline.ejecutar(consulta_actual, self.chat_history, _progreso_print)

            respuesta_final = ""
            context_docs = []

            for chunk in res_gen:
                if "answer" in chunk:
                    respuesta_final += chunk["answer"]
                    print(chunk["answer"], end="", flush=True)
                if "context_docs" in chunk:
                    context_docs = chunk["context_docs"]

            _print_sources(context_docs, respuesta_final=respuesta_final)

            # Guardar en memoria
            self.chat_history.append(HumanMessage(content=consulta_actual))
            self.chat_history.append(AIMessage(content=respuesta_final))

            # Loop interactivo
            try:
                print("\n")
                while True:
                    entrada = input(
                        "Haz una pregunta de seguimiento (o 'salir' para terminar, '/web' para opciones): "
                    ).strip()
                    
                    if not entrada:
                        continue

                    if entrada.lower() in ("salir", "exit", "quit", "q"):
                        return

                    if entrada.startswith("/web"):
                        comando = entrada[4:].strip()
                        if comando.lower() in ("off", "false"):
                            self.busqueda_web_alternativa = False
                            self.web_filter = ""
                            print("[*] Búsqueda web deshabilitada.")
                        elif comando.lower() in ("on", "true", ""):
                            self.busqueda_web_alternativa = True
                            self.web_filter = ""
                            print("[*] Búsqueda web habilitada (sin filtros).")
                        elif comando.lower().startswith("filter "):
                            self.busqueda_web_alternativa = True
                            self.web_filter = comando[7:].strip()
                            print(f"[*] Búsqueda web habilitada con filtro: {self.web_filter}")
                        else:
                            print("[-] Uso incorrecto de /web. Opciones:")
                            print("    /web on          -> Activar búsqueda web")
                            print("    /web off         -> Desactivar búsqueda web")
                            print("    /web filter <x>  -> Activar con filtro (ej: /web filter site:edu)")
                        continue

                    consulta_actual = entrada
                    break
            except (KeyboardInterrupt, EOFError):
                break
