import os
import logging
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

_log = logging.getLogger(__name__)

def obtener_modelos(
    provider: str,
    ollama_llm: str = "deepseek-r1:8b",
    ollama_embed: str = "granite-embedding:278m",
    ollama_host: str | None = None,
    num_ctx: int = 16384,
    embed_provider: str = "fastembed",
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[BaseChatModel, Embeddings]:
    """Inicializa y devuelve el LLM y el modelo de embeddings."""
    provider = provider.lower()
    embed_provider = embed_provider.lower()
    
    # 1. Selección de modelo de embeddings
    embeddings: Embeddings
    if embed_provider == "fastembed" or ollama_embed.startswith("fastembed:") or ollama_embed == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        model_name = (
            ollama_embed.split("fastembed:", 1)[1]
            if ollama_embed.startswith("fastembed:")
            else fastembed_model
        )
        _log.info(f"[*] Inicializando FastEmbed local (ONNX/C++) con modelo: {model_name}")
        embeddings = FastEmbedEmbeddings(model_name=model_name)
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        from langchain_ollama import OllamaEmbeddings
        raw_host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = raw_host.strip('"').strip("'")
        if host in ("http://0.0.0.0", "0.0.0.0"):
            host = "http://127.0.0.1:11434"
        _embed_keep_alive = int(os.getenv("OLLAMA_EMBED_KEEP_ALIVE", "0"))
        embeddings = OllamaEmbeddings(model=ollama_embed, base_url=host, keep_alive=_embed_keep_alive)

    # 2. Selección de LLM
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0), embeddings
        
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        raw_host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = raw_host.strip('"').strip("'")
        if host in ("http://0.0.0.0", "0.0.0.0"):
            host = "http://127.0.0.1:11434"
            
        _log.info(f"[*] Conectando a Ollama en {host} (LLM: {ollama_llm})")
        _keep_alive = int(os.getenv("OLLAMA_KEEP_ALIVE", "300"))
        return (
            ChatOllama(model=ollama_llm, temperature=0, base_url=host, num_ctx=num_ctx, keep_alive=_keep_alive),
            embeddings,
        )
        
    raise ValueError(f"[-] Proveedor no soportado: {provider}")


def obtener_llm_critico(
    provider: str,
    ollama_critic_llm: str = "granite3-guardian:2b",
    ollama_host: str | None = None,
    num_ctx: int = 8192,
) -> BaseChatModel:
    """Inicializa y devuelve el LLM específico para el supervisor (Critic)."""
    provider = provider.lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0)
        
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        raw_host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = raw_host.strip('"').strip("'")
        
        if host == "http://0.0.0.0" or host == "0.0.0.0":
            host = "http://127.0.0.1:11434"
            
        _log.info(f"[*] Conectando a Ollama Crítico en {host} (LLM: {ollama_critic_llm})")
        _keep_alive = int(os.getenv("OLLAMA_KEEP_ALIVE", "300"))
        return ChatOllama(model=ollama_critic_llm, temperature=0, base_url=host, num_ctx=num_ctx, keep_alive=_keep_alive)
        
    raise ValueError(f"[-] Proveedor no soportado: {provider}")