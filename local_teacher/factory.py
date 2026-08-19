import os
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

def obtener_modelos(
    provider: str,
    ollama_llm: str = "deepseek-r1:8b",
    ollama_embed: str = "granite-embedding:278m",
    ollama_host: str | None = None,
    num_ctx: int = 16384,
) -> tuple[BaseChatModel, Embeddings]:
    """Inicializa y devuelve el LLM y el modelo de embeddings."""
    provider = provider.lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0), OpenAIEmbeddings()
        
    if provider == "ollama":
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        raw_host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = raw_host.strip('"').strip("'")
        
        if host == "http://0.0.0.0" or host == "0.0.0.0":
            host = "http://127.0.0.1:11434"
        
            
        print(f"[*] Conectando a Ollama en {host} (LLM: {ollama_llm}, Embed: {ollama_embed})")
        return (
            ChatOllama(model=ollama_llm, temperature=0, base_url=host, num_ctx=num_ctx, keep_alive=300),
            OllamaEmbeddings(model=ollama_embed, base_url=host, keep_alive=300),
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
            
        print(f"[*] Conectando a Ollama Crítico en {host} (LLM: {ollama_critic_llm})")
        return ChatOllama(model=ollama_critic_llm, temperature=0, base_url=host, num_ctx=num_ctx, keep_alive=300)
        
    raise ValueError(f"[-] Proveedor no soportado: {provider}")