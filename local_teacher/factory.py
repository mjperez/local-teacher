from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from providers.openai import OpenAIProvider
from providers.ollama import OllamaProvider

def obtener_modelos(provider: str, ollama_llm: str = "llama3", ollama_embed: str = "nomic-embed-text") -> tuple[BaseChatModel, Embeddings]:
    if provider == "openai":
        return OpenAIProvider().get_models()
    elif provider == "ollama":
        return OllamaProvider(ollama_llm, ollama_embed).get_models()
    else:
        raise ValueError(f"[-] Proveedor no soportado: {provider}")