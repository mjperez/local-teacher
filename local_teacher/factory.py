from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from local_teacher.providers.ollama import OllamaProvider
from local_teacher.providers.openai import OpenAIProvider


def obtener_modelos(provider: str, ollama_llm: str = "llama3", ollama_embed: str = "nomic-embed-text") -> tuple[BaseChatModel, Embeddings]:
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider().get_models()
    if provider == "ollama":
        return OllamaProvider(ollama_llm, ollama_embed).get_models()

    raise ValueError(f"[-] Proveedor no soportado: {provider}")