from typing import Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, ollama_llm: str = "llama3", ollama_embed: str = "nomic-embed-text"):
        self.ollama_llm = ollama_llm
        self.ollama_embed = ollama_embed

    def get_models(self) -> Tuple[BaseChatModel, Embeddings]:
        import ollama
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        try:
            ollama.list()
        except Exception as exc:
            raise SystemError(
                f"[-] No se pudo conectar a Ollama. Revísalo en localhost:11434. Detalle: {exc}"
            )

        return ChatOllama(model=self.ollama_llm, temperature=0), OllamaEmbeddings(model=self.ollama_embed)
