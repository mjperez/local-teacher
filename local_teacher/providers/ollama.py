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
        from langchain_ollama import ChatOllama
        from langchain_ollama import OllamaEmbeddings
        
        try:
            modelos_disponibles_lista = ollama.list()
        except Exception as e:
            raise SystemError(
                f"[-] Error Crítico: No se pudo conectar al demonio de Ollama.\n"
                f"¿Está el servicio activo (localhost:11434)? Detalles: {e}"
            )
            
        if hasattr(modelos_disponibles_lista, 'models'):
            modelos_instalados = [m.model for m in modelos_disponibles_lista.models]
        else:
            modelos_instalados = [m.get("model", m.get("name")) for m in modelos_disponibles_lista.get("models", [])]
        
        def resolve_model(requested: str, installed: list) -> str:
            if requested in installed:
                return requested
            if ":" not in requested and f"{requested}:latest" in installed:
                return requested 
            
            base_name = requested.split(":")[0]
            partial_matches = [m for m in installed if m.startswith(f"{base_name}:")]
            
            if partial_matches:
                raise ValueError(
                    f"[-] Ambigüedad detectada para '{requested}'. "
                    f"Tienes estas versiones instaladas: {partial_matches}. "
                    f"Declara el modelo con su tag exacto."
                )
                
            return None 

        try:
            resolved_llm = resolve_model(self.ollama_llm, modelos_instalados)
            resolved_embed = resolve_model(self.ollama_embed, modelos_instalados)
        except ValueError as e:
            raise ValueError(e)

        missing = []
        if not resolved_llm: missing.append(self.ollama_llm)
        if not resolved_embed: missing.append(self.ollama_embed)
            
        if missing:
            raise ValueError(
                f"[-] Modelos no encontrados localmente: {missing}.\n"
                f"[*] Instalados: {modelos_instalados if modelos_instalados else 'Ninguno'}."
            )

        return ChatOllama(model=self.ollama_llm, temperature=0), OllamaEmbeddings(model=self.ollama_embed)
