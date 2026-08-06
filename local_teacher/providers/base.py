from typing import Tuple
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

class LLMProvider(ABC):
    @abstractmethod
    def get_models(self) -> Tuple[BaseChatModel, Embeddings]:
        pass
