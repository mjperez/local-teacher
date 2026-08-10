from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def get_models(self) -> tuple[BaseChatModel, Embeddings]:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0), OpenAIEmbeddings()
