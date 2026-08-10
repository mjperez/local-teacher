from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore


def ejecutar_query(vectorstore: QdrantVectorStore, llm: BaseChatModel, query: str) -> dict[str, Any]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Responde solo con la información presente en el contexto. "
            "Si no está en el contexto, responde exactamente: 'No poseo información suficiente'.",
        ),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain.invoke({"input": query})