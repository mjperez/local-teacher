from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore


def ejecutar_query(vectorstore: QdrantVectorStore, llm: BaseChatModel, query: str, stream: bool = False) -> Any:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Responde solo con la información presente en el contexto.\n\n"
            "IMPORTANTE SOBRE EL FORMATO: Como esta respuesta se leerá en una consola de texto, NO uses formato LaTeX para las fórmulas matemáticas (no uses \\[ \\] ni \\( \\)). "
            "Escribe las fórmulas de la manera más sencilla, legible y común posible usando texto plano (ejemplo: usa 'E = h * f' en lugar de 'E = h \\nu').\n\n"
            "Contexto:\n{context}\n\n"
            "Si no está en el contexto, responde exactamente: 'No poseo información suficiente'.",
        ),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    if not stream:
        return rag_chain.invoke({"input": query})

    def _stream_generator():
        # Recuperamos documentos primero y pasamos al qa_chain para streamear de a tokens
        docs = retriever.invoke(query)
        # qa_chain.stream puede devolver strings o AIMessageChunks dependiendo de la versión de langchain
        has_yielded = False
        for chunk in qa_chain.stream({"context": docs, "input": query}):
            has_yielded = True
            text_chunk = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield {"answer": text_chunk}
        
        if not has_yielded:
            yield {"answer": "\n[Error: El modelo no devolvió ninguna respuesta (resultado vacío). Verifica la conexión o el modelo.]\n"}

    return _stream_generator()