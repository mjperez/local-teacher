import logging
from typing import Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from flashrank import Ranker, RerankRequest

_log = logging.getLogger(__name__)

_ranker: Optional[Ranker] = None


def get_ranker() -> Ranker:
    """Inicializa de forma diferida el modelo local de reranking FlashRank."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def recuperar_y_filtrar(
    retriever: BaseRetriever,
    consulta_optimizada: str,
    capitulo_filtro: Optional[str] = None,
    entidades_filtro: Optional[list[str]] = None,
) -> list[Document]:
    """Recupera candidatos híbridos desde Qdrant y aplica reranking neuronal con FlashRank."""
    docs: list[Document] = []
    vistos_hash = set()

    # Búsqueda individual por entidades clave
    if entidades_filtro:
        for entidad in entidades_filtro:
            if not entidad:
                continue
            retriever.search_kwargs = {"k": 15}
            res_entidad = retriever.invoke(str(entidad))
            for d in res_entidad:
                h = hash(d.page_content)
                if h not in vistos_hash:
                    docs.append(d)
                    vistos_hash.add(h)

    # Búsqueda con la consulta global enriquecida
    retriever.search_kwargs = {"k": 25}
    res_completa = retriever.invoke(consulta_optimizada)
    for d in res_completa:
        h = hash(d.page_content)
        if h not in vistos_hash:
            docs.append(d)
            vistos_hash.add(h)

    # Filtrar por capítulo si fue especificado
    if capitulo_filtro:
        docs_filtrados = []
        prefix = f"{capitulo_filtro}."
        for d in docs:
            ruta = d.metadata.get("ruta_seccion", "")
            if ruta.strip().startswith(prefix) or f" {prefix}" in ruta:
                docs_filtrados.append(d)
        if docs_filtrados:
            docs = docs_filtrados

    if not docs:
        return []

    # Reordenamiento neuronal con FlashRank
    ranker = get_ranker()
    passages = [
        {"id": idx, "text": d.page_content, "meta": d.metadata}
        for idx, d in enumerate(docs)
    ]

    rerank_request = RerankRequest(query=consulta_optimizada, passages=passages)
    resultados = ranker.rerank(rerank_request)

    final_docs: list[Document] = []
    min_score = 0.1

    for res in resultados[:10]:
        if res["score"] < min_score:
            continue
        doc = Document(page_content=res["text"], metadata=res["meta"])
        doc.metadata["rerank_score"] = res["score"]
        final_docs.append(doc)

    # Si ningún documento supera el umbral, retornar los 3 mejores como respaldo
    if not final_docs and resultados:
        for res in resultados[:3]:
            doc = Document(page_content=res["text"], metadata=res["meta"])
            doc.metadata["rerank_score"] = res["score"]
            final_docs.append(doc)

    return final_docs
