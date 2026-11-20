
"""
Rerank retrieved documents using a cross-encoder.

Scores query-document pairs and selects the most relevant documents.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def build_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    """Load a cross-encoder model for document reranking."""
    if not model_name.strip():
        raise ValueError("Model name cannot be empty.")

    return CrossEncoder(model_name)


def rerank_documents(
    query: str,
    documents: list[Document],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[Document]:
    """Return the top documents ranked by cross-encoder relevance."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not documents:
        return []

    pairs = [(query, document.page_content) for document in documents]
    scores = reranker.predict(pairs)

    if len(scores) != len(documents):
        raise ValueError("Cross-encoder returned an unexpected number of scores.")

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [document for document, _ in ranked[:top_k]]