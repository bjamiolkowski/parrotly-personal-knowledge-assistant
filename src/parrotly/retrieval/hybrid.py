"""
Hybrid retrieval combining dense and sparse search.

Uses Reciprocal Rank Fusion to merge FAISS and BM25 rankings.
"""

import faiss
from langchain_community.retrievers import BM25Retriever

from parrotly.retrieval.dense import dense_search
from parrotly.retrieval.sparse import sparse_search


def hybrid_search(
    query: str,
    index: faiss.Index,
    sparse_retriever: BM25Retriever,
    top_k: int = 5,
    dense_k: int = 20,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Retrieve chunk indices ranked by reciprocal rank fusion."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0 or dense_k <= 0 or rrf_k <= 0:
        raise ValueError("Retrieval parameters must be greater than zero.")

    if index.ntotal != len(sparse_retriever.docs):
        raise ValueError("Dense and sparse indexes must contain the same chunks.")

    dense_results = dense_search(query, index, top_k=dense_k)
    sparse_results = sparse_search(query, sparse_retriever)

    scores: dict[int, float] = {}

    for rank, (chunk_index, _) in enumerate(dense_results, start=1):
        scores[chunk_index] = scores.get(chunk_index, 0.0) + (
            1.0 / (rrf_k + rank)
        )

    for rank, document in enumerate(sparse_results, start=1):
        if document.id is None:
            raise ValueError("Sparse documents must have chunk indices as IDs.")

        try:
            chunk_index = int(document.id)
        except ValueError as exc:
            raise ValueError("Sparse document ID must be an integer.") from exc

        if not 0 <= chunk_index < index.ntotal:
            raise ValueError("Sparse document ID is outside the FAISS index.")

        scores[chunk_index] = scores.get(chunk_index, 0.0) + (
            1.0 / (rrf_k + rank)
        )

    ranked_results = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return ranked_results[:top_k]