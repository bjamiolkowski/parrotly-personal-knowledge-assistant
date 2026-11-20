"""
Dense retrieval module using FAISS.

Provides embedding-based semantic search for the RAG pipeline.
"""

import faiss

from src.parrotly.indexing.embedder import embed_text


def dense_search(query: str, index: faiss.Index, top_n: int = 20) -> list[tuple[int, float]]:
    """
    Retrieve nearest chunks using FAISS semantic search.
    """
    if not query.strip():
        return []

    query_vec = embed_text(query)
    scores, indices = index.search(query_vec, top_n)

    return [
        (int(idx), float(scores[0][rank]))
        for rank, idx in enumerate(indices[0])
        if idx != -1
    ]