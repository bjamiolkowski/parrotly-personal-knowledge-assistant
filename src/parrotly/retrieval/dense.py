"""Retrieve document chunks using FAISS semantic similarity search."""

import faiss
import numpy as np

from parrotly.indexing.embedder import embed_text


def dense_search(
    query: str,
    index: faiss.Index,
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """Return chunk indices and cosine similarity scores."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if index.ntotal == 0:
        return []

    query_embedding = np.asarray(embed_text(query), dtype=np.float32)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.shape != (1, index.d):
        raise ValueError("Query embedding dimensions do not match the FAISS index.")

    if not np.isfinite(query_embedding).all() or not np.any(query_embedding):
        raise ValueError("Query embedding contains invalid values.")

    query_embedding = np.ascontiguousarray(query_embedding)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, min(top_k, index.ntotal))

    return [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx != -1
    ]