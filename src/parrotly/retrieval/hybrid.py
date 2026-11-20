"""Hybrid retrieval helpers."""

import numpy as np

from src.parrotly.retrieval.dense import dense_search
from src.parrotly.retrieval.sparse import sparse_search


def normalize_scores(score_list: list[tuple[int, float]]) -> dict[int, float]:
    """Normalize scores to 0-1 range."""
    if not score_list:
        return {}

    values = np.array([score for _, score in score_list], dtype=float)
    min_score = values.min()
    max_score = values.max()

    if max_score - min_score < 1e-12:
        return {idx: 1.0 for idx, _ in score_list}

    return {
        idx: (score - min_score) / (max_score - min_score)
        for idx, score in score_list
    }


def build_result(
    chunk: dict,
    vector_score: float,
    tfidf_score: float,
    hybrid_score: float,
) -> dict:
    """Build retrieval result."""
    return {
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "vector_score": float(vector_score),
        "tfidf_score": float(tfidf_score),
        "hybrid_score": float(hybrid_score),
    }


def hybrid_search(
    query: str,
    index,
    chunks: list[dict],
    vectorizer,
    tfidf_matrix,
    top_k: int = 5,
    faiss_k: int = 20,
    tfidf_k: int = 20,
    alpha: float = 0.6,
) -> list[dict]:
    """Run hybrid search."""
    if not query.strip():
        return []

    alpha = min(max(alpha, 0.0), 1.0)

    dense_results = dense_search(query, index, top_n=faiss_k)
    sparse_results = sparse_search(query, vectorizer, tfidf_matrix, top_n=tfidf_k)

    dense_scores = normalize_scores(dense_results)
    sparse_scores = normalize_scores(sparse_results)

    candidates = []
    candidate_ids = set(dense_scores) | set(sparse_scores)

    for idx in candidate_ids:
        vector_score = dense_scores.get(idx, 0.0)
        tfidf_score = sparse_scores.get(idx, 0.0)
        hybrid_score = alpha * vector_score + (1 - alpha) * tfidf_score

        candidates.append(
            build_result(
                chunk=chunks[idx],
                vector_score=vector_score,
                tfidf_score=tfidf_score,
                hybrid_score=hybrid_score,
            )
        )

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return candidates[:top_k]


def retrieve_chunks(
    query: str,
    index,
    chunks: list[dict],
    vectorizer,
    tfidf_matrix,
    top_k: int = 5,
    faiss_k: int = 20,
    tfidf_k: int = 20,
    alpha: float = 0.6,
    mode: str = "hybrid",
) -> list[dict]:
    """Retrieve chunks."""
    if not query.strip():
        return []

    if mode == "hybrid":
        return hybrid_search(
            query=query,
            index=index,
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_k=top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
        )

    if mode == "dense":
        results = dense_search(query, index, top_n=top_k)
        return [
            build_result(
                chunk=chunks[idx],
                vector_score=score,
                tfidf_score=0.0,
                hybrid_score=score,
            )
            for idx, score in results
        ]

    if mode == "sparse":
        results = sparse_search(query, vectorizer, tfidf_matrix, top_n=top_k)
        return [
            build_result(
                chunk=chunks[idx],
                vector_score=0.0,
                tfidf_score=score,
                hybrid_score=score,
            )
            for idx, score in results
        ]

    raise ValueError(f"Unknown retrieval mode: {mode}")