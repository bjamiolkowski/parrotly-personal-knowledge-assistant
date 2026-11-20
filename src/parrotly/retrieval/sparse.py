"""
Sparse retrieval module using TF-IDF.

Provides keyword-based retrieval to complement semantic search.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_index(chunks: list[dict]) -> tuple[TfidfVectorizer, np.ndarray]:
    """
    Build a TF-IDF index from document chunks.
    """
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


def sparse_search(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """
    Retrieve relevant chunks using TF-IDF cosine similarity.
    """
    if not query.strip():
        return []

    # Vectorize query
    query_vec = vectorizer.transform([query])

    # Compute similarity
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top-N indices
    top_indices = np.argsort(sims)[::-1][:top_n]

    return [(int(idx), float(sims[idx])) for idx in top_indices]


def retrieve_chunks(query: str, index,chunks: list[dict], vectorizer, tfidf_matrix, top_k: int = 5, faiss_k: int = 20,
    tfidf_k: int = 20, alpha: float = 0.6, mode: str = "hybrid") -> list[dict]:
    """
    Retrieve chunks using selected retrieval mode:
    - dense: FAISS only
    - sparse: TF-IDF only
    - hybrid: FAISS + TF-IDF
    """
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
        dense_results = dense_search(query, index, top_n=top_k)

        return [
            {
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "vector_score": float(score),
                "tfidf_score": 0.0,
                "hybrid_score": float(score),
            }
            for idx, score in dense_results
        ]

    if mode == "sparse":
        sparse_results = sparse_search(
            query=query,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_n=top_k,
        )

        return [
            {
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "vector_score": 0.0,
                "tfidf_score": float(score),
                "hybrid_score": float(score),
            }
            for idx, score in sparse_results
        ]

    raise ValueError(f"Unknown retrieval mode: {mode}")