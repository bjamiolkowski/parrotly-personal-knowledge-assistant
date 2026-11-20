"""Integration tests for Ollama embedding generation."""

import numpy as np
from langchain_core.documents import Document

from parrotly.indexing.embedder import build_embeddings, embed_text


def test_ollama_embeddings() -> None:
    """Verify embedding generation with the configured Ollama model."""
    query = "What is retrieval-augmented generation?"

    chunks = [
        Document(page_content="RAG combines retrieval with text generation."),
        Document(page_content="FAISS enables efficient vector similarity search."),
    ]

    query_embedding = embed_text(query)
    document_embeddings = build_embeddings(chunks)

    assert query_embedding.shape[0] == 1
    assert document_embeddings.shape[0] == len(chunks)
    assert query_embedding.shape[1] == document_embeddings.shape[1] > 0

    assert query_embedding.dtype == np.float32
    assert document_embeddings.dtype == np.float32

    assert np.isfinite(query_embedding).all()
    assert np.isfinite(document_embeddings).all()

    np.testing.assert_allclose(
        np.linalg.norm(query_embedding, axis=1),
        1.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.linalg.norm(document_embeddings, axis=1),
        1.0,
        atol=1e-5,
    )