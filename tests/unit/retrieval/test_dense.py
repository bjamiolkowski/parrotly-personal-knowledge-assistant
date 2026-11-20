"""Unit tests for FAISS dense retrieval."""

import faiss
import numpy as np
import pytest

from parrotly.retrieval import dense


@pytest.fixture
def index():
    """Create a small FAISS index with normalized vectors."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    return index


def test_dense_search_returns_nearest_chunk(index, monkeypatch):
    monkeypatch.setattr(
        dense, "embed_text", lambda query: [1.0, 0.0, 0.0]
    )

    results = dense.dense_search("test query", index, top_k=2)

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(1.0)


def test_dense_search_normalizes_query_embedding(index, monkeypatch):
    monkeypatch.setattr(
        dense, "embed_text", lambda query: [5.0, 0.0, 0.0]
    )

    results = dense.dense_search("test query", index, top_k=1)

    assert results[0][1] == pytest.approx(1.0)


def test_dense_search_limits_results_to_index_size(index, monkeypatch):
    monkeypatch.setattr(
        dense, "embed_text", lambda query: [1.0, 0.0, 0.0]
    )

    results = dense.dense_search("test query", index, top_k=10)

    assert len(results) == index.ntotal


def test_dense_search_returns_empty_for_empty_index():
    index = faiss.IndexFlatIP(3)

    results = dense.dense_search("test query", index)

    assert results == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_dense_search_rejects_empty_query(query, index):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        dense.dense_search(query, index)


@pytest.mark.parametrize("top_k", [0, -1])
def test_dense_search_rejects_invalid_top_k(top_k, index):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        dense.dense_search("test query", index, top_k=top_k)


def test_dense_search_rejects_dimension_mismatch(index, monkeypatch):
    monkeypatch.setattr(
        dense, "embed_text", lambda query: [1.0, 0.0]
    )

    with pytest.raises(ValueError, match="dimensions do not match"):
        dense.dense_search("test query", index)


@pytest.mark.parametrize(
    "embedding",
    [
        [0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [np.inf, 0.0, 0.0],
    ],
)
def test_dense_search_rejects_invalid_embeddings(
    embedding, index, monkeypatch
):
    monkeypatch.setattr(
        dense, "embed_text", lambda query: embedding
    )

    with pytest.raises(ValueError, match="invalid values"):
        dense.dense_search("test query", index)
