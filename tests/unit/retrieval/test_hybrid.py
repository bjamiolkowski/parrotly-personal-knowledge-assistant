
"""Unit tests for hybrid retrieval using Reciprocal Rank Fusion."""

import faiss
import numpy as np
import pytest

from langchain_core.documents import Document

from parrotly.retrieval import dense, hybrid
from parrotly.retrieval.sparse import build_sparse_retriever


@pytest.fixture
def search_components():
    """Create matching FAISS and BM25 indexes."""
    chunks = [
        Document(page_content="BM25 keyword search"),
        Document(page_content="FAISS semantic retrieval"),
        Document(page_content="Python unit testing"),
    ]

    index = faiss.IndexFlatIP(3)
    index.add(np.eye(3, dtype=np.float32))

    retriever = build_sparse_retriever(chunks, top_k=3)

    return index, retriever


def mock_rankings(monkeypatch, dense_results, sparse_results):
    """Replace retrieval functions with deterministic rankings."""
    monkeypatch.setattr(
        hybrid,
        "dense_search",
        lambda query, index, top_k: dense_results,
    )
    monkeypatch.setattr(
        hybrid,
        "sparse_search",
        lambda query, retriever: sparse_results,
    )


def test_sparse_documents_have_matching_ids(search_components):
    _, retriever = search_components

    assert [document.id for document in retriever.docs] == [
        "0", "1", "2"
    ]


def test_hybrid_search_fuses_rankings(search_components, monkeypatch):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[(0, 0.9), (1, 0.8)],
        sparse_results=[retriever.docs[1], retriever.docs[2]],
    )

    results = hybrid.hybrid_search("test query", index, retriever)

    assert [chunk_id for chunk_id, _ in results] == [1, 0, 2]
    assert results[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert results[1][1] == pytest.approx(1 / 61)
    assert results[2][1] == pytest.approx(1 / 62)


def test_hybrid_search_deduplicates_chunks(search_components, monkeypatch):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[(0, 0.9)],
        sparse_results=[retriever.docs[0]],
    )

    results = hybrid.hybrid_search("test query", index, retriever)

    assert len(results) == 1
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(2 / 61)


def test_hybrid_search_respects_top_k(search_components, monkeypatch):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[(0, 0.9), (1, 0.8), (2, 0.7)],
        sparse_results=[],
    )

    results = hybrid.hybrid_search(
        "test query", index, retriever, top_k=2
    )

    assert len(results) == 2
    assert [chunk_id for chunk_id, _ in results] == [0, 1]


def test_hybrid_search_with_dense_results_only(
    search_components, monkeypatch
):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[(1, 0.9)],
        sparse_results=[],
    )

    results = hybrid.hybrid_search("test query", index, retriever)

    assert results == [(1, pytest.approx(1 / 61))]


def test_hybrid_search_with_sparse_results_only(
    search_components, monkeypatch
):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[],
        sparse_results=[retriever.docs[2]],
    )

    results = hybrid.hybrid_search("test query", index, retriever)

    assert results == [(2, pytest.approx(1 / 61))]


def test_hybrid_search_returns_empty_when_no_matches(
    search_components, monkeypatch
):
    index, retriever = search_components

    mock_rankings(monkeypatch, [], [])

    assert hybrid.hybrid_search("test query", index, retriever) == []


def test_hybrid_search_breaks_ties_by_chunk_index(
    search_components, monkeypatch
):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[(1, 0.9)],
        sparse_results=[retriever.docs[0]],
    )

    results = hybrid.hybrid_search("test query", index, retriever)

    assert [chunk_id for chunk_id, _ in results] == [0, 1]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_hybrid_search_rejects_empty_query(query, search_components):
    index, retriever = search_components

    with pytest.raises(ValueError, match="Query cannot be empty"):
        hybrid.hybrid_search(query, index, retriever)


@pytest.mark.parametrize(
    "parameters",
    [
        {"top_k": 0},
        {"dense_k": -1},
        {"rrf_k": 0},
    ],
)
def test_hybrid_search_rejects_invalid_parameters(
    parameters, search_components
):
    index, retriever = search_components

    with pytest.raises(ValueError, match="greater than zero"):
        hybrid.hybrid_search(
            "test query", index, retriever, **parameters
        )


def test_hybrid_search_rejects_index_size_mismatch(search_components):
    _, retriever = search_components

    index = faiss.IndexFlatIP(3)
    index.add(np.eye(2, 3, dtype=np.float32))

    with pytest.raises(ValueError, match="same chunks"):
        hybrid.hybrid_search("test query", index, retriever)


@pytest.mark.parametrize("document_id", [None, "invalid", "99"])
def test_hybrid_search_rejects_invalid_document_ids(
    document_id, search_components, monkeypatch
):
    index, retriever = search_components

    mock_rankings(
        monkeypatch,
        dense_results=[],
        sparse_results=[
            Document(page_content="Invalid document", id=document_id)
        ],
    )

    with pytest.raises(ValueError, match="ID|indices"):
        hybrid.hybrid_search("test query", index, retriever)


def test_hybrid_search_with_real_bm25(search_components, monkeypatch):
    """Combine real FAISS and BM25 without calling Ollama."""
    index, retriever = search_components

    monkeypatch.setattr(
        dense,
        "embed_text",
        lambda query: [1.0, 0.0, 0.0],
    )

    results = hybrid.hybrid_search("BM25", index, retriever)

    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(2 / 61)
