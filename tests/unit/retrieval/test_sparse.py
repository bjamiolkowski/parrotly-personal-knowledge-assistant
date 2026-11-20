"""Unit tests for BM25 sparse retrieval."""

import pytest

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from parrotly.retrieval.sparse import (
    build_sparse_retriever,
    sparse_search,
    tokenize,
)


@pytest.fixture
def chunks() -> list[Document]:
    """Create sample documents for sparse retrieval tests."""
    return [
        Document(
            page_content="BM25 keyword lexical search",
            metadata={"source": "bm25.txt"},
        ),
        Document(
            page_content="FAISS vector semantic retrieval",
            metadata={"source": "faiss.txt"},
        ),
        Document(
            page_content="Python programming and unit tests",
            metadata={"source": "python.txt"},
        ),
        Document(
            page_content="Ollama language model inference",
            metadata={"source": "ollama.txt"},
        ),
    ]


def test_tokenize():
    result = tokenize("Hello, WORLD! BM25-based retrieval.")

    assert result == [
        "hello",
        "world",
        "bm25",
        "based",
        "retrieval",
    ]


def test_build_sparse_retriever(chunks):
    retriever = build_sparse_retriever(chunks, top_k=2)

    assert isinstance(retriever, BM25Retriever)
    assert retriever.k == 2
    assert len(retriever.docs) == len(chunks)


def test_sparse_search_returns_relevant_document(chunks):
    retriever = build_sparse_retriever(chunks, top_k=2)

    results = sparse_search("BM25 keyword", retriever)

    assert results
    assert results[0].page_content == chunks[0].page_content


def test_sparse_search_preserves_metadata(chunks):
    retriever = build_sparse_retriever(chunks, top_k=2)

    results = sparse_search("BM25 keyword", retriever)

    assert results[0].metadata["source"] == "bm25.txt"


def test_sparse_search_respects_top_k(chunks):
    retriever = build_sparse_retriever(chunks, top_k=1)

    results = sparse_search("semantic retrieval", retriever)

    assert len(results) <= 1


def test_sparse_search_returns_empty_for_no_matches(chunks):
    retriever = build_sparse_retriever(chunks)

    results = sparse_search("quantum astrophysics", retriever)

    assert results == []


def test_sparse_search_returns_empty_for_no_tokens(chunks):
    retriever = build_sparse_retriever(chunks)

    results = sparse_search("!!!", retriever)

    assert results == []


def test_build_sparse_retriever_rejects_empty_chunks():
    with pytest.raises(ValueError, match="empty chunks"):
        build_sparse_retriever([])


@pytest.mark.parametrize("top_k", [0, -1])
def test_build_sparse_retriever_rejects_invalid_top_k(
    chunks, top_k
):
    with pytest.raises(ValueError, match="top_k"):
        build_sparse_retriever(chunks, top_k=top_k)


def test_build_sparse_retriever_rejects_empty_document():
    chunks = [Document(page_content="   ")]

    with pytest.raises(ValueError, match="searchable text"):
        build_sparse_retriever(chunks)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_sparse_search_rejects_empty_query(chunks, query):
    retriever = build_sparse_retriever(chunks)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        sparse_search(query, retriever)