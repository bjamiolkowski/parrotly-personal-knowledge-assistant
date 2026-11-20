"""Unit tests for cross-encoder document reranking."""

import pytest
from langchain_core.documents import Document

from parrotly.post_retrieval import reranking


class FakeCrossEncoder:
    """Provide deterministic relevance scores without loading a model."""

    def __init__(self):
        self.calls = []
        self.scores = {
            "Low relevance": 0.1,
            "High relevance": 0.9,
            "Medium relevance": 0.5,
        }

    def predict(self, pairs):
        self.calls.append(pairs)
        return [self.scores[text] for _, text in pairs]


@pytest.fixture
def documents():
    return [
        Document(
            page_content="Low relevance",
            metadata={"source": "low.txt"},
        ),
        Document(
            page_content="High relevance",
            metadata={"source": "high.txt"},
        ),
        Document(
            page_content="Medium relevance",
            metadata={"source": "medium.txt"},
        ),
    ]


@pytest.fixture
def reranker():
    return FakeCrossEncoder()


def test_rerank_documents_orders_by_relevance(documents, reranker):
    results = reranking.rerank_documents(
        "test query", documents, reranker, top_k=3
    )

    assert [doc.page_content for doc in results] == [
        "High relevance",
        "Medium relevance",
        "Low relevance",
    ]


def test_rerank_documents_preserves_metadata(documents, reranker):
    results = reranking.rerank_documents(
        "test query", documents, reranker
    )

    assert results[0] is documents[1]
    assert results[0].metadata["source"] == "high.txt"


def test_rerank_documents_passes_query_document_pairs(documents, reranker):
    reranking.rerank_documents("test query", documents, reranker)

    assert reranker.calls == [
        [
            ("test query", "Low relevance"),
            ("test query", "High relevance"),
            ("test query", "Medium relevance"),
        ]
    ]


def test_rerank_documents_respects_top_k(documents, reranker):
    results = reranking.rerank_documents(
        "test query", documents, reranker, top_k=1
    )

    assert len(results) == 1
    assert results[0].page_content == "High relevance"


def test_rerank_documents_returns_empty_for_no_documents(reranker):
    results = reranking.rerank_documents(
        "test query", [], reranker
    )

    assert results == []
    assert reranker.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_rerank_documents_rejects_empty_query(
    query, documents, reranker
):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranking.rerank_documents(query, documents, reranker)


@pytest.mark.parametrize("top_k", [0, -1])
def test_rerank_documents_rejects_invalid_top_k(
    top_k, documents, reranker
):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        reranking.rerank_documents(
            "test query", documents, reranker, top_k=top_k
        )


def test_rerank_documents_rejects_invalid_score_count(
    documents, reranker, monkeypatch
):
    monkeypatch.setattr(reranker, "predict", lambda pairs: [0.9])

    with pytest.raises(ValueError, match="unexpected number of scores"):
        reranking.rerank_documents("test query", documents, reranker)


def test_build_reranker(monkeypatch):
    fake_model = FakeCrossEncoder()
    model_names = []

    def fake_model_loader(model_name):
        model_names.append(model_name)
        return fake_model

    monkeypatch.setattr(
        reranking,
        "CrossEncoder",
        fake_model_loader,
    )

    result = reranking.build_reranker(model_name="test-model")

    assert result is fake_model
    assert model_names == ["test-model"]


@pytest.mark.parametrize("model_name", ["", "   "])
def test_build_reranker_rejects_empty_model_name(model_name):
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        reranking.build_reranker(model_name=model_name)