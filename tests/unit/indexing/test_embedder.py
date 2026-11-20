"""Unit tests for embedding generation and L2 normalization."""

from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.documents import Document

from parrotly.indexing import embedder


def test_embed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = Mock()
    mock_model.embed_query.return_value = [3.0, 4.0]

    monkeypatch.setattr(embedder, "get_embedding_model", lambda: mock_model)

    embedding = embedder.embed_text("  Sample query  ")

    mock_model.embed_query.assert_called_once_with("Sample query")
    mock_model.embed_documents.assert_not_called()

    assert embedding.shape == (1, 2)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(embedding, [[0.6, 0.8]], atol=1e-6)


def test_build_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        Document(page_content="First chunk"),
        Document(page_content="Second chunk"),
    ]

    mock_model = Mock()
    mock_model.embed_documents.return_value = [
        [3.0, 4.0],
        [0.0, 5.0],
    ]

    monkeypatch.setattr(embedder, "get_embedding_model", lambda: mock_model)

    embeddings = embedder.build_embeddings(chunks)

    mock_model.embed_documents.assert_called_once_with(
        ["First chunk", "Second chunk"]
    )
    mock_model.embed_query.assert_not_called()

    assert embeddings.shape == (2, 2)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(
        embeddings,
        [[0.6, 0.8], [0.0, 1.0]],
        atol=1e-6,
    )


def test_embed_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty text"):
        embedder.embed_text("   ")


def test_build_embeddings_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="empty chunk list"):
        embedder.build_embeddings([])


def test_build_embeddings_rejects_blank_chunk() -> None:
    chunks = [
        Document(page_content="Valid content"),
        Document(page_content="   "),
    ]

    with pytest.raises(ValueError, match="non-empty text"):
        embedder.build_embeddings(chunks)


def test_build_embeddings_rejects_mismatched_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        Document(page_content="First chunk"),
        Document(page_content="Second chunk"),
    ]

    mock_model = Mock()
    mock_model.embed_documents.return_value = [[1.0, 2.0]]

    monkeypatch.setattr(embedder, "get_embedding_model", lambda: mock_model)

    with pytest.raises(ValueError, match="Embedding count"):
        embedder.build_embeddings(chunks)


@pytest.mark.parametrize(
    "vectors",
    [
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
    ],
)
def test_normalize_embeddings_rejects_invalid_vectors(
    vectors: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        embedder.normalize_embeddings(vectors)


def test_normalize_embeddings_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="invalid matrix"):
        embedder.normalize_embeddings([])