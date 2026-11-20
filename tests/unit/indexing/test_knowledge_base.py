"""Unit tests for FAISS index construction and knowledge base rebuilding."""

from unittest.mock import Mock

import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from parrotly.indexing import knowledge_base


def test_build_faiss_index() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    index = knowledge_base.build_faiss_index(embeddings)

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.d == 2
    assert index.ntotal == 2

    scores, indices = index.search(embeddings[:1], 1)

    assert indices[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "embeddings",
    [
        np.empty((0, 2), dtype=np.float32),
        np.empty((2, 0), dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
    ],
)
def test_build_faiss_index_rejects_invalid_shape(
    embeddings: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="non-empty 2D array"):
        knowledge_base.build_faiss_index(embeddings)


def test_build_faiss_index_rejects_invalid_dtype() -> None:
    embeddings = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(TypeError, match="float32"):
        knowledge_base.build_faiss_index(embeddings)


def test_rebuild_knowledge_base_rejects_empty_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        knowledge_base, "load_documents", lambda _: []
    )

    with pytest.raises(ValueError, match="No documents found"):
        knowledge_base.rebuild_knowledge_base()


def test_rebuild_knowledge_base_rejects_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [Document(page_content="Example document")]

    monkeypatch.setattr(
        knowledge_base, "load_documents", lambda _: documents
    )
    monkeypatch.setattr(
        knowledge_base, "chunk_documents", lambda _: []
    )

    with pytest.raises(ValueError, match="No chunks were created"):
        knowledge_base.rebuild_knowledge_base()
