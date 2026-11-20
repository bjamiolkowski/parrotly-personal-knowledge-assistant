"""Integration tests for knowledge base persistence."""

from pathlib import Path

import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from parrotly.indexing import knowledge_base


def test_rebuild_and_load_knowledge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    storage_dir = tmp_path / "storage"

    data_dir.mkdir()
    (data_dir / "document.txt").write_text(
        "Retrieval-augmented generation combines retrieval and generation.",
        encoding="utf-8",
    )

    index_path = storage_dir / "index.faiss"
    chunks_path = storage_dir / "chunks.pkl"

    monkeypatch.setattr(knowledge_base, "DATA_DIR", data_dir)
    monkeypatch.setattr(knowledge_base, "STORAGE_DIR", storage_dir)
    monkeypatch.setattr(knowledge_base, "INDEX_PATH", index_path)
    monkeypatch.setattr(knowledge_base, "CHUNKS_PATH", chunks_path)

    def mock_embeddings(chunks: list[Document]) -> np.ndarray:
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(chunks), 1),
        )

    monkeypatch.setattr(
        knowledge_base, "build_embeddings", mock_embeddings
    )

    knowledge_base.rebuild_knowledge_base()

    assert index_path.is_file()
    assert chunks_path.is_file()

    from parrotly.utils import io

    monkeypatch.setattr(io, "INDEX_PATH", index_path)
    monkeypatch.setattr(io, "CHUNKS_PATH", chunks_path)

    index = io.load_index()
    chunks = io.load_chunks()

    assert index.ntotal == len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert "Retrieval-augmented generation" in chunks[0].page_content