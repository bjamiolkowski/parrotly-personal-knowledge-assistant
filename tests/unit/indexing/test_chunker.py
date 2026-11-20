"""Tests for document chunking."""

from langchain_core.documents import Document

from parrotly.indexing.chunker import chunk_documents


def test_short_document() -> None:
    document = Document(page_content="Short document.")

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short document."


def test_long_document() -> None:
    document = Document(page_content="Sample text. " * 200)

    chunks = chunk_documents([document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_preserves_metadata() -> None:
    document = Document(
        page_content="Sample content.",
        metadata={"source": "example.pdf", "page": 0},
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].metadata == document.metadata


def test_empty_documents() -> None:
    chunks = chunk_documents([])

    assert chunks == []
