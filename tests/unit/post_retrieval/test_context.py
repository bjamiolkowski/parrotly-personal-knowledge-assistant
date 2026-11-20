"""Unit tests for source-aware context construction."""

import pytest
from langchain_core.documents import Document

from parrotly.post_retrieval.context import build_context


@pytest.fixture
def documents() -> list[Document]:
    """Create sample documents with source metadata."""
    return [
        Document(
            page_content="Revenue increased in 2023.",
            metadata={"source": "report.pdf", "page": 2},
        ),
        Document(
            page_content="Operating costs decreased.",
            metadata={"source": "notes.txt"},
        ),
    ]


def test_build_context_formats_documents(documents):
    result = build_context(documents)

    expected = (
        "[1] | Source: report.pdf | Page: 3\n"
        "Revenue increased in 2023.\n\n"
        "[2] | Source: notes.txt\n"
        "Operating costs decreased."
    )

    assert result == expected


def test_build_context_preserves_document_order(documents):
    result = build_context(documents)

    assert result.index("Revenue increased") < result.index(
        "Operating costs decreased"
    )


def test_build_context_uses_page_label():
    documents = [
        Document(
            page_content="Sample content",
            metadata={"page": 3, "page_label": "iv"},
        )
    ]

    result = build_context(documents)

    assert result == "[1] | Page: iv\nSample content"


def test_build_context_converts_zero_based_page():
    documents = [
        Document(
            page_content="First page",
            metadata={"page": 0},
        )
    ]

    result = build_context(documents)

    assert result == "[1] | Page: 1\nFirst page"


def test_build_context_without_metadata():
    documents = [Document(page_content="Sample content")]

    result = build_context(documents)

    assert result == "[1]\nSample content"


def test_build_context_strips_whitespace():
    documents = [Document(page_content=" \n Sample content \n ")]

    result = build_context(documents)

    assert result == "[1]\nSample content"


def test_build_context_respects_exact_character_limit():
    documents = [Document(page_content="Hello")]
    expected = "[1]\nHello"

    result = build_context(documents, max_chars=len(expected))

    assert result == expected
    assert len(result) == len(expected)


def test_build_context_counts_separators():
    documents = [
        Document(page_content="First"),
        Document(page_content="Second"),
    ]

    expected = "[1]\nFirst\n\n[2]\nSecond"

    full_result = build_context(
        documents,
        max_chars=len(expected),
    )
    limited_result = build_context(
        documents,
        max_chars=len(expected) - 1,
    )

    assert full_result == expected
    assert limited_result == "[1]\nFirst"


def test_build_context_skips_oversized_documents():
    documents = [
        Document(page_content="A" * 100),
        Document(page_content="Short"),
    ]

    result = build_context(documents, max_chars=20)

    assert result == "[1]\nShort"


def test_build_context_returns_empty_when_nothing_fits():
    documents = [Document(page_content="Long document content")]

    result = build_context(documents, max_chars=5)

    assert result == ""


def test_build_context_returns_empty_for_no_documents():
    assert build_context([]) == ""


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_build_context_rejects_empty_document(content):
    documents = [Document(page_content=content)]

    with pytest.raises(
        ValueError,
        match="Documents must contain non-empty text",
    ):
        build_context(documents)


@pytest.mark.parametrize("max_chars", [0, -1])
def test_build_context_rejects_invalid_max_chars(max_chars):
    with pytest.raises(
        ValueError,
        match="max_chars must be greater than zero",
    ):
        build_context([], max_chars=max_chars)