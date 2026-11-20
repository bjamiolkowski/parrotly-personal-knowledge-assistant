
"""Tests for document loading and file format handling."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from parrotly.indexing import loader


def test_load_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "document.pdf"

    expected_documents = [Document(page_content="PDF content", metadata={"page": 0})]

    mock_pdf_loader = Mock()
    mock_pdf_loader.load.return_value = expected_documents
    mock_loader_class = Mock(return_value=mock_pdf_loader)

    monkeypatch.setattr(loader, "PyPDFLoader", mock_loader_class)

    documents = loader.load_pdf(file_path)

    mock_loader_class.assert_called_once_with(str(file_path))
    assert documents == expected_documents


def test_load_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.txt").write_text("TXT content", encoding="utf-8")
    (tmp_path / "b.pdf").touch()

    monkeypatch.setattr(
        loader,
        "load_pdf",
        lambda _: [Document(page_content="PDF content")],
    )

    documents = loader.load_documents(tmp_path)

    assert [doc.page_content for doc in documents] == [
        "TXT content",
        "PDF content",
    ]


def test_unsupported_file_format(tmp_path: Path) -> None:
    (tmp_path / "document.csv").write_text("Some data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_documents(tmp_path)


def test_loader_error_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "document.txt").write_text("Some text", encoding="utf-8")

    def failing_loader(file_path: Path) -> list[Document]:
        raise OSError("Cannot read document")

    monkeypatch.setattr(loader, "load_txt", failing_loader)

    with pytest.raises(OSError, match="Cannot read document"):
        loader.load_documents(tmp_path)











