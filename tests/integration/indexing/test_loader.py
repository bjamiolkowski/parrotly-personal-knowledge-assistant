
from pathlib import Path

from langchain_core.documents import Document

import pytest

from parrotly.indexing import loader


def test_load_txt(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("Random text for testing.", encoding="utf-8")

    documents = loader.load_txt(file_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Random text for testing."


def test_load_empty_txt(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="TXT file contains no text"):
        loader.load_txt(file_path)