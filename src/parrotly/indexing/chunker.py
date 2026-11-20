"""Split loaded documents into retrieval-ready chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)