"""Load supported TXT and PDF files as LangChain documents for indexing."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_txt(file_path: Path) -> list[Document]:
    """Load a UTF-8 text file as LangChain documents."""
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    if not any(document.page_content.strip() for document in documents):
        raise ValueError(f"TXT file contains no text: {file_path}")

    return documents


def load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF file as page-level LangChain documents."""
    loader = PyPDFLoader(str(file_path))
    documents = [document for document in loader.load() if document.page_content.strip()]

    if not documents:
        raise ValueError(f"PDF file contains no extractable text: {file_path}")

    return documents


def load_documents(data_dir: Path) -> list[Document]:
    """Load supported TXT and PDF files from a directory."""
    loaded_documents = []

    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            documents = load_txt(file_path)
        elif suffix == ".pdf":
            documents = load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        loaded_documents.extend(documents)

    if not loaded_documents:
        raise ValueError(f"No documents found in directory: {data_dir}")

    return loaded_documents