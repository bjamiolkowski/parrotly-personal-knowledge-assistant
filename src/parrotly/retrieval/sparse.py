"""
Sparse retrieval using BM25.

Provides keyword-based document retrieval to complement semantic search.
"""

import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and retrieval."""
    return re.findall(r"\b\w+\b", text.casefold())


def build_sparse_retriever(
    chunks: list[Document],
    top_k: int = 20,
) -> BM25Retriever:
    """Build a BM25 retriever from document chunks."""
    if not chunks:
        raise ValueError("Cannot build a BM25 retriever from empty chunks.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if any(not tokenize(chunk.page_content) for chunk in chunks):
        raise ValueError("Document chunks must contain searchable text.")

    return BM25Retriever.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[str(i) for i in range(len(chunks))],
        preprocess_func=tokenize,
        k=top_k,
    )


def sparse_search(
    query: str,
    retriever: BM25Retriever,
) -> list[Document]:
    """Retrieve documents ranked by BM25."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_terms = set(tokenize(query))

    if not query_terms:
        return []

    results = retriever.invoke(query)

    return [
        document
        for document in results
        if not query_terms.isdisjoint(tokenize(document.page_content))
    ]
