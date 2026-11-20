"""Generate L2-normalized embeddings using Ollama."""

import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import EMBED_MODEL, OLLAMA_BASE_URL


def get_embedding_model() -> OllamaEmbeddings:
    """Create the configured Ollama embedding model."""
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def normalize_embeddings(vectors: list[list[float]]) -> np.ndarray:
    """Validate and L2-normalize embedding vectors."""
    embeddings = np.asarray(vectors, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError("Embedding model returned an invalid matrix.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding vectors contain non-finite values.")

    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Embedding vectors must have non-zero norms.")

    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings)

    return embeddings


def embed_text(text: str) -> np.ndarray:
    """Generate a normalized embedding for a single query."""
    text = text.strip()

    if not text:
        raise ValueError("Cannot generate an embedding for empty text.")

    embedding_model = get_embedding_model()
    vector = embedding_model.embed_query(text)

    return normalize_embeddings([vector])


def build_embeddings(chunks: list[Document]) -> np.ndarray:
    """Generate normalized embeddings for document chunks."""
    if not chunks:
        raise ValueError("Cannot generate embeddings from an empty chunk list.")

    texts = [chunk.page_content.strip() for chunk in chunks]

    if any(not text for text in texts):
        raise ValueError("Document chunks must contain non-empty text.")

    embedding_model = get_embedding_model()
    vectors = embedding_model.embed_documents(texts)

    if len(vectors) != len(texts):
        raise ValueError("Embedding count does not match the chunk count.")

    return normalize_embeddings(vectors)