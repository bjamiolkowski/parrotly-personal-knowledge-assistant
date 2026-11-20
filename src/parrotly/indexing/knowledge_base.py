"""Build and persist a FAISS-backed knowledge base from source documents."""

import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np

from src.config import CHUNKS_PATH, DATA_DIR, INDEX_PATH, STORAGE_DIR
from parrotly.indexing.chunker import chunk_documents
from parrotly.indexing.embedder import build_embeddings
from parrotly.indexing.loader import load_documents


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from dense embeddings."""
    if embeddings.ndim != 2 or 0 in embeddings.shape:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    if embeddings.dtype != np.float32:
        raise TypeError("Embeddings must have dtype float32.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))

    return index


def rebuild_knowledge_base() -> None:
    """Rebuild the FAISS index and persist its associated document chunks."""
    documents = load_documents(DATA_DIR)
    if not documents:
        raise ValueError("No documents found in the data directory.")

    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were created from the documents.")

    embeddings = build_embeddings(chunks)
    index = build_faiss_index(embeddings)

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunk counts do not match.")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=STORAGE_DIR) as temporary_dir:
        temporary_path = Path(temporary_dir)

        temporary_index = temporary_path / INDEX_PATH.name
        temporary_chunks = temporary_path / CHUNKS_PATH.name

        faiss.write_index(index, str(temporary_index))

        with temporary_chunks.open("wb") as file:
            pickle.dump(chunks, file, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temporary_index, INDEX_PATH)
        os.replace(temporary_chunks, CHUNKS_PATH)


if __name__ == "__main__":
    print("Building knowledge base...")
    rebuild_knowledge_base()
    print("Knowledge base rebuilt successfully.")