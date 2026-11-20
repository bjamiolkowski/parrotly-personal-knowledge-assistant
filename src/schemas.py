"""
Data models for the Modular RAG system.

This module defines structured representations for:
- document chunks
- retrieval results
"""


from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


@dataclass
class RetrievedResult:
    source: str
    chunk_id: int
    text: str
    vector_score: float
    tfidf_score: float
    hybrid_score: float
    overlap_score: float = 0.0
    rerank_score: float = 0.0
    rank: Optional[int] = None