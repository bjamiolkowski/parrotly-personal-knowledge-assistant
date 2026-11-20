"""
Reranking module.

Refines retrieval results by combining hybrid relevance with keyword overlap.
"""

import re


def keyword_overlap_score(query: str, text: str) -> float:
    """
    Compute the fraction of query terms found in the document text.
    """
    query_words = [word for word in re.findall(r"\w+", query.lower()) if len(word) > 2]

    if not query_words:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for word in query_words if word in text_lower)

    return matches / len(query_words)


def rerank_results(results: list[dict], query: str) -> list[dict]:
    """
    Rerank retrieval results using hybrid score and keyword overlap.
    """
    reranked = []

    for result in results:
        overlap_score = keyword_overlap_score(query, result["text"])
        rerank_score = 0.8 * result["hybrid_score"] + 0.2 * overlap_score

        item = result.copy()
        item["overlap_score"] = overlap_score
        item["rerank_score"] = rerank_score

        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank

    return reranked