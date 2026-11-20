"""
Post-retrieval filtering and context construction.

Validates retrieval quality and builds grounded context for LLM generation.
"""

from src.config import (
    MAX_CONTEXT_CHARS,
    MIN_CONTEXT_SCORE,
    MIN_HYBRID_SCORE,
    MIN_RESULTS,
)


def is_context_sufficient(
    results: list[dict],
    min_results: int = MIN_RESULTS,
    min_hybrid_score: float = MIN_HYBRID_SCORE,
) -> bool:
    """
    Check whether retrieved results are strong enough for generation.
    """
    if not results or len(results) < min_results:
        return False

    strong_results = [
        result
        for result in results
        if result.get("hybrid_score", 0.0) >= min_hybrid_score
    ]

    return len(strong_results) >= min_results


def filter_results(
    results: list[dict],
    min_score: float = MIN_CONTEXT_SCORE,
) -> list[dict]:
    """
    Keep high-confidence results, with a small fallback if filtering removes all.
    """
    filtered = [
        result
        for result in results
        if result.get("hybrid_score", 0.0) >= min_score
    ]

    return filtered if filtered else results[:2]


def build_grounded_context(
    results: list[dict],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Build source-aware context for LLM generation.
    """
    filtered = filter_results(results)
    context_blocks = []
    total_chars = 0

    for result in filtered:
        block = (
            f"[Source: {result['source']} | chunk: {result['chunk_id']} | "
            f"score: {result['hybrid_score']:.3f}]\n"
            f"{result['text']}\n"
        )

        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)

    return "\n".join(context_blocks).strip()