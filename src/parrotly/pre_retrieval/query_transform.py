"""
Pre-retrieval query processing module.

Normalizes and optionally corrects user queries before retrieval.
"""

import re
from difflib import get_close_matches

from wordfreq import top_n_list


COMMON_ENGLISH_WORDS = set(top_n_list("en", 50000))

MIN_CORRECTION_LENGTH = 4
DEFAULT_CUTOFF = 0.75


def build_vocabulary(chunks: list[dict]) -> set[str]:
    """Build vocabulary from indexed document chunks."""
    vocab = set()

    for chunk in chunks:
        words = re.findall(r"\w+", chunk["text"].lower())
        vocab.update(words)

    return vocab


def correct_word(
    word: str,
    vocab: set[str],
    cutoff: float = DEFAULT_CUTOFF,
) -> str:
    """Correct a query word using English and document vocabulary."""
    normalized = word.lower()

    if normalized in COMMON_ENGLISH_WORDS:
        return normalized

    if normalized in vocab:
        return normalized

    if len(normalized) < MIN_CORRECTION_LENGTH:
        return normalized

    correction_vocab = COMMON_ENGLISH_WORDS | vocab

    candidates = [
        candidate
        for candidate in correction_vocab
        if abs(len(candidate) - len(normalized)) <= 3
    ]

    matches = get_close_matches(
        normalized,
        candidates,
        n=1,
        cutoff=cutoff,
    )

    return matches[0] if matches else normalized


def correct_query(query: str, vocab: set[str]) -> str:
    """Correct query typos."""
    words = re.findall(r"\w+", query.lower())

    corrected_words = [
        correct_word(word, vocab)
        for word in words
    ]

    return " ".join(corrected_words)


def rewrite_query(
    query: str,
    vocab: set[str] | None = None,
) -> tuple[str, str]:
    """Return original query and corrected query."""
    original = query.strip()

    if not original:
        return original, original

    corrected = correct_query(
        original,
        vocab or set(),
    )

    return original, corrected


def expand_query(query: str) -> list[str]:
    """Return query variants for retrieval."""
    query = query.strip()

    return [query] if query else []