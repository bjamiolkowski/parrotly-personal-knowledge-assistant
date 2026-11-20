"""
Build source-aware context from retrieved documents.

Formats document content and metadata for grounded LLM generation.
"""

from langchain_core.documents import Document


def build_context(
    documents: list[Document],
    max_chars: int = 12000,
) -> str:
    """Build a source-aware context within a character limit."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    blocks = []
    total_chars = 0

    for document in documents:
        content = document.page_content.strip()

        if not content:
            raise ValueError("Documents must contain non-empty text.")

        metadata = document.metadata
        header = [f"[{len(blocks) + 1}]"]

        if source := metadata.get("source"):
            header.append(f"Source: {source}")

        if metadata.get("page_label") is not None:
            header.append(f"Page: {metadata['page_label']}")
        elif isinstance(metadata.get("page"), int):
            header.append(f"Page: {metadata['page'] + 1}")

        block = f"{' | '.join(header)}\n{content}"

        required_chars = len(block) + (2 if blocks else 0)

        if total_chars + required_chars > max_chars:
            continue

        blocks.append(block)
        total_chars += required_chars

    return "\n\n".join(blocks)