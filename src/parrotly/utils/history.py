"""
Conversation history utility.

Formats recent chat messages into a string for LLM context.
"""

from src.config import MAX_HISTORY_TURNS


def build_history(
    messages: list[dict],
    max_turns: int = MAX_HISTORY_TURNS,
) -> str:
    """
    Format recent conversation history for the LLM.
    """
    if not messages:
        return ""

    recent = messages[-max_turns:]

    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)