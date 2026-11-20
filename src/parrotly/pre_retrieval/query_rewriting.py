"""
Rewrite user queries to improve retrieval quality.

Supports query clarification, typo correction, and contextual rewriting
using conversation history while preserving the original intent.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



QUERY_REWRITING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query rewriting assistant for a document retrieval system.

            Your task is to rewrite user queries to improve retrieval quality.

            Rules:
            - Improve clarity and correct obvious typos.
            - Use conversation history to resolve missing context when needed.
            - Preserve the original intent, names, dates, numbers, and technical terms.
            - Do not invent information or introduce unsupported assumptions.
            - Keep queries unchanged if they are already clear and self-contained.
            - Return only the rewritten query without explanations.""",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{query}"),
    ]
)


def rewrite_query(
    query: str,
    llm: BaseChatModel,
    chat_history: list[BaseMessage] | None = None,
) -> str:
    """Rewrite a query for retrieval using optional conversation history."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    chain = QUERY_REWRITING_PROMPT | llm | StrOutputParser()

    rewritten_query = chain.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
        }
    ).strip()

    if not rewritten_query:
        raise ValueError("Query rewriting returned an empty response.")

    return rewritten_query
