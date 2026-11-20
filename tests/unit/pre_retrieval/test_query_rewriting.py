
"""Unit tests for query rewriting."""

import pytest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from parrotly.pre_retrieval.query_rewriting import (
    QUERY_REWRITING_PROMPT,
    rewrite_query,
)


def test_rewrite_query_returns_model_response():
    llm = FakeListChatModel(
        responses=["  What is Apple's revenue in 2023?  "]
    )

    result = rewrite_query(
        query="apple revenue 2023",
        llm=llm,
    )

    assert result == "What is Apple's revenue in 2023?"


def test_rewrite_query_with_chat_history():
    llm = FakeListChatModel(
        responses=["What was Apple's revenue in 2024?"]
    )
    history = [
        HumanMessage(content="What was Apple's revenue in 2023?"),
        AIMessage(content="Apple reported revenue of ..."),
    ]

    result = rewrite_query(
        query="And in 2024?",
        llm=llm,
        chat_history=history,
    )

    assert result == "What was Apple's revenue in 2024?"


def test_prompt_includes_chat_history():
    history = [
        HumanMessage(content="What was Apple's revenue in 2023?"),
        AIMessage(content="Apple reported revenue of ..."),
    ]

    messages = QUERY_REWRITING_PROMPT.format_messages(
        query="And in 2024?",
        chat_history=history,
    )

    assert messages[1:3] == history
    assert messages[-1].content == "And in 2024?"


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_rewrite_query_rejects_empty_query(query):
    llm = FakeListChatModel(responses=["Some response"])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        rewrite_query(query=query, llm=llm)


def test_rewrite_query_rejects_empty_model_response():
    llm = FakeListChatModel(responses=["   "])

    with pytest.raises(
        ValueError,
        match="Query rewriting returned an empty response",
    ):
        rewrite_query(query="What is EBITDA?", llm=llm)
