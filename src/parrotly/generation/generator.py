"""LLM generation helpers."""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.parrotly.chains.qa_chain import build_qa_chain
from src.parrotly.chains.summary_chain import build_summary_chain
from src.config import DEFAULT_PROVIDER, GEN_MODEL, OLLAMA_BASE_URL


PRICING = {
    "gpt-4.1-mini": {
        "input": 0.40 / 1_000_000,
        "output": 1.60 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate OpenAI request cost."""
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0

    return (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
    )


def _get_llm(provider: str, model: str):
    """Create LangChain chat model for the selected provider."""
    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")

        return ChatOpenAI(
            model=model,
            temperature=0,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _extract_usage(response) -> tuple[int, int]:
    """Extract token usage from LangChain chat model response."""
    usage = getattr(response, "usage_metadata", None)

    if not usage:
        return 0, 0

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return input_tokens, output_tokens


def _run_generation_chain(
    chain,
    values: dict,
    model: str,
    provider: str,
) -> dict:
    """Run a LangChain generation chain and return UI-compatible metadata."""
    response = chain.invoke(values)

    input_tokens, output_tokens = _extract_usage(response)
    answer = StrOutputParser().invoke(response).strip()

    return {
        "answer": answer,
        "input_tokens": input_tokens if provider == "openai" else None,
        "output_tokens": output_tokens if provider == "openai" else None,
        "cost_usd": (
            _calculate_cost(model, input_tokens, output_tokens)
            if provider == "openai"
            else 0.0
        ),
    }


def generate_answer(
    query: str,
    context: str,
    history: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = GEN_MODEL,
) -> dict:
    """Generate answer from retrieved context."""
    llm = _get_llm(provider, model)
    chain = build_qa_chain(llm)

    return _run_generation_chain(
        chain,
        {
            "query": query,
            "context": context,
            "history": history,
        },
        model,
        provider,
    )


def generate_summary(
    topic: str,
    context: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = GEN_MODEL,
) -> dict:
    """Generate summary from retrieved context."""
    llm = _get_llm(provider, model)
    chain = build_summary_chain(llm)

    return _run_generation_chain(
        chain,
        {
            "topic": topic,
            "context": context,
        },
        model,
        provider,
    )


def stream_answer(
    query: str,
    context: str,
    history: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = GEN_MODEL,
):
    """Stream answer tokens from retrieved context."""
    llm = _get_llm(provider, model)
    chain = build_qa_chain(llm)

    for chunk in chain.stream(
        {
            "query": query,
            "context": context,
            "history": history,
        }
    ):
        yield StrOutputParser().invoke(chunk)