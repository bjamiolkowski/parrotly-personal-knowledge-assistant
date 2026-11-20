"""
Core orchestration module for the Modular RAG system.
"""

from __future__ import annotations

import time

from src.config import (
    DEFAULT_ALPHA,
    DEFAULT_FAISS_K,
    DEFAULT_TFIDF_K,
    DEFAULT_TOP_K,
    GENERATION_MODES,
)
from src.parrotly.generation.generator import generate_answer, generate_summary, stream_answer
from src.parrotly.observability.cost import estimate_cost_usd, estimate_tokens
from src.parrotly.observability.logger import log_query
from src.parrotly.post_retrieval.filters import (
    build_grounded_context,
    is_context_sufficient,
)
from src.parrotly.post_retrieval.reranker import rerank_results
from src.parrotly.pre_retrieval.query_transform import build_vocabulary, rewrite_query
from src.parrotly.retrieval.hybrid import retrieve_chunks


FALLBACK_ANSWER = "I could not find the answer in the documents."
FALLBACK_SUMMARY = "I could not find enough information in the documents."


class ModularRAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self, index, chunks: list[dict], vectorizer, tfidf_matrix) -> None:
        self.index = index
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.vocab = build_vocabulary(chunks)

    def _rewrite_query(self, query: str) -> tuple[str, str | None]:
        original_query, rewritten_query = rewrite_query(query, self.vocab)

        def normalize(text: str) -> str:
            return text.casefold().strip().rstrip("?.!")

        if normalize(rewritten_query) != normalize(original_query):
            corrected_query = rewritten_query
        else:
            corrected_query = None

        return rewritten_query, corrected_query

    def _get_generation_settings(self, generation_mode: str) -> dict:
        if generation_mode not in GENERATION_MODES:
            raise ValueError(
                f"Unknown generation_mode='{generation_mode}'. "
                f"Available modes: {list(GENERATION_MODES.keys())}"
            )

        return GENERATION_MODES[generation_mode]

    def _build_sources(self, results: list[dict]) -> list[dict]:
        return [
            {
                "source": result.get("source"),
                "score": result.get("score"),
                "text": result.get("text"),
            }
            for result in results
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        faiss_k: int = DEFAULT_FAISS_K,
        tfidf_k: int = DEFAULT_TFIDF_K,
        alpha: float = DEFAULT_ALPHA,
        retrieval_mode: str = "hybrid",
    ) -> list[dict]:
        """Retrieve and rerank relevant document chunks."""
        rewritten_query, _ = self._rewrite_query(query)

        retrieved = retrieve_chunks(
            query=rewritten_query,
            index=self.index,
            chunks=self.chunks,
            vectorizer=self.vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            top_k=max(top_k, faiss_k, tfidf_k),
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            mode=retrieval_mode,
        )

        reranked = rerank_results(retrieved, rewritten_query)
        return reranked[:top_k]

    def run_chat(
        self,
        query: str,
        history: list,
        top_k: int = 5,
        faiss_k: int = 20,
        tfidf_k: int = 20,
        alpha: float = 0.6,
        retrieval_mode: str = "hybrid",
        generation_mode: str = "balanced",
        llm_provider: str = "ollama",
        llm_model: str = "llama3",
    ) -> dict:
        """Run the RAG pipeline for question answering."""
        start_time = time.time()

        generation_settings = self._get_generation_settings(generation_mode)
        selected_top_k = top_k or generation_settings["top_k"]
        max_context_chars = generation_settings["max_context_chars"]

        rewritten_query, corrected_query = self._rewrite_query(query)

        results = self.retrieve(
            query=rewritten_query,
            top_k=selected_top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            retrieval_mode=retrieval_mode,
        )

        if not results:
            latency_sec = round(time.time() - start_time, 3)

            log_query(
                {
                    "query": query,
                    "retrieval_mode": retrieval_mode,
                    "generation_mode": generation_mode,
                    "llm_provider": llm_provider,
                    "model": llm_model,
                    "top_k": selected_top_k,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": latency_sec,
                    "num_sources": 0,
                    "fallback": True,
                }
            )

            return {
                "answer": FALLBACK_ANSWER,
                "results": [],
                "sources": [],
                "corrected_query": corrected_query,
                "rewritten_query": rewritten_query,
                "retrieval_mode": retrieval_mode,
                "generation_mode": generation_mode,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "latency": latency_sec,
                "tokens": {"input": 0, "output": 0},
                "cost_usd": 0.0,
            }

        context = build_grounded_context(results)
        context = context[:max_context_chars]

        generation_output = generate_answer(
            query=rewritten_query,
            context=context,
            history=history,
            provider=llm_provider,
            model=llm_model,
        )

        if isinstance(generation_output, dict):
            answer = generation_output.get("answer", "")
            input_tokens = generation_output.get("input_tokens")
            output_tokens = generation_output.get("output_tokens")
            cost_usd = generation_output.get("cost_usd")
        else:
            answer = generation_output
            input_tokens = None
            output_tokens = None
            cost_usd = None

        if input_tokens is None:
            input_tokens = estimate_tokens(
                rewritten_query + "\n" + context + "\n" + str(history or "")
            )

        if output_tokens is None:
            output_tokens = estimate_tokens(answer)

        if llm_provider == "ollama":
            cost_usd = 0.0
        else:
            cost_usd = estimate_cost_usd(
                model=llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        latency_sec = round(time.time() - start_time, 3)
        sources = self._build_sources(results)

        log_query(
            {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "generation_mode": generation_mode,
                "llm_provider": llm_provider,
                "model": llm_model,
                "top_k": selected_top_k,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "latency_sec": latency_sec,
                "num_sources": len(results),
                "fallback": False,
            }
        )

        return {
            "answer": answer,
            "results": results,
            "sources": sources,
            "corrected_query": corrected_query,
            "rewritten_query": rewritten_query,
            "retrieval_mode": retrieval_mode,
            "generation_mode": generation_mode,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "latency": latency_sec,
            "tokens": {"input": input_tokens, "output": output_tokens},
            "cost_usd": cost_usd,
        }

    def run_chat_stream(
        self,
        query: str,
        history: list,
        top_k: int = 5,
        faiss_k: int = 20,
        tfidf_k: int = 20,
        alpha: float = 0.6,
        retrieval_mode: str = "hybrid",
        generation_mode: str = "balanced",
        llm_provider: str = "ollama",
        llm_model: str = "llama3",
    ) -> dict:
        """Run the RAG pipeline and stream the generated answer."""
        start_time = time.time()

        generation_settings = self._get_generation_settings(generation_mode)
        selected_top_k = top_k or generation_settings["top_k"]
        max_context_chars = generation_settings["max_context_chars"]

        rewritten_query, corrected_query = self._rewrite_query(query)

        if corrected_query is None and rewritten_query.casefold() != query.casefold():
            corrected_query = rewritten_query

        results = self.retrieve(
            query=rewritten_query,
            top_k=selected_top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            retrieval_mode=retrieval_mode,
        )

        if not results:
            latency_sec = round(time.time() - start_time, 3)

            def fallback_stream():
                yield FALLBACK_ANSWER

            log_query(
                {
                    "query": query,
                    "retrieval_mode": retrieval_mode,
                    "generation_mode": generation_mode,
                    "llm_provider": llm_provider,
                    "model": llm_model,
                    "top_k": selected_top_k,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": latency_sec,
                    "num_sources": 0,
                    "fallback": True,
                }
            )

            return {
                "answer_stream": fallback_stream(),
                "results": [],
                "sources": [],
                "corrected_query": corrected_query,
                "rewritten_query": rewritten_query,
                "retrieval_mode": retrieval_mode,
                "generation_mode": generation_mode,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "latency": latency_sec,
                "tokens": {"input": 0, "output": 0},
                "cost_usd": 0.0,
            }

        context = build_grounded_context(results)
        context = context[:max_context_chars]

        input_tokens = estimate_tokens(
            rewritten_query + "\n" + context + "\n" + str(history or "")
        )

        stream_state = {
            "full_answer": "",
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_sec": 0.0,
        }

        def answer_stream():
            for token in stream_answer(
                query=rewritten_query,
                context=context,
                history=history,
                provider=llm_provider,
                model=llm_model,
            ):
                stream_state["full_answer"] += token
                yield token

            stream_state["output_tokens"] = estimate_tokens(
                stream_state["full_answer"]
            )

            if llm_provider == "ollama":
                stream_state["cost_usd"] = 0.0
            else:
                stream_state["cost_usd"] = estimate_cost_usd(
                    model=llm_model,
                    input_tokens=input_tokens,
                    output_tokens=stream_state["output_tokens"],
                )

            stream_state["latency_sec"] = round(time.time() - start_time, 3)

            log_query(
                {
                    "query": query,
                    "retrieval_mode": retrieval_mode,
                    "generation_mode": generation_mode,
                    "llm_provider": llm_provider,
                    "model": llm_model,
                    "top_k": selected_top_k,
                    "input_tokens": input_tokens,
                    "output_tokens": stream_state["output_tokens"],
                    "cost_usd": stream_state["cost_usd"],
                    "latency_sec": stream_state["latency_sec"],
                    "num_sources": len(results),
                    "fallback": False,
                }
            )

        latency_sec = round(time.time() - start_time, 3)
        sources = self._build_sources(results)

        return {
            "answer_stream": answer_stream(),
            "results": results,
            "sources": sources,
            "corrected_query": corrected_query,
            "rewritten_query": rewritten_query,
            "retrieval_mode": retrieval_mode,
            "generation_mode": generation_mode,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "latency": latency_sec,
            "tokens": {"input": input_tokens, "output": None},
            "cost_usd": None,
        }

    def run_summary(
        self,
        topic: str,
        top_k: int = DEFAULT_TOP_K,
        faiss_k: int = DEFAULT_FAISS_K,
        tfidf_k: int = DEFAULT_TFIDF_K,
        alpha: float = DEFAULT_ALPHA,
        retrieval_mode: str = "hybrid",
        llm_provider: str = "ollama",
        llm_model: str = "llama3",
    ) -> dict:
        """Run the RAG pipeline for topic summarization."""
        start_time = time.time()

        results = self.retrieve(
            query=topic,
            top_k=top_k,
            faiss_k=faiss_k,
            tfidf_k=tfidf_k,
            alpha=alpha,
            retrieval_mode=retrieval_mode,
        )

        if not results:
            latency_sec = round(time.time() - start_time, 3)

            log_query(
                {
                    "query": topic,
                    "retrieval_mode": retrieval_mode,
                    "generation_mode": "summary",
                    "llm_provider": llm_provider,
                    "model": llm_model,
                    "top_k": top_k,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": latency_sec,
                    "num_sources": 0,
                    "fallback": True,
                }
            )

            return {
                "summary": FALLBACK_SUMMARY,
                "results": [],
                "retrieval_mode": retrieval_mode,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "latency": latency_sec,
                "tokens": {"input": 0, "output": 0},
                "cost_usd": 0.0,
            }

        if not is_context_sufficient(results):
            latency_sec = round(time.time() - start_time, 3)

            log_query(
                {
                    "query": topic,
                    "retrieval_mode": retrieval_mode,
                    "generation_mode": "summary",
                    "llm_provider": llm_provider,
                    "model": llm_model,
                    "top_k": top_k,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": latency_sec,
                    "num_sources": len(results),
                    "fallback": True,
                }
            )

            return {
                "summary": FALLBACK_SUMMARY,
                "results": results,
                "retrieval_mode": retrieval_mode,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "latency": latency_sec,
                "tokens": {"input": 0, "output": 0},
                "cost_usd": 0.0,
            }

        context = build_grounded_context(results)

        generation_output = generate_summary(
            topic=topic,
            context=context,
            provider=llm_provider,
            model=llm_model,
        )

        if isinstance(generation_output, dict):
            summary = generation_output.get("summary") or generation_output.get(
                "answer", ""
            )
            input_tokens = generation_output.get("input_tokens")
            output_tokens = generation_output.get("output_tokens")
            cost_usd = generation_output.get("cost_usd")
        else:
            summary = generation_output
            input_tokens = None
            output_tokens = None
            cost_usd = None

        if input_tokens is None:
            input_tokens = estimate_tokens(topic + "\n" + context)

        if output_tokens is None:
            output_tokens = estimate_tokens(summary)

        if llm_provider == "ollama":
            cost_usd = 0.0
        else:
            cost_usd = estimate_cost_usd(
                model=llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        latency_sec = round(time.time() - start_time, 3)

        log_query(
            {
                "query": topic,
                "retrieval_mode": retrieval_mode,
                "generation_mode": "summary",
                "llm_provider": llm_provider,
                "model": llm_model,
                "top_k": top_k,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "latency_sec": latency_sec,
                "num_sources": len(results),
                "fallback": False,
            }
        )

        return {
            "summary": summary,
            "results": results,
            "retrieval_mode": retrieval_mode,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "latency": latency_sec,
            "tokens": {"input": input_tokens, "output": output_tokens},
            "cost_usd": cost_usd,
        }