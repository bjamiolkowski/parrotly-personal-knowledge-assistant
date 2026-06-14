"""
Retrieval evaluation script.

Compares retrieval configurations across predefined test queries.

The evaluation focuses on retrieval quality, not final answer quality.
It checks whether the RAG pipeline retrieves the expected source documents.

Outputs:
- evaluation/results/retrieval_comparison.csv
- evaluation/results/retrieval_details.json
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, TypedDict

from evaluation.test_cases import TEST_QUERIES
from rag.orchestration.pipeline import ModularRAGPipeline
from rag.retrieval.sparse import build_tfidf_index
from rag.utils.io import load_chunks, load_index


RESULTS_DIR = Path("evaluation/results")
SUMMARY_CSV_PATH = RESULTS_DIR / "retrieval_comparison.csv"
DETAILS_JSON_PATH = RESULTS_DIR / "retrieval_details.json"


class TestQuery(TypedDict):
    """Single retrieval evaluation test case."""

    query: str
    expected_sources: list[str]


class ExperimentConfig(TypedDict):
    """Retrieval experiment configuration."""

    name: str
    retrieval_mode: str
    top_k: int
    faiss_k: int
    tfidf_k: int
    alpha: float


class ExperimentSummary(TypedDict):
    """Aggregated metrics for one retrieval experiment."""

    experiment: str
    retrieval_mode: str
    top_k: int
    faiss_k: int
    tfidf_k: int
    alpha: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    avg_latency_sec: float
    num_queries: int


EXPERIMENTS: list[ExperimentConfig] = [
    {
        "name": "dense_top5",
        "retrieval_mode": "dense",
        "top_k": 5,
        "faiss_k": 20,
        "tfidf_k": 20,
        "alpha": 0.6,
    },
    {
        "name": "sparse_top5",
        "retrieval_mode": "sparse",
        "top_k": 5,
        "faiss_k": 20,
        "tfidf_k": 20,
        "alpha": 0.6,
    },
    {
        "name": "hybrid_a03_top5",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "faiss_k": 20,
        "tfidf_k": 20,
        "alpha": 0.3,
    },
    {
        "name": "hybrid_a06_top5",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "faiss_k": 20,
        "tfidf_k": 20,
        "alpha": 0.6,
    },
    {
        "name": "hybrid_a08_top5",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "faiss_k": 20,
        "tfidf_k": 20,
        "alpha": 0.8,
    },
    {
        "name": "hybrid_a06_top10",
        "retrieval_mode": "hybrid",
        "top_k": 10,
        "faiss_k": 30,
        "tfidf_k": 30,
        "alpha": 0.6,
    },
]


def build_pipeline() -> ModularRAGPipeline:
    """Build the RAG pipeline from saved retrieval artifacts."""
    index = load_index()
    chunks = load_chunks()
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)

    return ModularRAGPipeline(
        index=index,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """Return reciprocal rank of the first expected source in retrieved results."""
    expected_set = set(expected_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_set:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    k: int,
) -> float:
    """Compute Recall@K for expected source documents."""
    expected_set = set(expected_sources)

    if not expected_set:
        return 0.0

    retrieved_top_k = set(retrieved_sources[:k])
    return len(retrieved_top_k & expected_set) / len(expected_set)


def hit_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    k: int,
) -> bool:
    """Return True if any expected source appears in the top K results."""
    expected_set = set(expected_sources)
    return any(source in expected_set for source in retrieved_sources[:k])


def evaluate_experiment(
    test_queries: list[TestQuery],
    pipeline: ModularRAGPipeline,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    """Evaluate a single retrieval configuration on all test queries."""
    if not test_queries:
        raise ValueError("No test queries provided.")

    details: list[dict[str, Any]] = []

    hit_at_1_total = 0
    hit_at_3_total = 0
    hit_at_5_total = 0

    mrr_total = 0.0
    recall_at_1_total = 0.0
    recall_at_3_total = 0.0
    recall_at_5_total = 0.0
    latency_total = 0.0

    for item in test_queries:
        query = item["query"]
        expected_sources = item["expected_sources"]

        start_time = time.perf_counter()

        results = pipeline.retrieve(
            query=query,
            top_k=experiment["top_k"],
            faiss_k=experiment["faiss_k"],
            tfidf_k=experiment["tfidf_k"],
            alpha=experiment["alpha"],
            retrieval_mode=experiment["retrieval_mode"],
        )

        latency_sec = round(time.perf_counter() - start_time, 4)
        latency_total += latency_sec

        retrieved_sources = [result["source"] for result in results]

        hit_1 = hit_at_k(retrieved_sources, expected_sources, 1)
        hit_3 = hit_at_k(retrieved_sources, expected_sources, 3)
        hit_5 = hit_at_k(retrieved_sources, expected_sources, 5)

        rr = reciprocal_rank(retrieved_sources, expected_sources)
        r_at_1 = recall_at_k(retrieved_sources, expected_sources, 1)
        r_at_3 = recall_at_k(retrieved_sources, expected_sources, 3)
        r_at_5 = recall_at_k(retrieved_sources, expected_sources, 5)

        hit_at_1_total += int(hit_1)
        hit_at_3_total += int(hit_3)
        hit_at_5_total += int(hit_5)

        mrr_total += rr
        recall_at_1_total += r_at_1
        recall_at_3_total += r_at_3
        recall_at_5_total += r_at_5

        details.append(
            {
                "experiment": experiment["name"],
                "query": query,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "hit_at_1": hit_1,
                "hit_at_3": hit_3,
                "hit_at_5": hit_5,
                "reciprocal_rank": rr,
                "recall_at_1": r_at_1,
                "recall_at_3": r_at_3,
                "recall_at_5": r_at_5,
                "latency_sec": latency_sec,
            }
        )

    num_queries = len(test_queries)

    summary: ExperimentSummary = {
        "experiment": experiment["name"],
        "retrieval_mode": experiment["retrieval_mode"],
        "top_k": experiment["top_k"],
        "faiss_k": experiment["faiss_k"],
        "tfidf_k": experiment["tfidf_k"],
        "alpha": experiment["alpha"],
        "hit_at_1": round(hit_at_1_total / num_queries, 4),
        "hit_at_3": round(hit_at_3_total / num_queries, 4),
        "hit_at_5": round(hit_at_5_total / num_queries, 4),
        "mrr": round(mrr_total / num_queries, 4),
        "recall_at_1": round(recall_at_1_total / num_queries, 4),
        "recall_at_3": round(recall_at_3_total / num_queries, 4),
        "recall_at_5": round(recall_at_5_total / num_queries, 4),
        "avg_latency_sec": round(latency_total / num_queries, 4),
        "num_queries": num_queries,
    }

    return {
        "summary": summary,
        "details": details,
    }


def run_experiments(
    test_queries: list[TestQuery] = TEST_QUERIES,
    experiments: list[ExperimentConfig] = EXPERIMENTS,
) -> dict[str, Any]:
    """Run all configured retrieval experiments."""
    pipeline = build_pipeline()

    summaries: list[ExperimentSummary] = []
    details: list[dict[str, Any]] = []

    for experiment in experiments:
        print(f"Running experiment: {experiment['name']}")

        result = evaluate_experiment(
            test_queries=test_queries,
            pipeline=pipeline,
            experiment=experiment,
        )

        summaries.append(result["summary"])
        details.extend(result["details"])

    return {
        "summaries": summaries,
        "details": details,
    }


def save_results(results: dict[str, Any]) -> None:
    """Save experiment summaries to CSV and detailed results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = results["summaries"]
    details = results["details"]

    if summaries:
        with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    with open(DETAILS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=2, ensure_ascii=False)


def print_comparison_table(summaries: list[ExperimentSummary]) -> None:
    """Print a compact comparison table for all retrieval experiments."""
    print("\n" + "=" * 110)
    print("RETRIEVAL EXPERIMENT COMPARISON")
    print("=" * 110)

    print(
        f"{'Experiment':<20} "
        f"{'Mode':<8} "
        f"{'TopK':>5} "
        f"{'Alpha':>7} "
        f"{'Hit@1':>8} "
        f"{'Hit@3':>8} "
        f"{'Hit@5':>8} "
        f"{'MRR':>8} "
        f"{'R@5':>8} "
        f"{'Latency':>10}"
    )
    print("-" * 110)

    for row in summaries:
        print(
            f"{row['experiment']:<20} "
            f"{row['retrieval_mode']:<8} "
            f"{row['top_k']:>5} "
            f"{row['alpha']:>7.2f} "
            f"{row['hit_at_1']:>8.4f} "
            f"{row['hit_at_3']:>8.4f} "
            f"{row['hit_at_5']:>8.4f} "
            f"{row['mrr']:>8.4f} "
            f"{row['recall_at_5']:>8.4f} "
            f"{row['avg_latency_sec']:>10.4f}"
        )


def print_best_experiment(summaries: list[ExperimentSummary]) -> None:
    """Print the best experiment ranked by MRR, Recall@5 and latency."""
    if not summaries:
        return

    ranked = sorted(
        summaries,
        key=lambda item: (
            item["mrr"],
            item["recall_at_5"],
            -item["avg_latency_sec"],
        ),
        reverse=True,
    )

    best = ranked[0]

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Experiment: {best['experiment']}")
    print(f"Retrieval mode: {best['retrieval_mode']}")
    print(f"Top K: {best['top_k']}")
    print(f"Alpha: {best['alpha']}")
    print(f"MRR: {best['mrr']:.4f}")
    print(f"Recall@5: {best['recall_at_5']:.4f}")
    print(f"Average latency: {best['avg_latency_sec']:.4f}s")


if __name__ == "__main__":
    experiment_results = run_experiments()
    save_results(experiment_results)

    print_comparison_table(experiment_results["summaries"])
    print_best_experiment(experiment_results["summaries"])

    print("\nSaved results:")
    print(f"- {SUMMARY_CSV_PATH}")
    print(f"- {DETAILS_JSON_PATH}")