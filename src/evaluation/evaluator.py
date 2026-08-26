"""
Comparative evaluation engine for VerifiedRAG.
"""

from __future__ import annotations
from typing import Any, Callable, Optional
import pandas as pd
from src.evaluation.metrics import compute_citation_validity, compute_claim_metrics, bootstrap_confidence_interval


class RAGEvaluator:
    """Runs systematic benchmarks comparing Baseline and Verified RAG pipelines."""

    def __init__(self, verified_pipeline: Any) -> None:
        self.pipeline = verified_pipeline

    def evaluate_benchmark(
        self,
        benchmark_queries: list[dict[str, str]],
        top_k: int = 5
    ) -> dict[str, Any]:
        """Runs evaluation over benchmark query set and compiles comparative metrics."""
        results = []

        for item in benchmark_queries:
            query = item["query"]
            expected_paper = item.get("expected_paper", "")

            run_res = self.pipeline.run(query=query, top_k=top_k)
            retrieved_chunk_ids = [c["chunk_id"] for c in run_res["retrieved_chunks"]]

            # Citation analysis
            citation_diag = compute_citation_validity(run_res["final_answer"], retrieved_chunk_ids)

            # Paper hit check
            paper_hit = any(expected_paper in cid for cid in retrieved_chunk_ids) if expected_paper else True

            results.append({
                "query": query,
                "expected_paper": expected_paper,
                "paper_retrieval_hit": paper_hit,
                "answer": run_res["final_answer"],
                "verification": run_res["verification"],
                "repairs_done": run_res["repairs_done"],
                "latency_seconds": run_res["latency_seconds"],
                "citation_diagnostics": citation_diag,
                "support_rate": run_res["verification"]["support_rate"],
            })

        verifications = [r["verification"] for r in results]
        aggregate_metrics = compute_claim_metrics(verifications)

        df = pd.DataFrame([
            {
                "query": r["query"],
                "expected_paper": r["expected_paper"],
                "paper_hit": r["paper_retrieval_hit"],
                "support_rate": round(r["support_rate"], 4),
                "strict_pass": r["verification"]["strict_pass"],
                "repairs": r["repairs_done"],
                "has_citations": r["citation_diagnostics"]["has_citations"],
                "citation_validity": round(r["citation_diagnostics"]["citation_validity_rate"], 4),
                "latency_s": r["latency_seconds"],
            }
            for r in results
        ])

        return {
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "summary_dataframe": df,
        }
