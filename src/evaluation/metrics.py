"""
Evaluation metrics for RAG generation and claim-level verification.
"""

from __future__ import annotations
import re
from typing import Any
import numpy as np


def compute_citation_validity(answer: str, retrieved_chunk_ids: list[str]) -> dict[str, Any]:
    """Verify whether cited chunk IDs in the answer actually exist in the retrieved chunks."""
    cited_ids = re.findall(r"\[SOURCE:\s*([^\]]+)\]", answer)
    if not cited_ids:
        return {
            "has_citations": False,
            "valid_citations_count": 0,
            "invalid_citations_count": 0,
            "citation_validity_rate": 0.0,
            "cited_ids": [],
        }

    retrieved_set = set(retrieved_chunk_ids)
    valid_count = sum(1 for cid in cited_ids if cid in retrieved_set)
    invalid_count = len(cited_ids) - valid_count

    return {
        "has_citations": True,
        "valid_citations_count": valid_count,
        "invalid_citations_count": invalid_count,
        "citation_validity_rate": valid_count / len(cited_ids),
        "cited_ids": cited_ids,
    }


def compute_claim_metrics(verification_records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate claim-level verification statistics across query runs."""
    total_claims = sum(r.get("total_claims", 0) for r in verification_records)
    if total_claims == 0:
        return {
            "total_claims": 0,
            "support_rate": 0.0,
            "refutation_rate": 0.0,
            "insufficient_rate": 0.0,
            "strict_pass_rate": 0.0,
        }

    total_supported = sum(r.get("supported_count", 0) for r in verification_records)
    total_refuted = sum(r.get("refuted_count", 0) for r in verification_records)
    total_insufficient = sum(r.get("insufficient_count", 0) for r in verification_records)
    strict_passed = sum(1 for r in verification_records if r.get("strict_pass", False))

    return {
        "total_claims": total_claims,
        "support_rate": round(total_supported / total_claims, 4),
        "refutation_rate": round(total_refuted / total_claims, 4),
        "insufficient_rate": round(total_insufficient / total_claims, 4),
        "strict_pass_rate": round(strict_passed / len(verification_records), 4),
    }


def bootstrap_confidence_interval(
    baseline_scores: list[float],
    verified_scores: list[float],
    num_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> dict[str, float]:
    """Compute paired bootstrap 95% confidence interval for score differences."""
    rng = np.random.default_rng(seed)
    n = len(baseline_scores)
    diffs = np.array(verified_scores) - np.array(baseline_scores)

    sample_means = [
        float(np.mean(rng.choice(diffs, size=n, replace=True)))
        for _ in range(num_samples)
    ]

    lower = float(np.percentile(sample_means, 100 * (alpha / 2)))
    upper = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
    }
