"""
NLI-based fact-checking verifier.
"""

from __future__ import annotations
from typing import Any, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIVerifier:
    """Premise-hypothesis NLI classifier for claim validation."""

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: Optional[str] = None,
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.80,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_pair(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run NLI classification over a single premise-hypothesis pair."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        scores: dict[str, float] = {}

        for i, p in enumerate(probs):
            label = self.model.config.id2label[i].lower()
            if "entail" in label:
                scores["entailment"] = float(p)
            elif "contrad" in label:
                scores["contradiction"] = float(p)
            elif "neutral" in label:
                scores["neutral"] = float(p)

        return scores

    def verify_claim_against_chunks(
        self,
        claim: str,
        evidence_chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Verify a single claim against multiple retrieved evidence chunks."""
        pairs: list[dict[str, Any]] = []
        for item in evidence_chunks:
            pairs.append({
                "chunk_id": item.get("chunk_id", "unknown"),
                "text": item.get("text", ""),
                "retrieval_score": item.get("score", 0.0),
                "nli": self.predict_pair(item.get("text", ""), claim)
            })

        if not pairs:
            return {
                "label": "NOT_ENOUGH_EVIDENCE",
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
                "evidence": []
            }

        ent = max(x["nli"]["entailment"] for x in pairs)
        con = max(x["nli"]["contradiction"] for x in pairs)
        neu = max(x["nli"]["neutral"] for x in pairs)

        if con >= self.contradiction_threshold and con > ent:
            label = "REFUTED"
        elif ent >= self.entailment_threshold:
            label = "SUPPORTED"
        else:
            label = "NOT_ENOUGH_EVIDENCE"

        return {
            "label": label,
            "entailment": ent,
            "contradiction": con,
            "neutral": neu,
            "evidence": pairs
        }
