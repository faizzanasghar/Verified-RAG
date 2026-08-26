"""
Multi-Agent Orchestrator with claim-level NLI verification and iterative repair.
"""

from __future__ import annotations
from typing import Any, Optional
import time

from src.retrieval.retriever import Retriever
from src.verification.claim_extractor import ClaimExtractor
from src.verification.nli_verifier import NLIVerifier
from src.agents.generator import LLMGenerator
from src.agents.retriever_agent import RetrieverAgent


class MultiAgentVerifiedRAG:
    """End-to-end Verified RAG pipeline."""

    REPAIR_PROMPT = """You are a precision repair agent.
The following answer contains unsupported or refuted claims.
Rewrite the answer so that EVERY claim is strictly supported by the CONTEXT.
Remove any unverified assertions. Maintain citations [SOURCE: chunk_id].

CONTEXT:
{context}

QUESTION:
{query}

ORIGINAL ANSWER:
{answer}

UNSUPPORTED CLAIMS TO FIX:
{issues}

REVISED ANSWER:
"""

    def __init__(
        self,
        generator: Optional[LLMGenerator] = None,
        retriever_agent: Optional[RetrieverAgent] = None,
        verifier: Optional[NLIVerifier] = None,
        max_repairs: int = 1,
    ) -> None:
        self.generator = generator or LLMGenerator()
        self.retriever_agent = retriever_agent or RetrieverAgent()
        self.verifier = verifier or NLIVerifier()
        self.claim_extractor = ClaimExtractor(generation_fn=self.generator.generate)
        self.max_repairs = max_repairs

    def verify_answer(self, answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract atomic claims and verify each against retrieved chunks."""
        claims = self.claim_extractor.extract(answer)
        claim_verifications = []
        supported_count = 0
        refuted_count = 0
        insufficient_count = 0

        for claim in claims:
            res = self.verifier.verify_claim_against_chunks(claim, chunks)
            claim_verifications.append({"claim": claim, "result": res})
            
            label = res["label"]
            if label == "SUPPORTED":
                supported_count += 1
            elif label == "REFUTED":
                refuted_count += 1
            else:
                insufficient_count += 1

        total = max(len(claims), 1)
        return {
            "claims": claim_verifications,
            "total_claims": len(claims),
            "supported_count": supported_count,
            "refuted_count": refuted_count,
            "insufficient_count": insufficient_count,
            "support_rate": supported_count / total,
            "strict_pass": (supported_count == len(claims)) and (len(claims) > 0),
        }

    def repair_answer(
        self,
        query: str,
        current_answer: str,
        formatted_context: str,
        verification_data: dict[str, Any]
    ) -> str:
        """Attempt to repair unsupported/refuted claims."""
        issues = []
        for item in verification_data.get("claims", []):
            label = item["result"]["label"]
            if label in ("REFUTED", "NOT_ENOUGH_EVIDENCE"):
                issues.append(f"- [{label}] {item['claim']}")

        if not issues:
            return current_answer

        prompt = self.REPAIR_PROMPT.format(
            context=formatted_context,
            query=query,
            answer=current_answer,
            issues="\n".join(issues),
        )

        messages = [
            {"role": "system", "content": "You repair and fact-align machine generated answers."},
            {"role": "user", "content": prompt},
        ]
        return self.generator.generate(messages, max_new_tokens=350)

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Execute full Verified RAG pipeline: Retrieve -> Generate -> Verify -> Repair."""
        start_time = time.time()

        # 1. Retrieval
        retrieval_res = self.retriever_agent.run(query=query, top_k=top_k)
        chunks = retrieval_res["chunks"]
        context = retrieval_res["formatted_context"]

        # 2. Initial Generation
        answer = self.generator.generate_grounded(query=query, formatted_context=context)

        # 3. Verification
        verification = self.verify_answer(answer, chunks)

        # 4. Repair Loop
        repairs_done = 0
        while not verification["strict_pass"] and repairs_done < self.max_repairs:
            repairs_done += 1
            repaired_answer = self.repair_answer(query, answer, context, verification)
            new_verification = self.verify_answer(repaired_answer, chunks)
            
            # Keep repair if it improved or matched support rate
            if new_verification["support_rate"] >= verification["support_rate"]:
                answer = repaired_answer
                verification = new_verification

        elapsed_time = round(time.time() - start_time, 3)

        return {
            "query": query,
            "final_answer": answer,
            "verification": verification,
            "retrieved_chunks": chunks,
            "repairs_done": repairs_done,
            "latency_seconds": elapsed_time,
        }
