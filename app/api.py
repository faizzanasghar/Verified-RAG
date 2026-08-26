"""
FastAPI Backend for Verified Multi-Agent RAG.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure repo root is on system path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents import LLMGenerator, RetrieverAgent, MultiAgentVerifiedRAG
from src.verification import NLIVerifier

app = FastAPI(
    title="Verified RAG API",
    version="1.0.0",
    description="Fact-Checked Technical Question Answering with Claim-Level Verification and Bounded Repair."
)

# Global pipeline instance
pipeline: Optional[MultiAgentVerifiedRAG] = None


class QueryRequest(BaseModel):
    query: str = Field(..., example="What problem does residual learning address?")
    top_k: int = Field(default=5, ge=1, le=10)
    max_repairs: int = Field(default=1, ge=0, le=3)


class ClaimVerification(BaseModel):
    claim: str
    label: str
    entailment: float
    contradiction: float
    neutral: float


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    support_rate: float
    strict_pass: bool
    repairs_done: int
    latency_seconds: float
    claims: List[ClaimVerification]
    sources: List[str]


@app.on_event("startup")
def startup_event():
    global pipeline
    generator = LLMGenerator()
    retriever_agent = RetrieverAgent()
    verifier = NLIVerifier()
    pipeline = MultiAgentVerifiedRAG(
        generator=generator,
        retriever_agent=retriever_agent,
        verifier=verifier,
        max_repairs=1
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_initialized": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def run_query(payload: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized.")

    pipeline.max_repairs = payload.max_repairs
    result = pipeline.run(query=payload.query, top_k=payload.top_k)

    claims_out = []
    for item in result["verification"].get("claims", []):
        res = item.get("result", {})
        claims_out.append(
            ClaimVerification(
                claim=item.get("claim", ""),
                label=res.get("label", "NOT_ENOUGH_EVIDENCE"),
                entailment=round(res.get("entailment", 0.0), 4),
                contradiction=round(res.get("contradiction", 0.0), 4),
                neutral=round(res.get("neutral", 0.0), 4),
            )
        )

    sources = [c.get("chunk_id", "unknown") for c in result.get("retrieved_chunks", [])]

    return QueryResponse(
        query=result["query"],
        final_answer=result["final_answer"],
        support_rate=round(result["verification"]["support_rate"], 4),
        strict_pass=result["verification"]["strict_pass"],
        repairs_done=result["repairs_done"],
        latency_seconds=result["latency_seconds"],
        claims=claims_out,
        sources=sources,
    )
