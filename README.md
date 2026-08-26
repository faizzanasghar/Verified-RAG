# Verified-RAG

**Multi-Agent Hallucination Mitigation via Claim-Level NLI Verification and Bounded Self-Repair**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Generator: Qwen2.5-3B](https://img.shields.io/badge/Generator-Qwen2.5--3B-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Verifier: DeBERTa-v3](https://img.shields.io/badge/Verifier-DeBERTa--v3--NLI-red.svg)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

---

## Overview

Standard Retrieval-Augmented Generation (RAG) cuts down on hallucination but still lets through subtle failure modes: distorted facts, unsupported logical leaps, and fabricated citations. This is especially costly in technical/scientific QA, where an ungrounded claim can look just as confident as a grounded one.

**Verified-RAG** treats answer generation as a **multi-agent claim-verification game** rather than a single generation pass:

1. **Dense Retrieval** — context chunks retrieved via cosine similarity over dense embeddings (FAISS + MiniLM).
2. **Grounded Generation** — a 4-bit quantized LLM (Qwen2.5-3B-Instruct) produces a draft answer, with every statement required to carry a `[SOURCE: chunk_id]` citation.
3. **Atomic Claim Extraction** — the draft is decomposed into independently verifiable atomic claims.
4. **NLI Verification** — each claim is checked against its cited premise chunk by an independent DeBERTa-v3 NLI model, producing a `SUPPORTED / REFUTED / NOT_ENOUGH_EVIDENCE` verdict.
5. **Bounded Self-Repair** — any claim that isn't `SUPPORTED` triggers a single, targeted revision under a fixed repair budget (`B=1`), followed by re-verification.

The generator and verifier are intentionally separate models. This avoids the failure mode where a model "verifies" its own hallucination as fact — the NLI verifier has no access to the generator's reasoning, only the claim and the retrieved evidence.

```
Query
  │
  ▼
[Retriever Agent] ──► FAISS Index (MiniLM Embeddings) ──► Top-k Chunks
                                                              │
                                                              ▼
                                          [Generator Agent] (Qwen2.5-3B-Instruct)
                                                              │
                                                              ▼
                                                     Preliminary Answer
                                                              │
                                                              ▼
                                          [Claim Extractor] ──► Atomic Claims {c₁, c₂, ..., cₘ}
                                                              │
                                                              ▼
                              [NLI Verifier Agent] (DeBERTa-v3-base-mnli-fever-anli)
                                          ◄── verified against Premise Chunks
                                                              │
                                                              ▼
                              Claim Verdicts: SUPPORTED | REFUTED | NOT_ENOUGH_EVIDENCE
                                                              │
                                    ┌─────────────── Strict Pass? ───────────────┐
                                    │ YES                                  NO   │
                                    ▼                                            ▼
                          Output Verified Answer          [Repair Agent] (Budget B=1)
                                                                    │
                                                                    ▼
                                                     Revised Answer ──► Re-Verification
```

---

## Results

Evaluated on a benchmark of technical-paper QA drawn from *Deep Residual Learning for Image Recognition* (He et al.) and the *Adam Optimizer* paper (Kingma & Ba).

| Pipeline | Claim Support Rate | Refutation Rate | Strict Pass Rate | Citation Validity | Mean Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline Standard RAG | 74.05% | 4.17% | 50.00% | 16.67% | **12.4s** |
| RAG + NLI Verification (no repair) | 74.05% | 4.17% | 50.00% | 16.67% | 18.2s |
| **Verified-RAG (Repair B=1)** | **81.55%** | **0.00%** | **66.67%** | **33.33%** | 24.8s |

- **Refutation eliminated:** the bounded repair loop drove the refuted-claim rate from 4.17% → 0.00% on this benchmark.
- **Support rate improved by +7.50 points** (paired bootstrap 95% CI: `[0.000, 0.1583]`).
- **Cost of verification:** roughly 2x the latency of an unverified baseline — the trade-off is explicit, not hidden.

> **Note on scale:** this benchmark is a small proof-of-concept (≈24 atomic claims across 2 source papers), not a large-scale evaluation — several of the percentages above resolve to counts like 1/24 or 2/6. The *direction* of the effect (refutations eliminated, support and citation validity up) is a meaningful signal that the verify-and-repair loop works as designed, but the confidence interval is wide and these numbers should be read as early evidence, not a generalizable benchmark result. Expanding to more papers/domains is the natural next step (see [Limitations](#limitations--future-work)).

---

## Repository Structure

```
Verified-RAG/
├── notebooks/                     # Exploratory notebooks & development logs
│   ├── 01_data_preparation.ipynb  # PDF parsing, semantic chunking, FAISS index
│   ├── 02_baseline_rag.ipynb      # Standard generator without verification
│   ├── 03_multi_agent_rag.ipynb   # Multi-agent claim extraction & repair loop
│   └── 04_evaluation.ipynb        # Benchmark evaluation & bootstrap statistics
│
├── src/                            # Modular production package
│   ├── retrieval/                  # FAISS dense index & context formatting
│   │   └── retriever.py
│   ├── verification/               # Atomic claim extraction & DeBERTa NLI engine
│   │   ├── claim_extractor.py
│   │   └── nli_verifier.py
│   ├── agents/                     # Multi-agent orchestration loop
│   │   ├── generator.py
│   │   ├── retriever_agent.py
│   │   └── verifier_agent.py
│   └── evaluation/                 # Benchmark metrics & bootstrap statistics
│       ├── evaluator.py
│       └── metrics.py
│
├── app/                            # Interactive service layer
│   ├── api.py                      # FastAPI backend (/query, /health)
│   └── streamlit_app.py            # Streamlit dashboard with claim-level inspection
│
├── experiments/                    # Benchmark run configs & ablation logs
├── data/                           # Processed corpus metadata & vector store
├── results/                        # Serialized benchmark logs & evaluation CSVs
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/faizzanasghar/Verified-RAG.git
cd Verified-RAG
pip install -r requirements.txt
```

### 2. Run the pipeline

```python
from src.agents import LLMGenerator, RetrieverAgent, MultiAgentVerifiedRAG
from src.verification import NLIVerifier

pipeline = MultiAgentVerifiedRAG(
    generator=LLMGenerator(model_name="Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True),
    retriever_agent=RetrieverAgent(),
    verifier=NLIVerifier(model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
    max_repairs=1,
)

result = pipeline.run("What problem does residual learning address?", top_k=4)
print("Answer:", result["final_answer"])
print("Support Rate:", result["verification"]["support_rate"])
```

### 3. Launch the interactive services

```bash
# FastAPI backend
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Streamlit dashboard (claim-by-claim verdict inspection)
streamlit run app/streamlit_app.py
```

### 4. Reproduce the benchmark

```bash
jupyter nbconvert --to notebook --execute notebooks/04_evaluation.ipynb
```
This regenerates the results table above from `data/` and writes fresh logs to `results/`.

---

## Limitations & Future Work

- **Small evaluation set.** Current results are drawn from two papers and a limited claim set (~24 atomic claims). The reported deltas are a promising signal, not a statistically robust benchmark — the confidence interval on the accuracy gain is wide (`[0.000, 0.1583]`).
- **Repair budget is fixed at B=1.** It's untested whether B=2+ yields further gains or diminishing/negative returns (e.g., a claim that gets "repaired" into a different unsupported claim).
- **Single-verifier design.** The NLI verifier's own errors are a ceiling on the whole system — a claim it misjudges will not be caught. An ensemble or a second-pass human-in-the-loop check is a natural extension.
- **Domain scope.** Evaluated only on ML-paper QA. Generalization to other technical domains (medicine, law) is untested and citation-style/claim-density differences could affect the extractor's accuracy.
- **Planned:** expand the benchmark corpus, add ablations over repair budget, and add an automated regression check so `results/` stays reproducible as the pipeline changes.

---

## Citation

```bibtex
@misc{asghar2026verifiedrag,
  author       = {Muhammad Faizan Asghar},
  title        = {Verified-RAG: Multi-Agent Hallucination Mitigation with Claim-Level NLI Verification and Bounded Self-Repair},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/faizzanasghar/Verified-RAG}}
}
```

## License

Released under the [MIT License](LICENSE).
