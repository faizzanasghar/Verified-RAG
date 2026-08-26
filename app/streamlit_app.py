"""
Streamlit Interface for Verified Multi-Agent RAG.
"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents import LLMGenerator, RetrieverAgent, MultiAgentVerifiedRAG
from src.verification import NLIVerifier

st.set_page_config(
    page_title="Verified-RAG | Hallucination-Resistant QA",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Verified Multi-Agent RAG")
st.caption("Factual Technical Question Answering with Claim-Level NLI Verification & Bounded Self-Repair")


@st.cache_resource(show_spinner="Loading models into GPU memory...")
def load_pipeline():
    generator = LLMGenerator()
    retriever_agent = RetrieverAgent()
    verifier = NLIVerifier()
    return MultiAgentVerifiedRAG(
        generator=generator,
        retriever_agent=retriever_agent,
        verifier=verifier,
        max_repairs=1
    )

pipeline = load_pipeline()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=4)
    max_repairs = st.slider("Max Repair Iterations", min_value=0, max_value=2, value=1)
    st.markdown("---")
    st.markdown("**Core Models:**")
    st.markdown("- **Generator:** `Qwen/Qwen2.5-3B-Instruct` (4-bit)")
    st.markdown("- **Verifier:** `DeBERTa-v3-base-mnli-fever-anli`")
    st.markdown("- **Retriever:** `all-MiniLM-L6-v2` + FAISS")

# Preset questions
sample_questions = [
    "What problem does residual learning address?",
    "How do residual shortcuts help optimize very deep networks?",
    "What is the mathematical formulation of the residual building block?",
    "How does Adam combine Momentum and RMSprop?",
    "What bias correction mechanism does Adam introduce?",
    "What are the default hyperparameter values recommended for Adam?"
]

selected_query = st.selectbox("Select a benchmark query:", sample_questions)
custom_query = st.text_input("Or enter a custom question:", value=selected_query)

if st.button("Run Verified RAG", type="primary"):
    if not custom_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving, generating, and verifying claims..."):
            pipeline.max_repairs = max_repairs
            res = pipeline.run(query=custom_query, top_k=top_k)

        st.subheader("📝 Verified Answer")
        st.markdown(res["final_answer"])

        # Metric Badges
        col1, col2, col3, col4 = st.columns(4)
        support_pct = res["verification"]["support_rate"] * 100
        col1.metric("Support Rate", f"{support_pct:.1f}%")
        col2.metric("Strict Pass", "✅ Yes" if res["verification"]["strict_pass"] else "⚠️ Partial")
        col3.metric("Repairs Applied", res["repairs_done"])
        col4.metric("Latency", f"{res['latency_seconds']}s")

        st.markdown("---")
        
        # Claims Breakdown Table
        st.subheader("🔍 Claim-Level Verification Breakdown")
        for idx, item in enumerate(res["verification"]["claims"], start=1):
            lbl = item["result"]["label"]
            scores = item["result"]
            badge = "🟢 SUPPORTED" if lbl == "SUPPORTED" else ("🔴 REFUTED" if lbl == "REFUTED" else "🟡 INSUFFICIENT")
            
            with st.expander(f"Claim {idx}: {badge} — {item['claim'][:80]}..."):
                st.write(f"**Claim:** {item['claim']}")
                st.write(f"**Label:** `{lbl}`")
                st.write(f"**Entailment:** `{scores.get('entailment', 0.0):.4f}` | **Contradiction:** `{scores.get('contradiction', 0.0):.4f}` | **Neutral:** `{scores.get('neutral', 0.0):.4f}`")

        # Retrieved Sources
        st.subheader("📚 Retrieved Context Sources")
        for chunk in res["retrieved_chunks"]:
            with st.expander(f"Source Chunk: {chunk.get('chunk_id')} (Score: {round(chunk.get('score', 0), 4)})"):
                st.text(chunk.get("text", ""))
