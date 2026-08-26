"""
Retriever Agent wrapping dense retrieval operations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from src.retrieval.retriever import Retriever


class RetrieverAgent:
    """Agent in charge of querying FAISS and returning structured contexts."""

    def __init__(
        self,
        vector_store_dir: str | Path = "/content/verified-rag/data/vector_store",
        processed_dir: str | Path = "/content/verified-rag/data/processed",
        device: Optional[str] = None,
    ) -> None:
        self.retriever = Retriever(
            vector_store_dir=vector_store_dir,
            processed_dir=processed_dir,
            device=device,
        )

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Execute retrieval and prepare formatted context."""
        chunks = self.retriever.retrieve(query=query, top_k=top_k)
        formatted_context = self.retriever.format_context(chunks)
        return {
            "query": query,
            "top_k": top_k,
            "chunks": chunks,
            "formatted_context": formatted_context,
        }
