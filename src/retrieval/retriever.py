"""
FAISS-based dense retriever for VerifiedRAG.

Expected files:
    data/processed/corpus_metadata.json
    data/vector_store/faiss.index
    data/vector_store/chunks_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Retriever:
    """Dense retriever backed by SentenceTransformers + FAISS."""

    def __init__(
        self,
        vector_store_dir: str | Path = "/content/verified-rag/data/vector_store",
        processed_dir: str | Path = "/content/verified-rag/data/processed",
        device: str | None = None,
    ) -> None:
        self.vector_store_dir = Path(vector_store_dir)
        self.processed_dir = Path(processed_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        metadata_path = self.processed_dir / "corpus_metadata.json"
        index_path = self.vector_store_dir / "faiss.index"
        chunks_path = self.vector_store_dir / "chunks_metadata.json"

        for path in (metadata_path, index_path, chunks_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required retrieval artifact not found: {path}"
                )

        with metadata_path.open("r", encoding="utf-8") as f:
            corpus_metadata = json.load(f)

        with chunks_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(str(index_path))

        self.embedding_model_name = corpus_metadata.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device,
        )

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index size does not match chunks metadata: "
                f"{self.index.ntotal} vectors vs {len(self.chunks)} chunks."
            )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query as a normalized float32 vector."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        vector = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(vector, dtype="float32")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k relevant chunks."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        top_k = min(top_k, self.index.ntotal)

        query_vector = self.encode_query(query)
        scores, indices = self.index.search(query_vector, top_k)

        results: list[dict[str, Any]] = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            item = dict(self.chunks[int(index)])
            item["score"] = float(score)
            results.append(item)

        return results

    @staticmethod
    def format_context(results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks for an LLM prompt."""
        blocks = []
        for item in results:
            chunk_id = item.get("chunk_id", "unknown")
            text = item.get("text", "")
            blocks.append(f"[SOURCE: {chunk_id}]\n{text}")

        return "\n\n".join(blocks)
