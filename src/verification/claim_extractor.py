"""
Claim extraction utilities for atomic claim verification.
"""

from __future__ import annotations
import re
from typing import Callable, Optional


class ClaimExtractor:
    """Extracts atomic factual claims from generated text."""

    CLAIM_PROMPT = """Extract atomic factual statements from this text.
Return ONLY a numbered list (1., 2., 3.). Each line must be a single short factual claim.

TEXT:
{answer}
"""

    def __init__(self, generation_fn: Optional[Callable[[list[dict[str, str]], int], str]] = None) -> None:
        self.generation_fn = generation_fn

    @staticmethod
    def parse_claims(text: str) -> list[str]:
        """Parse numbered/bulleted lines or sentence splits into clean atomic claims."""
        claims: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            match = re.match(r"^(?:\d+[\).\:\-]|\-|\*)\s*(.+)$", line)
            if match and len(match.group(1).strip()) > 10:
                claims.append(match.group(1).strip())
        
        # Fallback: Split by sentence if regex list parsing failed
        if not claims:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            claims = [s.strip() for s in sentences if len(s.strip()) > 15]
            
        return claims

    def extract(self, answer: str, max_new_tokens: int = 250) -> list[str]:
        """Extract claims using LLM generation with reliable sentence-fallback."""
        if not answer.strip():
            return []

        # Remove source tags from claim text before NLI evaluation
        clean_text = re.sub(r"\[SOURCE:\s*[^\]]+\]", "", answer).strip()

        if self.generation_fn is not None:
            try:
                output = self.generation_fn([
                    {"role": "system", "content": "You are a factual claim extractor. Output only a numbered list."},
                    {"role": "user", "content": self.CLAIM_PROMPT.format(answer=clean_text)}
                ], max_new_tokens=max_new_tokens)
                claims = self.parse_claims(output)
                if claims:
                    return claims
            except Exception:
                pass

        return self.parse_claims(clean_text)
