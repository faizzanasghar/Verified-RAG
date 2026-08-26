"""
Quantized LLM Generator for VerifiedRAG.
"""

from __future__ import annotations
from typing import Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMGenerator:
    """Wrapper around Qwen2.5 causal LM with 4-bit quantization."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a strict research assistant. Answer the question using ONLY the provided context blocks.\n"
        "Cite every factual statement using [SOURCE: chunk_id].\n"
        "If the context does not contain enough information, state that clearly."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        bnb_config = None
        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if bnb_config else None,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 300,
        temperature: float = 0.1,
    ) -> str:
        """Run chat-template generation."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def generate_grounded(
        self,
        query: str,
        formatted_context: str,
        max_new_tokens: int = 300,
    ) -> str:
        """Helper to generate a direct context-grounded answer."""
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{formatted_context}\n\nQUESTION: {query}",
            },
        ]
        return self.generate(messages, max_new_tokens=max_new_tokens)
