"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract LLM provider with token tracking."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> tuple[str, TokenUsage]:
        """Generate a response. Returns (text, token_usage)."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4
