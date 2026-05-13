"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract LLM provider with token tracking."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        *,
        cached_prefix: str | None = None,
        purpose: str = "",
    ) -> tuple[str, TokenUsage]:
        """Generate a response. Returns (text, token_usage).

        cached_prefix: stable text preceding the variable prompt.
        Claude caches this block; other providers prepend it to prompt.
        purpose: label for logging (speakers, cleanup, chapters, etc.)
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4
