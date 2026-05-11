"""Anthropic Claude LLM provider with prompt caching."""

from __future__ import annotations

import os

from podbook.ai.providers.base import LLMProvider, TokenUsage


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        *,
        cached_prefix: str | None = None,
    ) -> tuple[str, TokenUsage]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        # Cache the system prompt — stable across all chunks in a pipeline run
        system_param = None
        if system:
            system_param = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Cache the stable prefix (context, speaker info) separately from the
        # variable chunk. Claude reuses the cached prefix on subsequent chunk calls.
        if cached_prefix:
            user_content: list[dict] | str = [
                {
                    "type": "text",
                    "text": cached_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            user_content = prompt

        kwargs: dict = {"messages": [{"role": "user", "content": user_content}]}
        if system_param is not None:
            kwargs["system"] = system_param

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            **kwargs,
        )

        content = response.content[0].text if response.content else ""
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )
        return content, usage
