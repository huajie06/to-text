"""OpenAI-compatible LLM provider."""

from __future__ import annotations

import os

from podbook.ai.providers.base import LLMProvider, TokenUsage


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict | None = None,
        *,
        name: str | None = None,
    ):
        self.model = model
        if name is not None:
            self.name = name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.extra_body = extra_body

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        *,
        cached_prefix: str | None = None,
        purpose: str = "",
    ) -> tuple[str, TokenUsage]:
        import time as _time
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        full_prompt = f"{cached_prefix}\n\n{prompt}" if cached_prefix else prompt
        messages.append({"role": "user", "content": full_prompt})

        kwargs = {"model": self.model, "messages": messages}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        t0 = _time.monotonic()
        resp = client.chat.completions.create(**kwargs)
        latency_ms = (_time.monotonic() - t0) * 1000
        content = resp.choices[0].message.content or ""
        usage = TokenUsage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

        from podbook.logging import log_llm_call
        log_llm_call(
            provider=self.name,
            model=self.model,
            purpose=purpose,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            latency_ms=latency_ms,
            prompt_length=len(full_prompt),
            system_length=len(system) if system else 0,
            response_length=len(content),
        )
        return content, usage
