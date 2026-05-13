"""Ollama (local) LLM provider via OpenAI-compatible API."""

from __future__ import annotations

from podbook.ai.providers.base import LLMProvider, TokenUsage


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.base_url = base_url

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

        client = OpenAI(base_url=self.base_url, api_key="ollama")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        full_prompt = f"{cached_prefix}\n\n{prompt}" if cached_prefix else prompt
        messages.append({"role": "user", "content": full_prompt})

        t0 = _time.monotonic()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            frequency_penalty=0.3,
            presence_penalty=0.2,
            max_tokens=2048,
        )
        latency_ms = (_time.monotonic() - t0) * 1000
        content = resp.choices[0].message.content or ""
        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
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
