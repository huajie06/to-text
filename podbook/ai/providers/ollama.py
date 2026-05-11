"""Ollama (local) LLM provider via OpenAI-compatible API."""

from __future__ import annotations

from podbook.ai.providers.base import LLMProvider, TokenUsage


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str | None = None) -> tuple[str, TokenUsage]:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key="ollama")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            frequency_penalty=0.3,
            presence_penalty=0.2,
            max_tokens=2048,
        )
        content = resp.choices[0].message.content or ""
        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return content, usage
