"""Structured logging for pipeline runs and LLM calls."""

from __future__ import annotations

import json
import time
from pathlib import Path

LOG_DIR = Path("output")


def _log_line(filename: str, data: dict) -> None:
    """Append a JSONL line to the log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG_DIR / filename, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


def log_llm_call(
    *,
    provider: str,
    model: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    latency_ms: float,
    prompt_length: int,
    system_length: int,
    response_length: int,
) -> None:
    """Log an LLM API call to output/llm_calls.jsonl."""
    _log_line("llm_calls.jsonl", {
        "provider": provider,
        "model": model,
        "purpose": purpose,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "latency_ms": round(latency_ms, 1),
        "prompt_length": prompt_length,
        "system_length": system_length,
        "response_length": response_length,
    })


def log_pipeline_run(
    *,
    source: str,
    source_type: str,
    source_title: str,
    channel: str,
    duration_seconds: float,
    segment_count: int,
    content_segment_count: int,
    cleanup: bool,
    enrich: bool,
    glossary: bool,
    speakers: bool,
    provider: str,
    model: str,
    total_tokens: int,
    status: str,
    output_epub: str,
    output_md: str,
    phase_metrics: list[dict] | None = None,
    error: str = "",
) -> None:
    """Log a pipeline run to output/runs.jsonl."""
    _log_line("runs.jsonl", {
        "source": source,
        "source_type": source_type,
        "source_title": source_title,
        "channel": channel,
        "duration_seconds": duration_seconds,
        "segment_count": segment_count,
        "content_segment_count": content_segment_count,
        "cleanup": cleanup,
        "enrich": enrich,
        "glossary": glossary,
        "speakers": speakers,
        "provider": provider,
        "model": model,
        "total_tokens": total_tokens,
        "status": status,
        "output_epub": output_epub,
        "output_md": output_md,
        "phase_metrics": phase_metrics or [],
        "error": error[:500] if error else "",
    })
