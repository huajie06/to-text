# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install and sync dependencies
uv sync
uv sync --extra dev          # includes pytest

# Run CLI
podbook --help
podbook build <url>
podbook build --dry-run <url>
podbook build --force-transcribe <url>
podbook build --max-tokens 50000 <url>
podbook build --cleanup --enrich --provider claude <url>
podbook build --cleanup --enrich --glossary --provider claude <url>
podbook build --cleanup --enrich --provider deepseek <url>

podbook transcript <url>                 # extract + save transcript JSON
podbook epub transcript.json             # generate EPUB from saved transcript

podbook cache list                       # show cached artifacts
podbook cache clear --type audio         # clear a specific artifact type

# Run tests
uv run pytest
uv run pytest tests/test_preprocess.py  # single file

# Quick module test
uv run python -c "from podbook.models import Segment, Transcript; ..."
```

## Architecture

The pipeline is a linear chain: **source → transcript → preprocess → AI passes → markdown → EPUB**. Each stage depends only on the output of the previous stage.

### Data model (`podbook/models.py`)

`Segment` and `Transcript` are the canonical, **immutable** data types that flow through the pipeline. `Segment` holds timing + speaker + text. `Transcript` holds metadata + a list of `Segment`. Never bypass these — all modules consume/generate `Transcript`.

`TokenUsage` tracks `input_tokens`, `output_tokens`, `cache_write_tokens`, and `cache_read_tokens`. All LLM calls return a `TokenUsage`.

### Source layer (`podbook/sources/`)

Each source module returns a `Transcript`. YouTube is the only source that can return populated `segments` (via subtitles). All others return `segments=[]`, which triggers the transcription fallback in the pipeline.

### Transcript layer (`podbook/transcript/`)

- `subtitles.py` — SRT/VTT parsing
- `whisper.py` — wraps `pywhispercpp` (whisper.cpp Python bindings), expects 16kHz mono WAV
- `normalize.py` — merges short segments, fixes overlaps, strips empties; always called before any downstream use
- `preprocess.py` — classifies segments as CONTENT / AD / SELF_PROMO / META / FILLER using regex triggers + contextual fixes; `filter_content()` keeps only CONTENT before LLM passes
- `chunking.py` — splits at sentence/paragraph boundaries for LLM processing, never mid-sentence

### AI layer (`podbook/ai/`)

Providers follow the `LLMProvider` ABC (`providers/base.py`). Every call returns `(str, TokenUsage)`.

**`generate(prompt, system, *, cached_prefix)`** — the `cached_prefix` kwarg carries stable context that precedes the variable prompt. Claude marks it as a cached content block; other providers prepend it to the prompt. This is the mechanism for prompt caching in multi-chunk passes.

Supported providers (via `--provider`):

| Flag value | Class | Default model |
|---|---|---|
| `ollama` | `OllamaProvider` | `llama3.2` |
| `openai` | `OpenAIProvider` | `gpt-4o-mini` |
| `claude` | `ClaudeProvider` | `claude-haiku-4-5-20251001` |
| `deepseek` | `OpenAIProvider` (alt base_url) | `deepseek-chat` |

Token usage is tracked from the start — the CLI enforces `--max-tokens` and `--dry-run` skips LLM calls entirely.

#### AI passes

- `cleanup.py` — chunked transcript cleaning; stable context prefix is cached by Claude across all chunks
- `summarize.py` — `generate_chapters()`, `generate_takeaways()`, `generate_summary()`, `generate_glossary()`; each call passes a `cached_prefix` containing the shared context + transcript block

### Ebook layer (`podbook/ebook/`)

- `markdown.py` — generates canonical markdown from a `Transcript` + optional chapters/takeaways/summary/glossary
- `epub.py` — converts markdown to EPUB via `ebooklib`; uses the `markdown` Python library (with `extra` + `nl2br` extensions) for MD→HTML conversion; splits on H1 headings into chapters

### Pipeline (`podbook/pipeline.py`)

`run_pipeline()` orchestrates the full flow. `extract_transcript()` is also exported for use by the `transcript` CLI subcommand. Phases:

1. Extract transcript (subtitles → audio → whisper fallback)
2. Normalize + preprocess (classify and filter non-content segments)
3. AI passes: cleanup, chapters, takeaways, summary, glossary (all optional)
4. Generate markdown
5. Generate EPUB

Cache files live in `output/.cache/` keyed by SHA-256 of the source URL.

## Design rules

- **Transcript-first**: always prefer existing subtitles over transcription. Transcription is the fallback.
- **Immutable canonical transcript**: `Transcript.segments` is the ground truth. LLM cleanup produces derivative structures (`Chapter`, cleaned segments, summaries, takeaways, glossary) — never mutates the original.
- **Chunk before LLM**: never send an entire podcast to an LLM in one call. Use `chunking.py` with sentence/paragraph boundary awareness.
- **Token tracking on day one**: all LLM calls must return `TokenUsage`. CLI enforces `--max-tokens`. `--dry-run` estimates without spending.
- **Cache everything**: downloaded audio, subtitles, transcripts, markdown, EPUB. Cache files are keyed by content hash.
- **`cached_prefix` for stable context**: any text that is identical across multiple LLM calls in the same run (system prompt, podcast context, full transcript) must be passed via `cached_prefix` so Claude can cache it. Other providers ignore it gracefully.
- **No premature complexity**: no vector DBs, agents, streaming, microservices, or heavy web UIs until the core pipeline is stable and proven.

## Tests

```bash
uv run pytest          # run all tests
```

Tests live in `tests/`. Key coverage:

- `test_normalize.py` — segment merging, overlap fixing, empty removal
- `test_preprocess.py` — regex classification of ads, self-promo, meta, filler
- `test_chunking.py` — sentence boundary splitting, segment preservation
- `test_markdown.py` — markdown generation with all enrichments (chapters, takeaways, summary, glossary)
- `test_epub.py` — H1 splitting, title extraction, HTML conversion, EPUB file validity
