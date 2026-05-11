# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install and sync dependencies
uv sync

# Run CLI (after uv sync)
podbook --help
podbook build <url>
podbook build --dry-run <url>
podbook build --force-transcribe <url>
podbook build --max-tokens 50000 <url>
podbook transcript <url>
podbook epub transcript.md

# Quick module test via Python
uv run python -c "from podbook.models import Segment, Transcript; ..."
```

No test runner or linter is configured yet.

## Architecture

The pipeline is a linear chain: **source → transcript → markdown → EPUB**. Each stage depends only on the output of the previous stage.

### Data model (`podbook/models.py`)

`Segment` and `Transcript` are the canonical, **immutable** data types that flow through the pipeline. `Segment` holds timing + speaker + text. `Transcript` holds metadata + a list of `Segment`. Never bypass these — all modules consume/generate `Transcript`.

### Source layer (`podbook/sources/`)

Each source module returns a `Transcript`. YouTube is the only source that can return populated `segments` (via subtitles). All others return `segments=[]`, which triggers the transcription fallback in the pipeline.

### Transcript layer (`podbook/transcript/`)

- `subtitles.py` — SRT/VTT parsing, used for non-YouTube subtitle files
- `whisper.py` — wraps `pywhispercpp` (whisper.cpp Python bindings), expects 16kHz mono WAV
- `normalize.py` — merges short segments, fixes overlaps, strips empties; always called before any downstream use
- `chunking.py` — splits at sentence/paragraph boundaries for LLM processing, never mid-sentence

### AI layer (`podbook/ai/`)

Providers follow the `LLMProvider` ABC (`providers/base.py`). Every call returns `(str, TokenUsage)`. Token usage is tracked from the start — the CLI enforces `--max-tokens` and `--dry-run` skips LLM calls entirely. Not yet wired into the pipeline (Phase 2).

### Ebook layer (`podbook/ebook/`)

- `markdown.py` — generates canonical markdown from a `Transcript` + optional chapters/takeaways/summary
- `epub.py` — converts markdown to EPUB via `ebooklib`, splits on H1 headings into chapters, simple serif CSS

### Pipeline (`podbook/pipeline.py`)

`run_pipeline()` orchestrates the full flow. It is the single entry point called by the CLI `build` command. Phases: extract transcript → normalize → generate markdown → generate EPUB. Caching and LLM passes are not yet wired in (Phase 3+).

## Design rules

- **Transcript-first**: always prefer existing subtitles over transcription. Transcription is the fallback.
- **Immutable canonical transcript**: `Transcript.segments` is the ground truth. LLM cleanup produces derivative structures (`Chapter`, summaries, takeaways) — never mutates the original.
- **Chunk before LLM**: never send an entire podcast to an LLM. Use `chunking.py` with sentence/paragraph boundary awareness.
- **Token tracking on day one**: all LLM calls must report `TokenUsage`. CLI enforces `--max-tokens`. `--dry-run` estimates without spending.
- **Cache everything**: downloaded audio, subtitles, transcripts, LLM responses, markdown, EPUB. Use content-hash keys for LLM cache entries.
- **No premature complexity**: no vector DBs, agents, streaming, microservices, or heavy web UIs until the core pipeline is stable and proven.
