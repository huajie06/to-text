# AGENTS.md

## Setup

```bash
git clone <url> && cd to-text
uv sync --extra dev       # +pytest
uv sync --extra openai    # DeepSeek/OpenAI provider
uv sync --extra diarize   # pyannote speaker diarization (--force-diarize)
brew install ffmpeg       # audio transcription dep
```

`.env` auto-loaded via `python-dotenv` at CLI import time (not via `export`).

## Commands

```bash
uv run pytest                      # all tests; no CI, no linter, no typechecker
uv run podbook --help              # entrypoint: podbook.cli.main:app
uv run python -c "from podbook.models import Segment, Transcript"
```

## Architecture

Linear pipeline: **source → transcript → normalize+preprocess → AI passes → markdown → EPUB**.

- `Transcript` / `Segment` (pydantic) are immutable — LLM passes produce derivative structures, never mutate originals.
- YouTube always uses **faster-whisper** for transcription (`subs=False` on `extract_youtube()`). Subtitles are no longer used as a transcript source — only metadata (title, channel, description) is fetched from yt-dlp.
- Preprocessing runs before any LLM call: classifies segments as CONTENT/AD/SELF_PROMO/META/FILLER, filters to CONTENT only.
- `--speakers` auto-enabled with `--cleanup`.
- Speaker labeling has two paths:
  - **LLM-only** (default when HF token unavailable): one LLM call on utterance sample, then heuristic propagation to all segments.
  - **Acoustic** (`--force-diarize`): pyannote.audio produces `(start, end, SPEAKER_XX)` windows, merged into whisper segments via `assign_speakers()` using **any-overlap alignment**. If a segment overlaps both speakers, it gets a combined label (`SPEAKER_00_SPEAKER_01`). A follow-up LLM call maps IDs to real names via `map_speaker_ids()`.
- `cached_prefix` param on `LLMProvider.generate()`: Claude gets ephemeral cache markers; other providers prepend it to prompt.
- Chunking never splits mid-sentence.
- Phase metrics (`PhaseMetric` dataclass in `pipeline.py`) track duration + tokens per phase, displayed as a Rich table at the end.

## Diarization merge logic

In `diarize.py`, `assign_speakers()` uses any-overlap alignment:

1. Group diarization `(start, end, SPEAKER_XX)` tuples by speaker ID
2. For each whisper segment, check **any time overlap** with each speaker independently
3. Single match → tag with that ID
4. Multiple matches → combined label (`SPEAKER_00_SPEAKER_01`)
5. No match → default to longest-duration speaker

This replaces the old max-overlap approach which always lost short-interjection speakers.

In `speakers.py`, `map_speaker_ids()` handles combined labels:
- `_parse_speaker_ids()` extracts individual IDs from both simple and combined labels
- `_resolve_speaker_label()` maps combined labels through name map: `SPEAKER_00_SPEAKER_01` → `Joe_Rogan_Theo_Von`

## Potential enhancements

- Split whisper segments at diarization window boundaries for cleaner per-speaker attribution
- Voice-activity-based segmentation aligned with speaker turns
- Embedding-based speaker clustering (d-vectors) instead of time-alignment
- Streaming diarization for real-time workflows
- Speaker-conditional cleanup: route segments per-speaker to separate LLM calls

## Output layout

```
output/{hash[:8]}-{slug}/          # per-source directory
├── transcript.json                # raw whisper transcript (no speakers)
├── {title}.wav                    # downloaded audio
├── {title}.16k.wav                # resampled for whisper
├── diarization.json               # pyannote output (cached)
├── {slug}-{provider}.md           # markdown with provider in filename
├── {slug}-{provider}.epub         # EPUB output
output/runs.jsonl                  # pipeline run log
output/llm_calls.jsonl             # LLM call log
```

## Known issues (doc/code drift)

- `--glossary` flag and CLI option do **not exist**; `generate_glossary` in `summarize.py` is not implemented. The `glossary` ref in `pipeline.py:332` is a dead variable that would raise `NameError` at runtime. Remove it if you touch that area.
- `*raw.md` save on `--cleanup` (mentioned in CLI help) is **not implemented** — no raw markdown is persisted.
- Tests pass 61/62 with 1 pre-existing failure (`test_heavy_filler`). Preprocess classifier sometimes misses dense filler text (single sentence with 8+ filler words).
- Pyannote diarization on CPU is ~1x realtime. GPU acceleration recommended for long podcasts.
- Multi-speaker combined labels (`SPEAKER_00_SPEAKER_01`) produce awkward name rendering in final markdown (e.g. `Joe_Rogan_Theo_Von`). A downstream normalization pass could replace these with a single dominant speaker label at render time.
