# Changelog

## 2026-05-12

### Fixed

- **YouTube subtitle parsing: rolling-caption deduplication** (`podbook/transcript/subtitles.py`)
  - YouTube auto-generated captions repeat the previous cumulative text as a prefix in each new cue, producing doubled text (e.g. "one of the many things that comes to one of the many things that comes to mind…"). Added `_dedup_rolling_captions()` which strips the repeated prefix from both VTT and SRT parsers.
  - Result: segment count halved (358 → 179 for an 8-min clip), text clean.

- **YouTube VTT header pollution** (`podbook/transcript/subtitles.py`)
  - VTT metadata lines (`Kind: captions`, `Language: en`) before the first cue were being collected as subtitle text and injected into the first transcript segment. Fixed `parse_vtt` to skip all lines before the first `-->` timestamp.

- **yt-dlp subtitle format: json3 → VTT** (`podbook/sources/youtube.py`)
  - Current yt-dlp dropped `json3` as a valid `--sub-format` / `--convert-subs` value. Switched to `--convert-subs vtt` and replaced the custom `_parse_json3` parser with the existing `parse_vtt`. Applied inline-tag stripping (`<00:00:03.540>`, `<c>`, `</c>`) at parse time.

### Added

- **macOS dev environment support**
  - Verified end-to-end on macOS (Apple Silicon, Darwin). Install path: `brew install ollama ffmpeg`, then `ollama pull gemma4:e2b`.
  - Updated `README.md` and `CLAUDE.md` with macOS setup instructions, system dep table, and correct `--provider ollama --model gemma4:e2b` flag usage.

---

## 2026-05-11

Initial working version of PodBook — a podcast-to-ebook pipeline.

### Added

- **Core pipeline** (`podbook/pipeline.py`)
  - Linear chain: source → transcript → normalize/preprocess → AI passes → markdown → EPUB.
  - Transcript-first strategy: prefers existing subtitles over transcription; faster-whisper is the fallback.
  - Per-source cache directories keyed by content hash (`output/{hash[:8]}-{slug}/`).

- **Source layer** (`podbook/sources/`)
  - YouTube: subtitle download (falling back to audio) via `yt-dlp`.
  - Podcast webpage: HTML scraping via `requests` + `BeautifulSoup`.
  - RSS feed: audio enclosure extraction via `feedparser`.
  - Local file: direct path handling.

- **Transcript layer** (`podbook/transcript/`)
  - `normalize.py` — merges short segments, fixes overlaps, removes empties.
  - `preprocess.py` — regex-based classification of segments as CONTENT / AD / SELF_PROMO / META / FILLER; filters non-content before LLM passes.
  - `subtitles.py` — SRT and VTT parsers.
  - `whisper.py` — faster-whisper (CTranslate2) transcription wrapper, expects 16 kHz mono WAV.
  - `chunking.py` — sentence/paragraph-boundary chunking for LLM calls.

- **AI layer** (`podbook/ai/`)
  - `LLMProvider` ABC with `generate(prompt, system, *, cached_prefix)` interface.
  - Providers: Ollama (local), Claude (with prompt caching), OpenAI, DeepSeek.
  - `cleanup.py` — chunked transcript cleanup; stable context cached across chunks for Claude.
  - `speakers.py` — hybrid speaker labeling: one LLM call to identify speakers from a sample, then heuristic propagation (turn-taking, question/length patterns, alternation fill).
  - `summarize.py` — chapters, key takeaways, summary, glossary; each call uses `cached_prefix` for the shared transcript block.
  - `TokenUsage` tracking on every LLM call; CLI enforces `--max-tokens`; `--dry-run` estimates without spending.

- **Ebook layer** (`podbook/ebook/`)
  - `markdown.py` — canonical markdown from transcript + optional enrichments (chapters, takeaways, summary, glossary, speaker labels).
  - `epub.py` — markdown → EPUB via `ebooklib`; splits on H1 headings into chapters.
  - Dual output when `--cleanup` is used: `*-raw.md` (pre-cleanup) saved alongside the cleaned `*.md`.

- **CLI** (`podbook/cli/main.py`)
  - `podbook build` — full pipeline with flags: `--cleanup`, `--enrich`, `--glossary`, `--speakers`, `--provider`, `--model`, `--max-tokens`, `--dry-run`, `--force-transcribe`.
  - `podbook transcript` — extract and save transcript JSON only.
  - `podbook epub` — generate EPUB from a saved transcript JSON.
  - `podbook cache` — list and clear cached artifacts by type.

- **Tests** — 63/64 passing (`test_heavy_filler` pre-existing failure).
