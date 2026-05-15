# Changelog

## 2026-05-14

### Added

- **YouTube always uses faster-whisper** (`podbook/sources/youtube.py`, `podbook/pipeline.py`)
  - YouTube subtitles no longer used as a transcript source. `extract_youtube()` accepts `subs=False` to skip subtitle download. Pipeline always downloads audio + whisper-transcribes.
  - Metadata (title, channel, description) still fetched via `yt-dlp --dump-json`.

- **Phase metrics** (`podbook/pipeline.py`)
  - `PhaseMetric` dataclass tracks `(name, duration_s, input_tokens, output_tokens, items)` per pipeline phase.
  - All phases (transcript extraction, preprocessing, speaker labeling, cleanup, chapters, takeaways, summary, markdown, EPUB) timed with `time.monotonic()`.
  - Rich summary table printed at end of run showing duration + token counts per phase.
  - Phase metrics serialized to `output/runs.jsonl`.

- **Acoustic speaker diarization** (`podbook/transcript/diarize.py`)
  - Updated to pyannote.audio 4.x API: `token=` parameter instead of `use_auth_token=`, `serialize()` method instead of `itertracks()`.
  - `assign_speakers()` rewritten with **any-overlap alignment**: checks each diarization speaker independently for any time overlap with each whisper segment. Single match → tag, multiple matches → combined label (`SPEAKER_00_SPEAKER_01`), no match → longest-duration fallback.
  - Replaces old max-overlap strategy which lost short-interjection speakers.

- **Combined speaker label handling** (`podbook/ai/speakers.py`)
  - `_parse_speaker_ids()` extracts individual IDs from both simple (`SPEAKER_00`) and combined (`SPEAKER_00_SPEAKER_01`) labels.
  - `_resolve_speaker_label()` maps combined labels through name map → combined names (`Joe_Rogan_Theo_Von`).

- **Standalone merge script** (`scripts/merge_diarization.py`)
  - Reads `transcript.json` + `diarization.json` from a source directory, applies any-overlap alignment, writes `diarization_transcript.json`.

- **Documentation**
  - `docs/pipeline-details.md` — comprehensive pipeline reference: transcription strategy, diarization setup, phase metrics, LLM providers, auth setup, output layout, source map.
  - `plan-idea/diarization-merge-enhancements.md` — three enhancement paths: boundary splitting, embedding clustering, VAD segmentation.
  - `examples/example1/` and `examples/example2/` — reorganized output references with updated README index.
  - `examples/README.md` updated with example2 (diarization) details and metrics table.

- **Dependency extras**
  - `uv sync --extra diarize` for pyannote.audio.
  - `uv sync --extra openai` for DeepSeek/OpenAI provider (was missing from install instructions).

### Changed

- **README.md**: flow diagram updated (YouTube → direct whisper, diarization path added), philosophy updated to "whisper-first", new "Speaker Diarization" section with merge logic and potential enhancements.
- **AGENTS.md**: YouTube subtitle preference → always whisper, diarization merge logic, potential enhancements, expanded output layout, updated known issues with diarization notes.

---

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

- **Structured logging** (`podbook/logging.py`)
  - `output/runs.jsonl` — pipeline run log (timestamp, source, duration, tokens, status, output paths).
  - `output/llm_calls.jsonl` — LLM call log (timestamp, provider/model, purpose, token counts, latency_ms, prompt/response lengths).
  - All LLM providers now time API calls and emit log entries automatically.
  - Pipeline logs a success entry on completion.

- **`purpose` parameter on `LLMProvider.generate()`**
  - Each AI pass call site passes `purpose=` (speakers, cleanup, chapters, takeaways, summary, glossary) for log traceability.

- **`.env` auto-loading** via `python-dotenv`
  - `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, and other API keys are loaded automatically — no manual `export` needed.

- **DeepSeek model from environment**
  - `DEEPSEEK_MODEL` in `.env` overrides the default model (`deepseek-chat`). Base URL set to `https://api.deepseek.com`.

- **Provider suffix in output filenames**
  - Markdown and EPUB files are named `{title}-{provider}.md` and `{title}-{provider}.epub` for easy differentiation between runs with different LLMs.

- **OpenAI provider `extra_body` support**
  - Constructor accepts optional `extra_body` dict for provider-specific API parameters (e.g., DeepSeek thinking mode).

- **`examples/` directory**
  - Reference comparison: raw VTT, transcript JSON, gemma4 vs deepseek outputs, and README with full prompt catalog and model comparison data.

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
