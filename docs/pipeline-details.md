# Pipeline Details

## Transcription Strategy

### YouTube source

The pipeline always uses **faster-whisper** (local, CTranslate2, base model) for transcription when the source is a YouTube URL, regardless of subtitle availability. Subtitles are no longer used as a transcript source — the subtitle download step is skipped entirely (`subs=False` on `extract_youtube()`). YouTube metadata (title, channel, description, duration) is still fetched via `yt-dlp --dump-json`.

**Why:** Whisper produces punctuation, casing, and cleaner sentence boundaries, which is critical for downstream LLM passes (cleanup, chapters, takeaways, summary). YouTube subtitles (especially auto-generated) lack punctuation and have timing artifacts.

**Flow:**

```text
yt-dlp --dump-json         → metadata (title, channel, description, duration)
yt-dlp --extract-audio     → download audio as WAV
faster-whisper (base)      → transcribe to segments
normalize + preprocess     → filter to content-only
```

### Transcript cache

The whisper result is cached as `transcript.json` in the source directory (`output/{hash}-{slug}/`). Subsequent runs reuse the cached transcript unless `--force-transcribe` is passed.

The transcript cache is a full `Transcript` pydantic model dump, preserving metadata + segments. It is loaded via `_load_transcript_cache()` at the start of `_extract_transcript()`.

### Audio cache

Downloaded WAV files are cached in the source directory. `download_audio()` checks for existing `*.wav` files before downloading. With `--force-transcribe`, the audio cache is still respected — only the transcript cache is bypassed, the re-transcription reuses the cached WAV.

### Flags

| Flag | Effect |
|---|---|
| `--force-transcribe` | Skip transcript cache, re-download audio (if not cached), re-run whisper |
| (none) | Use cached transcript if available; else YouTube → always whisper |

---

## Speaker Labeling

Two paths exist for speaker labeling, controlled by `--force-diarize` and token availability.

### Path A: Acoustic diarization via pyannote (`--force-diarize`)

Uses `pyannote/speaker-diarization-3.1` (gated model on Hugging Face). Requires:
1. A Hugging Face account with the token granted access to gated repos
2. `HUGGINGFACE_TOKEN` or `HF_TOKEN` set in `.env`
3. `uv sync --extra diarize` installed

**Model repositories (all gated — must accept terms):**
- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0
- https://hf.co/pyannote/speaker-diarization-community-1

**HF token permissions:** If using a fine-grained token, it must have **"Access to public gated repositories"** enabled (https://huggingface.co/settings/tokens). A classic "read" token works by default.

**Diarization output** (`diarize.py`):

Uses pyannote.audio 4.x API:

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token,  # use_auth_token deprecated in 4.x
)
output = pipeline(str(audio_path))
serialized = output.serialize()
# serialized["diarization"] → [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}, ...]
```

**Speaker ID → name mapping** (`map_speaker_ids` in `speakers.py`):

After pyannote assigns `SPEAKER_00`, `SPEAKER_01`, etc., one LLM call to DeepSeek maps IDs to real names by picking the longest utterance per speaker ID and sending it with podcast metadata:

```text
Title: Theo Asks Joe If He Thinks Epstein is Still Alive
Channel/Show: JRE Clips

SPEAKER_00: (longest utterance...)
SPEAKER_01: (longest utterance...)

Map each speaker ID to a real name...
Return {"SPEAKER_00": "Joe Rogan", "SPEAKER_01": "Theo Von"}
```

**Caching:** The diarization result is cached as `diarization.json` in the source directory. Subsequent runs on the same audio skip the ~1x realtime pyannote inference.

### Path B: LLM-only labeling (default when HF token unavailable)

Uses a **hybrid approach** in `label_speakers()` in `speakers.py`:
1. Build utterance groups from timing gaps
2. Extract a representative sample (~40 groups)
3. One LLM call to identify speakers and assign a subset
4. Heuristic propagation (anchor propagation, question/length rules, alternation fill, interjection merge) to label all segments

No audio download needed — works purely on transcript text.

### Known limitation

Pyannote can merge both speakers into a single cluster on short clips or when speakers have similar vocal characteristics. In the test run on a 6m32s JRE clip, pyannote found 87 diarization segments but only SPEAKER_00 was consistently assigned. The LLM-only path correctly identified both Joe Rogan and Theo Von.

---

## Phase Metrics

Each pipeline phase is timed with `time.monotonic()` and logged via the `PhaseMetric` dataclass:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Phase name (e.g. "Transcript Extraction", "Cleanup") |
| `duration_s` | `float` | Wall clock time in seconds |
| `input_tokens` | `int` | LLM input tokens (0 for non-LLM phases) |
| `output_tokens` | `int` | LLM output tokens (0 for non-LLM phases) |
| `items` | `int` | Items produced (segments, chapters, takeaways, etc.) |

### Phases tracked

| Phase | What it includes |
|---|---|
| Transcript Extraction | `_resolve_source_dir` + `_extract_transcript` + `_save_transcript_cache` |
| Preprocessing | Classification (content/ad/promo/filler/meta) + filtering + fractional |
| Speaker Labeling | Diarization (if enabled) + LLM speaker ID mapping or LLM-only labeling |
| Cleanup | Chunked LLM cleanup pass |
| Chapters | LLM chapter generation |
| Takeaways | LLM takeaway generation |
| Summary | LLM summary generation |
| Markdown | Markdown rendering + file write |
| EPUB Generation | EPUB generation via ebooklib |

### Example output

```
                          Pipeline Metrics                           
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃ Phase                 ┃ Duration ┃ Tokens In ┃ Tokens Out ┃ Items ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│ Transcript Extraction │ 0m 30s   │ -         │ -          │ 150   │
│ Preprocessing         │ 0m 0s    │ -         │ -          │ 150   │
│ Speaker Labeling      │ 0m 6s    │ 220       │ 513        │ 1     │
│ Cleanup               │ 0m 11s   │ 2,163     │ 1,856      │ 3     │
│ Chapters              │ 0m 8s    │ 1,996     │ 729        │ 4     │
│ Takeaways             │ 0m 7s    │ 1,970     │ 568        │ 7     │
│ Summary               │ 0m 7s    │ 1,965     │ 479        │ 1     │
│ Markdown              │ 0m 0s    │ -         │ -          │ 1     │
│ EPUB Generation       │ 0m 0s    │ -         │ -          │ 1     │
│ Total                 │ 1m 13s   │           │ 12,459     │       │
└───────────────────────┴──────────┴───────────┴────────────┴───────┘
```

Phase metrics are also serialized to `output/runs.jsonl` under the `phase_metrics` key for programmatic consumption.

---

## LLM Providers

### DeepSeek

Configured via `OpenAIProvider` with a custom base URL.

**`.env` entries:**

```env
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-v4-flash    # optional, defaults to deepseek-chat
```

**Provider instantiation** (`pipeline.py:_get_provider`):

```python
def _get_provider(provider: str, model: str | None):
    if provider == "deepseek":
        from podbook.ai.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
```

**Dependency:** Requires `uv sync --extra openai` for the `openai` Python package.

### Other providers

| `--provider` | Package | Extra | API key env var |
|---|---|---|---|
| `ollama` | `ollama>=0.4` | `--extra ollama` | — (local) |
| `openai` | `openai>=1.0` | `--extra openai` | `OPENAI_API_KEY` |
| `claude` | `anthropic>=0.50` | `--extra anthropic` | `ANTHROPIC_API_KEY` |
| `deepseek` | `openai>=1.0` | `--extra openai` | `DEEPSEEK_API_KEY` |

### Per-call logging

Every LLM call is logged to `output/llm_calls.jsonl` via `log_llm_call()`:

```json
{
  "timestamp": "2026-05-14T22:10:00",
  "provider": "openai",
  "model": "deepseek-v4-flash",
  "purpose": "speakers",
  "input_tokens": 980,
  "output_tokens": 3385,
  "cache_write_tokens": 0,
  "cache_read_tokens": 0,
  "latency_ms": 1234.5,
  "prompt_length": 1234,
  "system_length": 56,
  "response_length": 456
}
```

---

## Authentication Reference

### Hugging Face token for pyannote

```env
# .env
HUGGINGFACE_TOKEN=hf_...    # checked first
HF_TOKEN=hf_...             # fallback
```

**Setup steps:**
1. Create/obtain a token at https://huggingface.co/settings/tokens
2. If using a fine-grained token, enable **"Access to public gated repositories"**
3. Accept terms at:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
   - https://hf.co/pyannote/speaker-diarization-community-1
4. Install: `uv sync --extra diarize`
5. Run with: `--force-diarize`

### DeepSeek API key

```env
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-v4-flash  # optional
```

Install: `uv sync --extra openai`

---

## Dependencies

All dependency groups:

```bash
# Core (always installed)
uv sync

# Optional extras
uv sync --extra openai      # DeepSeek / OpenAI provider
uv sync --extra anthropic   # Claude provider (prompt caching)
uv sync --extra ollama      # Ollama provider
uv sync --extra diarize     # pyannote.audio speaker diarization
uv sync --extra dev         # pytest

# All at once
uv sync --extra openai --extra anthropic --extra diarize --extra dev
```

System dependencies:

```bash
brew install ffmpeg     # macOS (audio conversion)
```

---

## Output Layout

```
output/
├── {hash[:8]}-{slug}/          # per-source cache directory
│   ├── transcript.json         # cached Transcript (pydantic model dump)
│   ├── {title}.wav             # downloaded audio
│   ├── {title}.16k.wav         # resampled to 16kHz for whisper
│   ├── {title}.en.vtt          # YouTube subtitles (from runs before subs=False)
│   └── diarization.json        # cached pyannote result (start, end, SPEAKER_XX)
├── {slug}-{provider}.md        # final markdown
├── {slug}-{provider}.epub      # final ebook
├── runs.jsonl                  # pipeline run log
└── llm_calls.jsonl             # LLM call log
```

---

## File-by-File Source Map

| File | Responsibility |
|---|---|
| `cli/main.py` | CLI entry point, argument parsing, source type detection |
| `pipeline.py` | Pipeline orchestration, phase metrics, provider factory |
| `logging.py` | JSONL logging for runs and LLM calls |
| `models.py` | `Segment`, `Transcript`, `SourceType`, `TokenUsage`, `EbookConfig` |
| `sources/youtube.py` | YouTube metadata fetch (`yt-dlp --dump-json`), audio download (`--extract-audio`) |
| `sources/webpage.py` | Podcast webpage transcript extraction |
| `sources/rss.py` | RSS feed parsing |
| `sources/local.py` | Local audio/video file handling |
| `transcript/whisper.py` | faster-whisper transcription (CTranslate2, base model) |
| `transcript/subtitles.py` | VTT subtitle parsing |
| `transcript/normalize.py` | Segment normalization (text cleaning, dedup, merge) |
| `transcript/preprocess.py` | Segment classification (content/ad/promo/filler/meta) + filtering |
| `transcript/chunking.py` | Sentence-boundary-aware chunking for LLM context window |
| `transcript/diarize.py` | pyannote acoustic diarization wrapper (4.x API) |
| `ai/providers/base.py` | `LLMProvider` ABC with `cached_prefix` interface |
| `ai/providers/openai.py` | OpenAI-compatible provider (OpenAI + DeepSeek) |
| `ai/providers/anthropic.py` | Claude provider with prompt caching |
| `ai/providers/ollama.py` | Ollama local provider |
| `ai/cleanup.py` | Chunked transcript cleanup LLM pass |
| `ai/speakers.py` | Speaker labeling: hybrid LLM+heuristic and diarization ID mapping |
| `ai/summarize.py` | Chapters, takeaways, summary generation |
| `ai/context.py` | Podcast metadata → LLM context builder |
| `ebook/markdown.py` | Transcript + enrichments → markdown |
| `ebook/epub.py` | Markdown → EPUB via ebooklib |
