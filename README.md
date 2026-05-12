# PodBook — Podcast to Ebook Pipeline

Convert podcasts and videos into readable EPUB ebooks, optimized for Boox e-readers and iPad reading apps.

**Philosophy:** transcript-first, AI-enhanced — not AI-first transcription.

```text
podbook build <url>
```

## Development Environment

| Component | Detail |
|---|---|
| OS | Fedora Linux 44 (x86_64) · macOS (Apple Silicon) |
| Python | 3.12+ (developed on 3.14.x) |
| Package manager | uv |
| Transcription | faster-whisper (CTranslate2), base model |
| Local LLM | Ollama — `gemma4:e2b` (7.2 GB); requires `--model gemma4:e2b` flag |
| Cloud LLMs | Claude Haiku 4.5, GPT-4o-mini, DeepSeek |
| System deps | `ffmpeg` — required for audio download/transcription fallback |

## Flow

```mermaid
flowchart TD
    A[INPUT] --> B{source detection}
    B -->|YouTube| C[yt-dlp]
    B -->|Podcast Page| D[requests + BeautifulSoup]
    B -->|RSS Feed| E[feedparser]
    B -->|Local File| F[file metadata]

    C --> G{subtitles available?}
    D --> H{transcript on page?}
    E --> I{audio enclosure?}

    G -->|yes| J[download VTT subs]
    G -->|no| K[download audio]

    H -->|yes| L[extract text]
    H -->|no| M[extract audio URL]

    J --> O[normalize + preprocess]
    K --> N[faster-whisper transcription]
    M --> N
    I --> N
    F --> N

    N --> O
    L --> O

    O --> SP{--speakers or --cleanup?}
    SP -->|yes| SL[speaker labeling LLM call]
    SP -->|no| P
    SL --> P{AI passes?}
    P -->|--cleanup| Q[transcript cleanup]
    P -->|--enrich| R[chapters + takeaways + summary]
    P -->|--glossary| R2[glossary]
    Q --> S[markdown generation]
    R --> S
    R2 --> S
    P -->|skip| S

    S --> T[EPUB generation]
    T --> U[OUTPUT .epub]
```

## Install

```bash
uv sync
uv sync --extra anthropic   # Claude (recommended — includes prompt caching)
uv sync --extra openai      # OpenAI or DeepSeek
uv sync --extra dev         # adds pytest
```

**System dependencies** (required for audio download/transcription fallback):

```bash
# macOS
brew install ffmpeg ollama

# Fedora
sudo dnf install ffmpeg
```

**Ollama model** (if using local LLM):

```bash
ollama pull gemma4:e2b
```

## Usage

### Build an ebook

```bash
podbook build https://www.youtube.com/watch?v=jLFG_FZKbks
podbook build https://example.com/podcast/episode
podbook build ./episode.mp3
```

### AI-enhanced build

```bash
# Clean up filler words and add chapters/summary (Ollama, default provider)
# Ollama defaults to llama3.2 — override with --model for your local model
podbook build --cleanup --enrich --provider ollama --model gemma4:e2b <url>

# Label speakers in the transcript (auto-enabled with --cleanup)
podbook build --speakers <url>
podbook build --speakers --cleanup --provider ollama --model gemma4:e2b <url>

# When --cleanup is used, a *-raw.md is saved alongside the cleaned *.md
# so you can compare pre- and post-cleanup output.

# Use Claude with prompt caching (fastest for long podcasts)
podbook build --cleanup --enrich --provider claude <url>

# Use OpenAI
podbook build --cleanup --enrich --provider openai --model gpt-4o-mini <url>

# Use DeepSeek (requires DEEPSEEK_API_KEY)
podbook build --cleanup --enrich --provider deepseek <url>

# Add a glossary of key terms
podbook build --enrich --glossary --provider claude <url>
```

### Estimate costs before running

```bash
podbook build --dry-run --cleanup --enrich <url>
```

### Transcript only

```bash
# Save transcript as JSON (no EPUB)
podbook transcript <url>
podbook transcript <url> --output ./my-transcript.json
```

### EPUB from an existing transcript

```bash
podbook epub ./my-transcript.json
```

### Cache management

```bash
podbook cache list                       # show all cached artifacts
podbook cache list --output-dir ./out    # specify output directory
podbook cache clear                      # clear all cache
podbook cache clear --type audio         # clear only audio files
podbook cache clear --type transcript    # clear only transcript JSONs
```

### Token budget

```bash
# Hard cap on LLM usage — stops AI passes when budget is reached
podbook build --max-tokens 50000 --cleanup --enrich <url>
```

### Force local transcription

```bash
# Skip subtitle check, always transcribe with faster-whisper
podbook build --force-transcribe <url>
```

## Providers

| `--provider` | API key env var | Notes |
|---|---|---|
| `ollama` | — | Local, free. Default. Requires Ollama running. Use `--model` to select your pulled model (e.g. `--model gemma4:e2b`). |
| `claude` | `ANTHROPIC_API_KEY` | Prompt caching reduces cost on long podcasts. |
| `openai` | `OPENAI_API_KEY` | GPT-4o-mini default. |
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible API, competitive pricing. |

## Dependencies

| Tool | Purpose |
|---|---|
| `yt-dlp` | YouTube audio + subtitle extraction (VTT format with rolling-caption deduplication) |
| `faster-whisper` | Local transcription fallback (CTranslate2, base model) |
| `ffmpeg` | Audio conversion for transcription fallback (system dep, not pip) |
| `ebooklib` | EPUB generation |
| `markdown` | Markdown → HTML conversion for EPUB chapters |
| `typer` | CLI framework |
| `rich` | Terminal output |
| `beautifulsoup4` | Podcast webpage parsing |
| `feedparser` | RSS feed parsing |
| `pydantic` | Immutable data models |

## Project Structure

```text
podbook/
├── cli/main.py              CLI entry point (build, transcript, epub, cache)
├── models.py                Canonical data models (Segment, Transcript, TokenUsage)
├── pipeline.py              End-to-end orchestration
├── sources/
│   ├── youtube.py           YouTube subtitle + audio extraction
│   ├── webpage.py           Podcast page parsing
│   ├── rss.py               RSS feed parsing
│   └── local.py             Local file handling
├── transcript/
│   ├── whisper.py           faster-whisper transcription
│   ├── normalize.py         Segment normalization
│   ├── preprocess.py        Ad/promo/filler classification + filtering
│   └── chunking.py          Sentence-boundary chunking for LLM
├── ai/
│   ├── providers/
│   │   ├── base.py          LLMProvider ABC (cached_prefix interface)
│   │   ├── anthropic.py     Claude with prompt caching
│   │   ├── openai.py        OpenAI + DeepSeek
│   │   └── ollama.py        Local Ollama
│   ├── cleanup.py           Chunked transcript cleanup pass
│   ├── speakers.py          Hybrid LLM + heuristic speaker labeling
│   ├── summarize.py         Chapters, takeaways, summary, glossary
│   └── context.py           Podcast metadata → LLM context builder
└── ebook/
    ├── markdown.py          Markdown generation from transcript + enrichments
    └── epub.py              EPUB generation via ebooklib
tests/
├── test_normalize.py
├── test_preprocess.py
├── test_chunking.py
├── test_markdown.py
└── test_epub.py
```

## Running tests

```bash
uv sync --extra dev
uv run pytest
```
