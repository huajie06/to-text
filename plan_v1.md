# PodBook — Podcast to Ebook Pipeline (v1)

## Goal

Create a low-cost Python CLI application that converts podcasts/video content into readable ebook formats (primarily EPUB) optimized for:

- Boox e-readers
- iPad reading apps

The application should:

1. Prefer existing subtitles/transcripts whenever possible
2. Fall back to local transcription if subtitles do not exist
3. Support both YouTube and non-YouTube podcast sources
4. Improve transcript readability using LLMs
5. Generate ebook-friendly output for long-form reading and learning
6. Remain inexpensive to run and simple to maintain

---

# Core Philosophy

This project is:

> transcript-first, AI-enhanced

NOT:

> AI-first transcription

This distinction matters because it:

- reduces cost
- simplifies architecture
- improves reliability
- minimizes hallucinations
- makes debugging easier

---

# MVP Scope

## Initial CLI Goal

```bash
podbook build <url>
```

Expected behavior:

1. Detect source type
2. Download transcript/subtitles if available
3. Otherwise transcribe audio locally
4. Normalize transcript
5. Improve readability
6. Generate markdown
7. Generate EPUB

---

# Supported Inputs

## Phase 1

### YouTube URLs

Examples:

```text
https://www.youtube.com/watch?v=jLFG_FZKbks
```

Capabilities:

* existing subtitles
* auto-generated subtitles
* audio extraction

---

### Podcast Pages

Examples:

```text
https://blockworks.com/podcast/forwardguidance/298e30d6-4a90-11f1-8866-27ad2c3b315b
```

Capabilities:

* parse webpage
* detect embedded audio
* detect RSS feeds
* download mp3

---

### Local Files

Examples:

```text
podbook build ./episode.mp3
podbook build ./video.mp4
```

---

# Output Formats

## Primary

### Markdown

Intermediate canonical readable format.

---

### EPUB

Primary ebook output.

Best for:

* Boox
* iPad
* Apple Books
* Kobo
* generic readers

---

## Optional Future Outputs

* PDF
* HTML
* MOBI/AZW3
* Obsidian vault export

---

# Recommended Tech Stack

## CLI

### Typer

Reason:

* modern
* clean API
* type-hint friendly
* excellent developer experience

---

# Source Extraction

## yt-dlp

Primary ingestion tool for:

* YouTube
* subtitle download
* audio extraction
* metadata extraction

Capabilities:

* subtitles
* auto subtitles
* audio-only downloads

---

# Web Parsing

## Libraries

### requests

### BeautifulSoup

### feedparser

Use cases:

* RSS parsing
* podcast extraction
* webpage metadata

---

# Transcription Layer

## Preferred Engine

### whisper.cpp (via pywhispercpp)

Replaces: faster-whisper

Reason:

* lower resource usage on CPU
* faster on many hardware configurations
* local execution
* good accuracy for readable transcripts
* C++ backend, Python bindings

---

# Model Recommendation

## Initial Model

```python
model = "base"
```

Or:

```python
model = "small"
```

Do NOT start with large models.

Primary goal:

* readable transcripts
* not legal-grade transcription

---

# Important Product Insight

For ebooks:

## readability > perfect transcription accuracy

Most important improvements:

* punctuation
* paragraphing
* filler word removal
* section organization

Less important:

* exact wording fidelity

---

# Processing Pipeline

```text
INPUT
  ↓
source detection
  ↓
subtitle detection
  ↓
download transcript if available
  ↓
fallback to audio extraction
  ↓
local transcription
  ↓
transcript normalization
  ↓
LLM readability cleanup
  ↓
chapter segmentation
  ↓
summary generation
  ↓
markdown generation
  ↓
EPUB generation
```

---

# Recommended Architecture

```text
podbook/
├── cli/
├── sources/
│   ├── youtube.py
│   ├── rss.py
│   ├── webpage.py
│   └── local.py
│
├── transcript/
│   ├── subtitles.py
│   ├── whisper.py
│   ├── normalize.py
│   └── chunking.py
│
├── ai/
│   ├── cleanup.py
│   ├── summarize.py
│   ├── glossary.py
│   └── providers/
│       ├── base.py
│       ├── openai.py
│       └── ollama.py
│
├── ebook/
│   ├── markdown.py
│   ├── epub.py
│   └── templates/
│
├── cache/
├── outputs/
├── tests/
└── pyproject.toml
```

---

# Transcript Strategy

## Priority Order

### 1. Existing subtitles

Cheapest and fastest.

Sources:

* manual subtitles
* YouTube auto subtitles

Bonus: some YouTube subtitles include speaker labels — capture these when available.

---

### 2. Local transcription

Fallback only when subtitles unavailable.

---

# Canonical Transcript Format

Use structured transcript objects internally.

Example:

```json
[
  {
    "speaker": "Host",
    "start": 120.3,
    "end": 126.9,
    "text": "Inflation remains elevated..."
  }
]
```

This should remain immutable.

---

# Speaker Labeling

## Strategy

Speaker labels dramatically improve readability for interview/discussion formats.

### Tier 1 — Free labels

Some sources provide speaker info already:
- YouTube subtitles occasionally tag speakers
- Podcast RSS feeds sometimes include guest metadata

Capture these whenever available. No extra cost.

### Tier 2 — Basic heuristics

For two-person interviews with clear turn-taking:
- Alternate labels (Speaker A / Speaker B) based on transcript gaps
- Works well enough without ML

### Tier 3 — Diarization (future)

Potential tool: pyannote.audio

Not required for MVP. Evaluate after Tier 1 + Tier 2 are in place.

---

# Chunking Strategy

Chunking is a dedicated feature with its own processing stream. It must not be an afterthought in the LLM cleanup pass.

## Principles

- Never split mid-sentence
- Prefer paragraph/topic boundaries
- 2k–5k word chunks

## Approach

1. Segment transcript into sentences
2. Group sentences into paragraphs (using pause duration + semantic boundaries)
3. Accumulate paragraphs until approaching chunk size limit
4. Cut at the nearest paragraph boundary

## Benefits

* cheaper LLM calls
* more reliable (smaller context = fewer hallucinations)
* easier retries on failure
* chunk-level caching (reprocess one chunk, not the whole transcript)

---

# AI Usage Strategy

## Critical Principle

Use LLMs surgically.

Do NOT:

* regenerate entire transcript
* rewrite everything
* fully summarize source material

This becomes:

* expensive
* hallucination-prone
* less faithful

---

# Recommended LLM Workflow

## Pass 1 — Cleanup

Goal:

* punctuation
* formatting
* filler word removal
* readability

Prompt style:

```text
Clean this transcript for readability.
Preserve meaning.
Remove filler words.
Add punctuation.
Do not summarize.
```

---

## Pass 2 — Enrichment

Generate:

* chapter titles
* summaries
* key takeaways
* glossary terms

---

## Pass 3 — Optional Learning Layer

Future enhancement:

* concept explanations
* macro/finance context
* references
* timelines

---

# Cost Tracking & Token Limits

Token usage must be tracked from day one — not bolted on later.

## During Development / Testing

* Hard token limit per run (configurable, e.g. 50k tokens)
* CLI flag: `--max-tokens 50000`
* Exceeding limit aborts the run with a clear error

## In Production

* Per-episode cost estimate before any LLM call
* Hard cap per episode (configurable)
* Warning threshold (e.g. 80% of cap) — log and surface to user
* Dry-run mode: `podbook build --dry-run <url>` shows what WOULD be spent without spending it

## Tracking

* Token counter for every provider call
* Log: input tokens, output tokens, model, cost
* Accumulate per-run totals in CLI output

---

# Caching Strategy

Cache EVERYTHING.

## Cache Targets

### Downloaded audio

### Subtitle files

### Transcript JSON

### LLM responses (keyed by chunk hash + prompt hash)

### Generated markdown

### Generated EPUB

Benefits:

* lower cost
* faster iteration
* reproducibility

---

# Ebook Structure Recommendation

## Front Matter

```markdown
# Episode Title

Podcast: Forward Guidance
Date: 2026-05-10
Duration: 1h 42m

## Key Takeaways

- ...
- ...
- ...
```

---

# Chapter Structure

```markdown
# Inflation Outlook

(cleaned transcript)

# Fed Policy

(cleaned transcript)
```

---

# Ending Section

```markdown
# Final Summary

...
```

---

# EPUB Generation

## Preferred Tool

### ebooklib

Replaces: pandoc

Reason:

* full control over EPUB structure and CSS
* better styling for Boox/iPad targets
* programmatic chapter/section management
* still simple — no heavy toolchain
* Python-native (no subprocess calls)

pandoc remains a valid fallback for quick one-offs (`podbook epub transcript.md`).

---

# Optional Future Enhancements

## Speaker Diarization

Potential tool:

* pyannote.audio

Goal:

* identify speakers
* improve readability

Not required for MVP.

---

## Optional Future Web UI

Minimal interface only.

Possible stack:

* FastAPI
* HTMX
* Jinja templates

Avoid:

* React complexity
* frontend-heavy architecture

until core pipeline stabilizes.

---

# Recommended MVP Development Order

## Phase 1

### Core Pipeline

* CLI
* source detection
* subtitle extraction
* local transcription
* markdown generation
* EPUB generation

Goal:
working end-to-end pipeline

---

## Phase 2

### Readability Improvements

* transcript cleanup
* chapter segmentation
* summaries

---

## Phase 3

### Better UX

* caching
* progress bars
* config files
* retry handling
* token tracking & cost limits
* dry-run mode

---

## Phase 4

### Optional Intelligence

* glossary
* context explanations
* semantic chaptering
* vector search
* study-guide mode

---

# Things To Avoid Early

Do NOT prematurely add:

* vector databases
* agents
* streaming infrastructure
* distributed workers
* Kubernetes
* fancy web UI
* microservices

The core product value is:

> readable podcast ebooks

Stay focused on that.

---

# Example CLI Commands

## Python Environment

Use `uv` for python environment.

## Build

```bash
podbook build <url>
```

---

## Build from local file

```bash
podbook build ./episode.mp3
```

---

## Build with token limit

```bash
podbook build --max-tokens 50000 <url>
```

---

## Dry run (estimate costs, no LLM calls)

```bash
podbook build --dry-run <url>
```

---

## Generate only transcript

```bash
podbook transcript <url>
```

---

## Generate only EPUB

```bash
podbook epub transcript.md
```

---

# Suggested Dependencies

```text
typer
yt-dlp
pywhispercpp
requests
beautifulsoup4
feedparser
pydantic
rich
jinja2
markdown
ebooklib
```

Optional:

```text
openai
anthropic
ollama
```

---

# Changelog from plan.md → plan_v1.md

| Change | Reason |
|---|---|
| faster-whisper → whisper.cpp (pywhispercpp) | Better CPU performance |
| Chunking elevated to dedicated feature stream | Must not be an afterthought; sentence/paragraph boundary awareness |
| Speaker labeling section added | Free labels from sources + basic heuristics before ML diarization |
| pandoc → ebooklib | Full control over EPUB styling for Boox/iPad |
| Cost tracking & token limits section added | Hard limits in dev, caps + dry-run in production |
| New CLI flags: `--max-tokens`, `--dry-run` | Token budget enforcement |
| Token tracking moved to Phase 3 (UX) | Tracked from day one, polished in Phase 3 |

---

# Long-Term Vision

Potential future capabilities:

* RSS subscription automation
* daily ebook generation
* personalized summaries
* knowledge extraction
* note generation
* spaced repetition exports
* podcast knowledge base

But MVP should remain:

> simple, local-first, low-cost, reliable
```

