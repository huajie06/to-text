# PodBook — Podcast to Ebook Pipeline

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
````

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

### faster-whisper

Reason:

* low cost
* local execution
* CPU compatible
* fast
* accurate enough

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

# Chunking Strategy

Never send entire podcast to LLM at once.

Recommended:

* 2k–5k word chunks

Benefits:

* cheaper
* more reliable
* easier retries
* easier caching

---

# Caching Strategy

Cache EVERYTHING.

## Cache Targets

### Downloaded audio

### Subtitle files

### Transcript JSON

### LLM responses

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

### Pandoc

Workflow:

```bash
pandoc input.md -o output.epub
```

Benefits:

* stable
* simple
* widely supported

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

# Optional Future Web UI

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
faster-whisper
requests
beautifulsoup4
feedparser
pydantic
rich
jinja2
markdown
```

Optional:

```text
openai
anthropic
ollama
```

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