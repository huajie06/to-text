# Examples

Reference video: [Theo Asks Joe If He Thinks Epstein is Still Alive](https://www.youtube.com/watch?v=uAScbbOpmTI) — 6m32s, JRE Clips.

| Example | Focus |
|---|---|
| [example1/](./example1) | LLM model comparison: `gemma4:e2b` vs `deepseek-v4-flash` |
| [example2/](./example2) | Speaker diarization: pyannote + any-overlap merge |

---

## example2: Speaker Diarization with Any-Overlap Merge

Transcript extracted via **faster-whisper** (base model, `--force-transcribe`), speakers identified via **pyannote acoustic diarization** (`pyannote/speaker-diarization-3.1`), speaker IDs merged to segments via **any-overlap alignment** with multi-speaker labels for overlap regions.

**Pipeline run:** `--provider deepseek --cleanup --enrich --force-transcribe --force-diarize`

### File flow

```text
YouTube URL
  ├─ yt-dlp (metadata: title, channel, description)
  └─ yt-dlp --extract-audio → WAV
       └─ faster-whisper (base)
            └─ transcript.json           ← raw whisper segments (no speakers)
                 └─ pyannote diarization
                      ├─ diarization.json ← (start, end, SPEAKER_XX) tuples
                      └─ assign_speakers() any-overlap merge
                           └─ diarization_transcript.json  ← segments tagged with speaker IDs
                                └─ map_speaker_ids() LLM → real names
                                     └─ cleanup + enrich → final markdown + EPUB
```

### Any-overlap merge strategy

Instead of max-overlap (which loses short interjections), each segment is checked independently against each speaker's diarization windows:

- Overlaps only `SPEAKER_00` → tagged `SPEAKER_00`
- Overlaps only `SPEAKER_01` → tagged `SPEAKER_01`
- Overlaps **both** → tagged `SPEAKER_00_SPEAKER_01` (ambiguous interjection zone)
- Overlaps neither → defaults to longest-duration speaker

### Metrics

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

### Speaker distribution

| Segments | Label |
|---|---|
| 104 | `SPEAKER_00` (pure Joe Rogan) |
| 46 | `SPEAKER_00_SPEAKER_01` (both — Theo interjecting during Joe's turn) |

No segments were purely `SPEAKER_01` — Theo's interjections are always embedded inside Joe's longer speaking windows.

### Files in this directory

| File | Description |
|---|---|
| `transcript.json` | Raw whisper transcription (150 segments, no speakers) |
| `diarization.json` | Pyannote diarization output (87 windows, SPEAKER_00 + SPEAKER_01) |
| `diarization_transcript.json` | Any-overlap merged segments with speaker labels |
| `theo-asks-joe-if-he-thinks-epstein-is-still-alive-deepseek.md` | Final cleaned + enriched markdown |
| `theo-asks-joe-if-he-thinks-epstein-is-still-alive-deepseek.epub` | Final ebook |

---

## example1: Local vs. Cloud LLM

Reference: same video, 165 raw segments (YouTube subtitles), 392s duration.

### Models compared

| | Local | Cloud |
|---|---|---|
| Provider | Ollama | DeepSeek |
| Model | `gemma4:e2b` | `deepseek-v4-flash` |
| Size | ~7.2 GB | N/A (API) |
| Cost | Free | ~$0.005/run |

### Token usage & latency

| Pass | gemma4 in/out | deepseek in/out | gemma4 latency | deepseek latency |
|---|---|---|---|---|
| Speakers | 1,034 / 1,178 | 995 / 4,033 | 95s | 49s |
| Cleanup | 2,867 / 1,407 | 2,785 / 5,336 | 139s | 56s |
| Chapters | 452 / 606 | 2,004 / 492 | 41s | 10s |
| Takeaways | 420 / 533 | 1,978 / 356 | 36s | 7s |
| Summary | 416 / 691 | 1,973 / 501 | 45s | 10s |
| **Total** | **5,189 / 4,415** | **9,735 / 10,718** | **~6 min** | **~2 min** |

DeepSeek uses more output tokens because it produces proper, detailed output. gemma4 returns short hallucinated text. DeepSeek is 3x faster despite being a cloud API round-trip.

### Quality

| Aspect | gemma4:e2b | deepseek-v4-flash |
|---|---|---|
| Speaker names | "Speaker 1, Speaker 2" | Joe Rogan, Theo Von |
| Content fidelity | **Hallucinated** — generic filler replaces all real content | All factual content preserved verbatim |
| Chapters | "Reflecting on the current situation" (made up) | "Is Epstein Alive? Assassination Theory" (real) |
| Takeaways | 7 generic platitudes | 5 specific, accurate points with names/facts |
| Summary | Describes events that don't exist in output | Accurate, detailed, references specific quotes |

#### Example: same cleanup instruction, same input transcript

**Raw transcript (ground truth):**
> "They put him in a cell with Epstein. And Epstein got strangled. He was found guilty of killing four men."

**gemma4:e2b output:**
> "It's wild. And it's messed up."

**deepseek-v4-flash output:**
> "They put him in a cell with Epstein. And Epstein got strangled. Bro, he was found guilty of killing four men."

### Cost

DeepSeek v4 flash pricing (~$0.14/M input, ~$0.28/M output):
- 6-minute video: **~$0.005**
- 1-hour podcast (est. 10x tokens): **~$0.05**
- 2-hour podcast (est. 20x tokens): **~$0.10**

### Decision matrix

| Scenario | Recommendation |
|---|---|
| Dry runs / testing pipeline logic | gemma4:e2b (free) |
| Speaker labeling only | gemma4:e2b (handles it fine) |
| Cleanup / enrich / final output | Cloud model required |
| Batch processing 50+ episodes | Cloud, budget ~$2-5 total |

### Root cause of gemma4:e2b failure

The cleanup task requires the model to read ~3,000 words of messy, fragmented auto-captions (with `>>` markers, `[__]` profanity filters, timing artifacts) and restructure it into clean paragraphs while preserving every factual statement. This is a complex NLU task — it demands strong instruction-following and long-context coherence. A 7.2GB model lacks the capacity to do this reliably. It collapses into generating vague paraphrases instead of preserving the transcript's actual content.

The local model works for the speaker identification task (~1,000 tokens, simple classification + JSON output) where the quality difference is minor.

### Files in this directory

| File | Description |
|---|---|
| `source.vtt` | Raw YouTube auto-generated subtitles (73K) |
| `transcript.json` | Parsed + normalized transcript as JSON (165 segments) |
| `output-gemma4.md` | Cleaned output from local `gemma4:e2b` (hallucinated) |
| `output-deepseek.md` | Cleaned output from `deepseek-v4-flash` (accurate) |

---

# LLM Prompts

Each AI pass sends a **system prompt** + **cached prefix** + **prompt suffix** to the LLM.
The `cached_prefix` is stable across calls in the same run (Claude caches it; other providers prepend it).

## Step 1.5: Speaker Labeling

**System prompt** (`speakers.py:16-18`):

```
You are an expert at identifying speakers in podcast transcripts.
Analyze the provided transcript excerpts and metadata to determine who is speaking.
Return ONLY valid JSON — no preamble, no commentary.
```

**Prompt** (built from `speakers.py:160-203`):

```
Podcast metadata:
Title: {source_title}
Channel/Show: {channel}
Description: {description[:300]}

Transcript excerpts (with group indices):
[0] the Joe Rogan Experience.
[1] There's no way I can look at Chuck Schumer and think he's a good guy.
...

Analyze these excerpts and determine:
1. Who are the speakers?
2. Which speaker is the host?
3. Which speaker is the guest?

Return ONLY a JSON object with this exact structure:
{
  "speakers": { "speaker_a": "Name", "speaker_b": "Name" },
  "host": "speaker_a",
  "guest": "speaker_b",
  "assignments": { "0": "speaker_a", "1": "speaker_b" },
  "patterns": { "host_traits": "...", "guest_traits": "..." }
}
```

The sample is ~15 utterance groups, each trimmed to 300 chars. The LLM never sees the full transcript — just the excerpts.

## Step 2a: Transcript Cleanup

**System prompt** (`cleanup.py:10-23`):

```
You are a meticulous transcript editor. Your job is to clean up podcast
transcripts for ebook reading while preserving all meaning and nuance.

Rules:
- Remove filler words (um, uh, you know, like, I mean, sort of, kind of, right?, okay?)
- Remove false starts and stutters
- Add proper punctuation where missing
- Break into paragraphs at topic shifts
- Keep ALL factual statements, opinions, data points, and examples
- Do NOT summarize, shorten, or paraphrase
- Do NOT change the speaker's tone or style
- Do NOT add any commentary, analysis, or new content
- Preserve proper names, numbers, and technical terms exactly

Output the cleaned transcript only. No preamble, no commentary.
```

**Cached prefix** (`cleanup.py:26-34`):

```
Clean this transcript segment for readability.

Context:
{built from transcript metadata, title, channel, duration, description}

Speaker information:
{speaker hints from description}

Transcript segment:
```

**Prompt suffix** (`cleanup.py:37-41`):

```
---
{transcript_text}
---

Return the cleaned transcript, preserving all meaning and detail.
```

The transcript is split into ~3,000 word chunks (sentence-boundary aware).

## Step 2b: Chapters

**System prompt** (`summarize.py:10-23`):

```
You are an expert editor who organizes podcast transcripts into logical chapters.

Given a cleaned transcript from a podcast, identify 4-8 natural chapter breaks
based on topic shifts. For each chapter:
1. Write a short, descriptive title (5-8 words)
2. Note which topics are covered

Output format:
CHAPTER: <title>
TOPICS: <comma-separated topics>
---
CHAPTER: <title>
TOPICS: <comma-separated topics>

Only output the chapter list. No preamble.
```

**Cached prefix**: context + transcript chunk

**Prompt suffix** (`summarize.py:25`):

```
Identify the major chapter breaks in this transcript chunk.
```

## Step 2c: Key Takeaways

**System prompt** (`summarize.py:27-36`):

```
You extract concise, memorable key takeaways from podcast transcripts.

Rules:
- Produce 5-8 takeaways
- Each takeaway should be 1-2 sentences
- Focus on actionable insights, surprising facts, or core arguments
- Be specific — include names, numbers, or examples when present
- Do not editorialize or add opinions

Output one takeaway per line, starting with a dash.
```

**Cached prefix**: context + full cleaned transcript

**Prompt suffix** (`summarize.py:38`):

```
Extract the key takeaways from this transcript.
```

## Step 2d: Final Summary

**System prompt** (`summarize.py:40-49`):

```
You write concise, informative summaries of podcast episodes.

Write a 3-4 paragraph summary that:
1. States the main topic and who is speaking
2. Covers the core arguments or discussion points in order
3. Highlights the most important conclusions or insights
4. Uses specific names, numbers, and examples from the episode

Write in clear, journalistic prose. No preamble.
```

**Cached prefix**: context + full cleaned transcript

**Prompt suffix** (`summarize.py:50`):

```
Summarize this podcast episode.
```

## Prompt Assembly

All providers follow the same structure:

```
[System prompt]              ← STATIC (cached by Claude as system block)
[Cached prefix]              ← STABLE per-run (context + full transcript)
[Prompt suffix]              ← VARIABLE ("Clean this", "Summarize", etc.)
```

For **OpenAI/DeepSeek/Ollama**: `cached_prefix` is prepended to the prompt before sending (no caching).

For **Claude**: `cached_prefix` is a separate content block with `cache_control`, reused across all chunk calls in the same run.
