# Speaker Labeling Implementation Plan

## Context

The `Segment.speaker` field exists in the data model but is never populated — all output renders as bare text with no indication of who is speaking. For a 2-person interview podcast (The Diary Of A CEO: Steven Bartlett + Ben Felix), this makes the ebook hard to follow. The goal: at minimum "Person 1 / Person 2", ideally "Host / Guest" or actual names.

The pipeline already has:
- Rich YouTube metadata (title, channel, description with speaker names)
- `ai/context.py` functions that extract speaker hints from descriptions
- Markdown rendering that checks `seg.speaker` and renders `**Speaker:** text`
- An LLM provider (Gemma 4B) already used for cleanup + enrichment

The plan_v1.md calls this "Tier 2 — Basic heuristics for 2-person interviews." The approach: one cheap LLM call to identify who's who, then heuristic turn-taking for the rest.

## Approach

**Hybrid: 1 LLM call to identify speaker identities + patterns, then heuristic label propagation for all remaining segments.**

The LLM analyzes ~40 representative utterance groups to determine:
- Who are the speakers? (names from metadata, or "Speaker 1"/"Speaker 2")
- Who asks questions? (host)
- Who gives long explanations? (guest)

Then heuristic rules propagate labels to all ~1156 segments using turn-taking, question detection, and anchor propagation.

## Cost

| Operation | Tokens |
|---|---|
| LLM speaker identification (1 call) | ~2,500 |
| Heuristic propagation | 0 |
| **Total** | **~2,500** |

~15% overhead on the cleanup pass.

## Implementation Steps

### 1. New file: `podbook/ai/speakers.py` (~250 lines)

Core module with these functions:

- **`build_utterance_groups(segments, max_gap=1.5)`** — merge consecutive segments into speaker turns based on inter-segment gaps. A gap > 1.5s between segments is a strong turn indicator. Reduces ~1156 segments to ~200-400 utterance groups.

- **`extract_sample(groups, sample_size=40)`** — pick representative groups: first 3 (intro, always host), last 3 (outro, always host), plus evenly-spaced diverse groups from each third of the transcript. Total: ~40 groups, ~2000-3000 input tokens.

- **`llm_identify_speakers(sample, transcript, provider)`** — one LLM call. Sends context (title, channel, description, speaker name hints) + 40 sample groups. The LLM returns:
  - Speaker identities/roles
  - Per-excerpt speaker assignment
  - Distinguishing patterns (who asks questions, who gives long answers)

- **`classify_all_groups(groups, sample_labels, speaker_names)`** — propagate labels using:
  - **Anchor propagation**: unlabeled groups adjacent to labeled groups without turn boundaries get same speaker
  - **Question rule**: short groups ending in "?" → host
  - **Length rule**: long groups (>50 words) without questions → guest
  - **Intro/outro rule**: first and last 3 groups default to host
  - **Alternation**: between anchors, speakers alternate

- **`expand_labels(segments, labeled_groups)`** — map group labels back to individual segments

- **`label_speakers(transcript, provider)`** — main entry point

### 2. Fix: `podbook/ai/cleanup.py` — preserve speaker field

Two locations drop the `speaker` field:

- **`_text_to_segments()`** — new `Segment(...)` omits `speaker`. Fix: use time-window overlap majority vote to determine speaker per paragraph.

- **`cleanup_transcript()` assembly loop** — pass `speaker=seg.speaker` through when creating new Segment objects.

### 3. Fix: `podbook/transcript/chunking.py` — preserve speaker field

`_build_segment_chunk()` creates new `Segment` without `speaker`. Fix: carry forward speaker from any constituent segment that has one.

### 4. Edit: `podbook/pipeline.py` — add Phase 1.5

Insert speaker labeling after preprocessing/filtering, before cleanup. Auto-enable when `--cleanup` is used. Add `label_speakers` parameter. Print detected speakers to console.

### 5. Edit: `podbook/cli/main.py` — add `--speakers` flag

New CLI flag `--speakers` to explicitly request speaker labeling. Also auto-enabled when `--cleanup` is used.

### 6. Edit: `podbook/ai/context.py` — enhance regex

Broaden `build_speaker_context()` regex patterns to match more name formats and common podcast description patterns.

## Edge Cases

| Case | Handling |
|---|---|
| Solo podcast | All segments get same speaker |
| 3+ speakers | Alternation degrades, groups without strong features stay unlabeled |
| No metadata | LLM falls back to "Speaker 1" / "Speaker 2" based on conversation patterns |
| Overlapping speech | Time-window majority vote in cleanup reconstruction |
| Non-English | Language-agnostic — relies on turn-taking + question marks |
| Dry run | Cost estimate includes speaker labeling token estimate |
| No segments after filtering | Speaker labeling skipped, returns early |

## Verification

```bash
# Test with 1/8 transcript, speakers only
podbook build "https://www.youtube.com/watch?v=jLFG_FZKbks" \
  --output output/speaker-test \
  --speakers --fraction 0.125

# Test with speakers + cleanup + enrich
podbook build "https://www.youtube.com/watch?v=jLFG_FZKbks" \
  --output output/speaker-test-v2 \
  --speakers --cleanup --enrich --fraction 0.125

# Verify speakers appear in markdown
grep -E '\*\*.*:\*\*' output/speaker-test/*.md | head -20
```
