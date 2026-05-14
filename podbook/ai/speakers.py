"""Speaker labeling via hybrid LLM + heuristic approach.

One cheap LLM call identifies speaker identities and patterns from a sample
of utterance groups, then heuristic rules propagate labels to all segments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from podbook.ai.providers.base import LLMProvider, TokenUsage
from podbook.models import Segment, Transcript

SPEAKER_SYSTEM = """You are an expert at identifying speakers in podcast transcripts.
Analyze the provided transcript excerpts and metadata to determine who is speaking.
Return ONLY valid JSON — no preamble, no commentary."""

MAP_SPEAKER_SYSTEM = """You identify podcast speakers from transcript samples and show metadata.
Return ONLY valid JSON — no preamble, no commentary."""


@dataclass
class UtteranceGroup:
    """A group of consecutive segments from the same speaker turn."""

    start: float
    end: float
    text: str
    segment_indices: list[int]


def build_utterance_groups(
    segments: list[Segment],
    max_gap: float = 1.5,
    max_words: int = 200,
) -> list[UtteranceGroup]:
    """Merge consecutive segments into speaker turns.

    Primary heuristic: gap > max_gap seconds → speaker turn boundary.
    Fallback (for back-to-back whisper segments): split when a group
    exceeds max_words, at the nearest sentence boundary.

    Short host interjections can cause spurious boundaries; the
    interjection merge rule in classify_all_groups compensates.
    """
    if not segments:
        return []

    # First pass: split by timing gaps
    raw_groups: list[list[int]] = []
    current_indices: list[int] = []

    for i, seg in enumerate(segments):
        if current_indices and (seg.start - segments[i - 1].end > max_gap):
            raw_groups.append(list(current_indices))
            current_indices = []
        current_indices.append(i)

    if current_indices:
        raw_groups.append(list(current_indices))

    # Second pass: split oversized groups by word count at sentence boundaries
    groups: list[UtteranceGroup] = []

    for indices in raw_groups:
        combined_text = " ".join(segments[i].text for i in indices)
        words = combined_text.split()
        if len(words) <= max_words:
            groups.append(
                UtteranceGroup(
                    start=segments[indices[0]].start,
                    end=segments[indices[-1]].end,
                    text=combined_text,
                    segment_indices=indices,
                )
            )
        else:
            # Split at sentence boundaries within the word budget
            sub_texts: list[str] = []
            sub_indices: list[int] = []
            sub_words = 0

            for i in indices:
                seg_text = segments[i].text
                seg_words = len(seg_text.split())
                # Split before this segment if it would push us well over budget
                # and the previous segment ended with a sentence boundary
                if (
                    sub_texts
                    and sub_words + seg_words > max_words
                    and sub_texts[-1].rstrip().endswith((".", "?", "!"))
                ):
                    groups.append(
                        UtteranceGroup(
                            start=segments[sub_indices[0]].start,
                            end=segments[sub_indices[-1]].end,
                            text=" ".join(sub_texts),
                            segment_indices=list(sub_indices),
                        )
                    )
                    sub_texts = []
                    sub_indices = []
                    sub_words = 0

                sub_texts.append(seg_text)
                sub_indices.append(i)
                sub_words += seg_words

            if sub_texts:
                groups.append(
                    UtteranceGroup(
                        start=segments[sub_indices[0]].start,
                        end=segments[sub_indices[-1]].end,
                        text=" ".join(sub_texts),
                        segment_indices=list(sub_indices),
                    )
                )

    return groups


def extract_sample(
    groups: list[UtteranceGroup],
    sample_size: int = 40,
) -> list[tuple[int, UtteranceGroup]]:
    """Pick representative groups: first 3, last 3, plus evenly-spaced from each third."""
    n = len(groups)
    if n <= sample_size:
        return [(i, g) for i, g in enumerate(groups)]

    sample: list[tuple[int, UtteranceGroup]] = []
    already: set[int] = set()

    for i in range(min(3, n)):
        sample.append((i, groups[i]))
        already.add(i)
    for i in range(max(0, n - 3), n):
        if i not in already:
            sample.append((i, groups[i]))
            already.add(i)

    remaining = sample_size - len(sample)
    per_third = max(1, remaining // 3)

    for third_idx in range(3):
        start = third_idx * n // 3
        end = (third_idx + 1) * n // 3
        candidates = [(i, groups[i]) for i in range(start, end) if i not in already]
        if not candidates:
            continue
        step = max(1, len(candidates) // per_third)
        for j in range(0, len(candidates), step):
            if len(sample) >= sample_size:
                break
            sample.append(candidates[j])

    sample.sort(key=lambda x: x[0])
    return sample[:sample_size]


def _build_speaker_prompt(
    sample: list[tuple[int, UtteranceGroup]],
    transcript: Transcript,
) -> str:
    """Build the prompt for the speaker identification LLM call."""
    context_parts: list[str] = []
    if transcript.source_title:
        context_parts.append(f"Title: {transcript.source_title}")
    if transcript.channel:
        context_parts.append(f"Channel/Show: {transcript.channel}")
    if transcript.description:
        context_parts.append(f"Description: {transcript.description[:300]}")

    excerpts: list[str] = []
    for idx, group in sample:
        text = group.text[:300].replace("\n", " ")
        excerpts.append(f"[{idx}] {text}")

    return f"""Podcast metadata:
{chr(10).join(context_parts) if context_parts else "No metadata available"}

Transcript excerpts (with group indices):
{chr(10).join(excerpts)}

Analyze these excerpts and determine:
1. Who are the speakers? Assign names from the description if possible. Otherwise use "Speaker 1" and "Speaker 2".
2. Which speaker is the host (asks questions, introduces topics, guides conversation)?
3. Which speaker is the guest (gives longer explanations, answers questions)?

Return ONLY a JSON object with this exact structure:
{{
  "speakers": {{
    "speaker_a": "Name or Speaker 1",
    "speaker_b": "Name or Speaker 2"
  }},
  "host": "speaker_a",
  "guest": "speaker_b",
  "assignments": {{ "0": "speaker_a", "1": "speaker_b" }},
  "patterns": {{
    "host_traits": "short description",
    "guest_traits": "short description"
  }}
}}

For assignments, map each group index to either "speaker_a" or "speaker_b".
Only include groups you are confident about — leave out uncertain ones."""


def llm_identify_speakers(
    sample: list[tuple[int, UtteranceGroup]],
    transcript: Transcript,
    provider: LLMProvider,
) -> tuple[dict, TokenUsage]:
    """One LLM call to identify speakers and assign sample groups.

    Returns a dict with keys: speakers, host, guest, assignments, patterns
    and the token usage for the call.
    """
    prompt = _build_speaker_prompt(sample, transcript)
    response_text, usage = provider.generate(prompt, system=SPEAKER_SYSTEM, purpose="speakers")

    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except json.JSONDecodeError:
        data = {}

    return data, usage


def classify_all_groups(
    groups: list[UtteranceGroup],
    speaker_data: dict,
    speaker_names: dict[str, str],
) -> list[str]:
    """Propagate speaker labels to all groups using heuristics.

    Uses anchor propagation, question/length rules, intro/outro defaults,
    alternation, and interjection merging.
    """
    n = len(groups)
    labels: list[str | None] = [None] * n

    host_key = speaker_data.get("host", "speaker_a")
    guest_key = speaker_data.get("guest", "speaker_b")
    host_name = speaker_names.get(host_key, "Host")
    guest_name = speaker_names.get(guest_key, "Guest")

    # Apply LLM assignments as anchors
    assignments: dict[int, str] = {}
    for key, val in speaker_data.get("assignments", {}).items():
        try:
            assignments[int(key)] = val
        except (ValueError, KeyError):
            pass

    for idx, speaker_key in assignments.items():
        if 0 <= idx < n:
            labels[idx] = speaker_names.get(speaker_key, speaker_key)

    # Intro/outro rule: first and last 3 default to host
    for i in range(min(3, n)):
        if labels[i] is None:
            labels[i] = host_name
    for i in range(max(0, n - 3), n):
        if labels[i] is None:
            labels[i] = host_name

    # Anchor propagation: unlabeled groups adjacent to labeled ones
    # without a turn gap get the same speaker. Use 1.5s — same threshold
    # as build_utterance_groups — so propagation fires across all within-turn
    # gaps, not just word-count-split sub-groups.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if labels[i] is not None:
                continue
            if i > 0 and labels[i - 1] is not None:
                if groups[i].start - groups[i - 1].end < 1.5:
                    labels[i] = labels[i - 1]
                    changed = True
                    continue
            if i < n - 1 and labels[i + 1] is not None:
                if groups[i + 1].start - groups[i].end < 1.5:
                    labels[i] = labels[i + 1]
                    changed = True

    # Question rule + length rule for remaining unlabeled
    for i in range(n):
        if labels[i] is not None:
            continue
        text = groups[i].text
        words = len(text.split())
        if words < 30 and text.strip().endswith("?"):
            labels[i] = host_name
        elif words > 50:
            labels[i] = guest_name

    # Interjection merge: short groups (< 5 words) flanked by same speaker
    # get merged into that speaker (catches host backchanneling)
    for i in range(1, n - 1):
        words = len(groups[i].text.split())
        if words < 5 and labels[i - 1] is not None and labels[i + 1] is not None:
            if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
                labels[i] = labels[i - 1]

    # Alternation fill between anchors: handles both same-speaker spans and
    # transition zones (Host-None-Guest). For transitions, pick the nearer
    # neighbor — the unlabeled group likely belongs to whoever it sits closest to.
    for i in range(1, n - 1):
        if labels[i] is not None:
            continue
        prev_label = labels[i - 1]
        next_label = labels[i + 1]
        if prev_label is None or next_label is None:
            continue
        if prev_label == next_label:
            labels[i] = prev_label
        else:
            gap_prev = groups[i].start - groups[i - 1].end
            gap_next = groups[i + 1].start - groups[i].end
            labels[i] = prev_label if gap_prev <= gap_next else next_label

    # Fill remaining with host as safe default
    for i in range(n):
        if labels[i] is None:
            labels[i] = host_name

    return [lbl or host_name for lbl in labels]


def expand_labels(
    segments: list[Segment],
    groups: list[UtteranceGroup],
    group_labels: list[str],
) -> list[Segment]:
    """Map group-level speaker labels back to individual segments."""
    speaker_map: dict[int, str] = {}
    for group, label in zip(groups, group_labels):
        for idx in group.segment_indices:
            speaker_map[idx] = label

    result: list[Segment] = []
    for i, seg in enumerate(segments):
        speaker = speaker_map.get(i, seg.speaker)
        result.append(seg.model_copy(update={"speaker": speaker}))

    return result


def map_speaker_ids(
    transcript: Transcript,
    provider: LLMProvider,
) -> tuple[Transcript, TokenUsage]:
    """Map diarization speaker IDs (SPEAKER_00, SPEAKER_01, ...) to real names.

    Picks the longest utterance per speaker ID, sends them to the LLM with
    podcast metadata, and applies the returned name mapping to all segments.
    Falls back to Host / Guest 1 / Guest 2 when the LLM can't identify names.
    """
    segments = transcript.segments

    longest: dict[str, str] = {}
    for seg in segments:
        if seg.speaker and seg.speaker.startswith("SPEAKER_"):
            if len(seg.text) > len(longest.get(seg.speaker, "")):
                longest[seg.speaker] = seg.text

    if not longest:
        return transcript, TokenUsage()

    speaker_ids = sorted(longest.keys())

    context_parts: list[str] = []
    if transcript.source_title:
        context_parts.append(f"Title: {transcript.source_title}")
    if transcript.channel:
        context_parts.append(f"Channel/Show: {transcript.channel}")
    if transcript.description:
        context_parts.append(f"Description: {transcript.description[:300]}")

    excerpts = "\n".join(f"{spk}: {longest[spk][:400]}" for spk in speaker_ids)
    context_block = "\n".join(context_parts) if context_parts else "No metadata available"

    prompt = f"""{context_block}

Longest utterance per speaker ID:
{excerpts}

Map each speaker ID to a real name using the podcast metadata.
If a name cannot be confidently determined, use "Host" for the first speaker and "Guest" for others.
Return ONLY a JSON object like:
{{"SPEAKER_00": "Joe Rogan", "SPEAKER_01": "Theo Von"}}"""

    response_text, usage = provider.generate(prompt, system=MAP_SPEAKER_SYSTEM)

    name_map: dict[str, str] = {}
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            name_map = json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    if not name_map:
        name_map = {
            spk: ("Host" if i == 0 else f"Guest {i}")
            for i, spk in enumerate(speaker_ids)
        }

    labeled = [
        seg.model_copy(update={"speaker": name_map.get(seg.speaker, seg.speaker)})
        for seg in segments
    ]
    return transcript.model_copy(update={"segments": labeled}), usage


def label_speakers(
    transcript: Transcript,
    provider: LLMProvider,
) -> tuple[Transcript, TokenUsage]:
    """Label speakers in a transcript using hybrid LLM + heuristic approach.

    Returns the transcript with speaker labels populated, and token usage.
    """
    segments = transcript.segments
    if not segments:
        return transcript, TokenUsage()

    groups = build_utterance_groups(segments)

    # Solo podcast detection: a single utterance group → single speaker
    if len(groups) <= 1:
        labeled = [seg.model_copy(update={"speaker": "Host"}) for seg in segments]
        return transcript.model_copy(update={"segments": labeled}), TokenUsage()

    sample = extract_sample(groups)

    speaker_data, usage = llm_identify_speakers(sample, transcript, provider)

    speaker_names = speaker_data.get("speakers", {})
    if not speaker_names:
        speaker_names = {"speaker_a": "Host", "speaker_b": "Guest"}

    group_labels = classify_all_groups(groups, speaker_data, speaker_names)
    labeled_segments = expand_labels(segments, groups, group_labels)

    return transcript.model_copy(update={"segments": labeled_segments}), usage
