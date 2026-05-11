"""Transcript cleanup — chunked LLM pass for readability."""

from __future__ import annotations

from podbook.ai.context import build_context, build_speaker_context
from podbook.ai.providers.base import LLMProvider, TokenUsage
from podbook.models import Segment, Transcript
from podbook.transcript.chunking import chunk_by_words

CLEANUP_SYSTEM = """You are a meticulous transcript editor. Your job is to clean up podcast transcripts for ebook reading while preserving all meaning and nuance.

Rules:
- Remove filler words (um, uh, you know, like, I mean, sort of, kind of, right?, okay?)
- Remove false starts and stutters (repeated words/phrases at sentence starts)
- Add proper punctuation where missing
- Break into paragraphs at topic shifts
- Keep ALL factual statements, opinions, data points, and examples
- Do NOT summarize, shorten, or paraphrase
- Do NOT change the speaker's tone or style
- Do NOT add any commentary, analysis, or new content
- Preserve proper names, numbers, and technical terms exactly

Output the cleaned transcript only. No preamble, no commentary."""

# Stable per-run prefix — cached by Claude across all chunk calls
CLEANUP_PREFIX = """Clean this transcript segment for readability.

Context:
{context}

Speaker information:
{speaker_context}

Transcript segment:"""

# Variable per-chunk suffix — not cached
CLEANUP_SUFFIX = """---
{transcript_text}
---

Return the cleaned transcript, preserving all meaning and detail."""


def cleanup_transcript(
    transcript: Transcript,
    provider: LLMProvider,
    *,
    target_chunk_words: int = 3000,
    max_chunk_words: int = 5000,
) -> tuple[list[Segment], TokenUsage]:
    """Clean up a full transcript by processing it in chunks.

    Returns cleaned segments and total token usage.
    """
    context = build_context(transcript)
    speaker_context = build_speaker_context(transcript)

    # Build the stable prefix once — Claude caches this across all chunk calls
    cached_prefix = CLEANUP_PREFIX.format(context=context, speaker_context=speaker_context)

    chunks = chunk_by_words(
        transcript.segments,
        target_size=target_chunk_words,
        max_size=max_chunk_words,
    )

    total_usage = TokenUsage()
    cleaned_segments: list[Segment] = []

    for chunk in chunks:
        chunk_text = " ".join(seg.text for seg in chunk)
        prompt = CLEANUP_SUFFIX.format(transcript_text=chunk_text)

        cleaned_text, usage = provider.generate(
            prompt,
            system=CLEANUP_SYSTEM,
            cached_prefix=cached_prefix,
        )
        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens
        total_usage.cache_write_tokens += usage.cache_write_tokens
        total_usage.cache_read_tokens += usage.cache_read_tokens

        cleaned_segments.extend(_text_to_segments(cleaned_text, chunk))

    return cleaned_segments, total_usage


def _text_to_segments(
    cleaned_text: str,
    original_chunk: list[Segment],
) -> list[Segment]:
    """Convert cleaned text back to segments, distributing time proportionally.

    Speaker is inferred by finding which original segment's time range contains
    the midpoint of each cleaned paragraph.
    """
    if not original_chunk:
        return []

    paragraphs = [p.strip() for p in cleaned_text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        return list(original_chunk)

    total_start = original_chunk[0].start
    total_end = original_chunk[-1].end
    total_duration = total_end - total_start

    para_word_counts = [len(p.split()) for p in paragraphs]
    total_words = sum(para_word_counts) or 1

    segments: list[Segment] = []
    current_time = total_start

    for i, para in enumerate(paragraphs):
        proportion = para_word_counts[i] / total_words
        para_duration = total_duration * proportion
        midpoint = current_time + para_duration / 2
        speaker = _speaker_at(midpoint, original_chunk)
        segments.append(
            Segment(
                start=current_time,
                end=current_time + para_duration,
                speaker=speaker,
                text=para,
            )
        )
        current_time += para_duration

    return segments


def _speaker_at(time: float, segments: list[Segment]) -> str | None:
    """Return the speaker active at a given timestamp within a segment list."""
    for seg in segments:
        if seg.start <= time < seg.end:
            return seg.speaker
    return segments[-1].speaker if segments else None
