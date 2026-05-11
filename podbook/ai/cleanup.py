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

CLEANUP_USER = """Clean this transcript segment for readability.

Context:
{context}

Speaker information:
{speaker_context}

Transcript segment:
---
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
    chunks = chunk_by_words(
        transcript.segments,
        target_size=target_chunk_words,
        max_size=max_chunk_words,
    )

    total_usage = TokenUsage()
    cleaned_segments: list[Segment] = []
    shift = 0.0  # track timing shift from text changes

    for i, chunk in enumerate(chunks):
        chunk_text = " ".join(seg.text for seg in chunk)
        prompt = CLEANUP_USER.format(
            context=context,
            speaker_context=speaker_context,
            transcript_text=chunk_text,
        )

        cleaned_text, usage = provider.generate(prompt, system=CLEANUP_SYSTEM)
        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens

        # Map cleaned text back to segments, preserving timing
        cleaned_segs = _text_to_segments(cleaned_text, chunk, shift)
        for seg in cleaned_segs:
            seg_start = seg.start + shift
            seg_end = seg.end + shift
            cleaned_segments.append(
                Segment(start=seg_start, end=seg_end, text=seg.text)
            )

    return cleaned_segments, total_usage


def _text_to_segments(
    cleaned_text: str,
    original_chunk: list[Segment],
    shift: float,
) -> list[Segment]:
    """Convert cleaned text back to segments, distributing time proportionally."""
    if not original_chunk:
        return []

    paragraphs = [p.strip() for p in cleaned_text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        return original_chunk

    total_start = original_chunk[0].start
    total_end = original_chunk[-1].end
    total_duration = total_end - total_start

    # Distribute time proportionally by word count
    para_word_counts = [len(p.split()) for p in paragraphs]
    total_words = sum(para_word_counts) or 1

    segments = []
    current_time = total_start

    for i, para in enumerate(paragraphs):
        proportion = para_word_counts[i] / total_words
        para_duration = total_duration * proportion
        segments.append(
            Segment(
                start=current_time,
                end=current_time + para_duration,
                text=para,
            )
        )
        current_time += para_duration

    return segments
