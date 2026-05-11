"""Enrichment pass — chapter segmentation, key takeaways, and summary."""

from __future__ import annotations

from podbook.ai.context import build_context
from podbook.ai.providers.base import LLMProvider, TokenUsage
from podbook.models import Chapter, Segment, Transcript
from podbook.transcript.chunking import chunk_by_segments

CHAPTER_SYSTEM = """You are an expert editor who organizes podcast transcripts into logical chapters.

Given a cleaned transcript from a podcast, identify 4-8 natural chapter breaks based on topic shifts. For each chapter:
1. Write a short, descriptive title (5-8 words)
2. Note which topics are covered

Output format:
CHAPTER: <title>
TOPICS: <comma-separated topics>
---
CHAPTER: <title>
TOPICS: <comma-separated topics>

Only output the chapter list. No preamble."""

CHAPTER_USER = """Identify the major chapter breaks in this podcast transcript.

Context:
{context}

Transcript:
---
{transcript_text}
---"""

TAKEAWAY_SYSTEM = """You extract concise, memorable key takeaways from podcast transcripts.

Rules:
- Produce 5-8 takeaways
- Each takeaway should be 1-2 sentences
- Focus on actionable insights, surprising facts, or core arguments
- Be specific — include names, numbers, or examples when present
- Do not editorialize or add opinions

Output one takeaway per line, starting with a dash."""

TAKEAWAY_USER = """Extract the key takeaways from this transcript.

Context:
{context}

Transcript:
---
{transcript_text}
---"""

SUMMARY_SYSTEM = """You write concise, informative summaries of podcast episodes.

Write a 3-4 paragraph summary that:
1. States the main topic and who is speaking
2. Covers the core arguments or discussion points in order
3. Highlights the most important conclusions or insights
4. Uses specific names, numbers, and examples from the episode

Write in clear, journalistic prose. No preamble."""

SUMMARY_USER = """Summarize this podcast episode.

Context:
{context}

Transcript:
---
{transcript_text}
---"""

GLOSSARY_SYSTEM = """You extract key terms, concepts, and acronyms from podcast transcripts and write brief definitions based on how they are used in context.

Output one term per line:
TERM: brief 1-sentence definition

Include:
- Technical terms or jargon
- Acronyms and abbreviations
- People mentioned with significant roles
- Companies or products discussed"""

GLOSSARY_USER = """Extract key terms and concepts from this transcript that would benefit from glossary entries.

Context:
{context}

Transcript:
---
{transcript_text}
---"""


def generate_chapters(
    transcript: Transcript,
    provider: LLMProvider,
    segments_per_chunk: int = 200,
) -> tuple[list[Chapter], TokenUsage]:
    """Generate chapter titles from a transcript.

    Processes transcript in large chunks (200+ segments each) to
    identify topic shifts, then returns a list of Chapters.
    """
    context = build_context(transcript)
    total_usage = TokenUsage()
    all_chapters: list[Chapter] = []

    # Use large chunks for chapter detection
    chunks = chunk_by_segments(
        transcript.segments,
        target_segments=segments_per_chunk,
        max_segments=segments_per_chunk * 2,
    )

    for chunk in chunks:
        chunk_text = "\n\n".join(seg.text for seg in chunk)
        text, usage = provider.generate(
            CHAPTER_USER.format(context=context, transcript_text=chunk_text),
            system=CHAPTER_SYSTEM,
        )
        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens

        for line in text.strip().split("\n"):
            if line.startswith("CHAPTER:"):
                title = line[len("CHAPTER:"):].strip()
                all_chapters.append(Chapter(title=title, segments=[]))

    return all_chapters, total_usage


def generate_takeaways(
    transcript: Transcript,
    provider: LLMProvider,
) -> tuple[list[str], TokenUsage]:
    """Generate key takeaways from the full transcript.

    Sends the entire transcript in one call — assumes it fits in context
    or the model has sufficient context window (most modern models do).
    """
    context = build_context(transcript)
    transcript_text = "\n\n".join(seg.text for seg in transcript.segments)

    text, usage = provider.generate(
        TAKEAWAY_USER.format(context=context, transcript_text=transcript_text),
        system=TAKEAWAY_SYSTEM,
    )

    takeaways = [
        line.lstrip("- ").strip()
        for line in text.strip().split("\n")
        if line.strip().startswith("-")
    ]
    return takeaways, usage


def generate_summary(
    transcript: Transcript,
    provider: LLMProvider,
) -> tuple[str, TokenUsage]:
    """Generate a final summary of the episode."""
    context = build_context(transcript)
    transcript_text = "\n\n".join(seg.text for seg in transcript.segments)

    text, usage = provider.generate(
        SUMMARY_USER.format(context=context, transcript_text=transcript_text),
        system=SUMMARY_SYSTEM,
    )
    return text.strip(), usage


def generate_glossary(
    transcript: Transcript,
    provider: LLMProvider,
) -> tuple[dict[str, str], TokenUsage]:
    """Generate a glossary of key terms from the transcript."""
    context = build_context(transcript)
    transcript_text = "\n\n".join(seg.text for seg in transcript.segments)

    text, usage = provider.generate(
        GLOSSARY_USER.format(context=context, transcript_text=transcript_text),
        system=GLOSSARY_SYSTEM,
    )

    glossary: dict[str, str] = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            term, _, definition = line.partition(":")
            glossary[term.strip()] = definition.strip()
    return glossary, usage
