"""Markdown generation from transcript segments."""

from __future__ import annotations

from datetime import datetime

from podbook.models import Chapter, Segment, Transcript


def generate_markdown(
    transcript: Transcript,
    chapters: list[Chapter] | None = None,
    key_takeaways: list[str] | None = None,
    final_summary: str | None = None,
    glossary: dict[str, str] | None = None,
) -> str:
    """Generate a full markdown document from a transcript and enrichments."""
    lines: list[str] = []

    # Title
    title = transcript.source_title or "Untitled"
    lines.append(f"# {title}")
    lines.append("")

    # Metadata block
    meta_parts = []
    if transcript.date:
        meta_parts.append(f"Date: {transcript.date.strftime('%Y-%m-%d')}")
    if transcript.duration:
        meta_parts.append(f"Duration: {_format_duration(transcript.duration)}")
    if transcript.source_url:
        meta_parts.append(f"Source: {transcript.source_url}")
    if meta_parts:
        lines.extend(meta_parts)
        lines.append("")

    # Key takeaways
    if key_takeaways:
        lines.append("## Key Takeaways")
        lines.append("")
        for kt in key_takeaways:
            lines.append(f"- {kt}")
        lines.append("")

    # Chapters
    if chapters:
        # Table of contents
        lines.append("## Chapters")
        lines.append("")
        for i, ch in enumerate(chapters):
            lines.append(f"{i + 1}. {ch.title}")
        lines.append("")

        # Chapter content
        total_segs = sum(len(ch.segments) for ch in chapters)
        if total_segs > 0:
            # Chapters have mapped segments — render per chapter
            for ch in chapters:
                if ch.segments:
                    lines.append(f"# {ch.title}")
                    lines.append("")
                    for seg in ch.segments:
                        if seg.speaker:
                            lines.append(f"**{seg.speaker}:** {seg.text}")
                        else:
                            lines.append(seg.text)
                        lines.append("")
        else:
            # Chapters are outline only — render as section headers
            segs_per_chapter = max(1, len(transcript.segments) // len(chapters))
            for i, ch in enumerate(chapters):
                start = i * segs_per_chapter
                end = start + segs_per_chapter if i < len(chapters) - 1 else len(transcript.segments)
                lines.append(f"# {ch.title}")
                lines.append("")
                for seg in transcript.segments[start:end]:
                    if seg.speaker:
                        lines.append(f"**{seg.speaker}:** {seg.text}")
                    else:
                        lines.append(seg.text)
                    lines.append("")
    else:
        # Raw transcript fallback
        for seg in transcript.segments:
            if seg.speaker:
                lines.append(f"**{seg.speaker}:** {seg.text}")
            else:
                lines.append(seg.text)
            lines.append("")

    # Final summary
    if final_summary:
        lines.append("# Final Summary")
        lines.append("")
        lines.append(final_summary)
        lines.append("")

    # Glossary
    if glossary:
        lines.append("# Glossary")
        lines.append("")
        for term, definition in sorted(glossary.items()):
            lines.append(f"**{term}**: {definition}")
            lines.append("")

    return "\n".join(lines)


def segments_to_markdown(segments: list[Segment]) -> str:
    """Convert segments to simple markdown (no front matter, no chapters)."""
    lines: list[str] = []
    for seg in segments:
        if seg.speaker:
            lines.append(f"**{seg.speaker}:** {seg.text}")
        else:
            lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"
