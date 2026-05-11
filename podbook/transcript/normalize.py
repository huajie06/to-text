"""Transcript normalization — merge segments, fix timing, clean whitespace."""

from __future__ import annotations

from podbook.models import Segment


def normalize(segments: list[Segment]) -> list[Segment]:
    """Normalize a transcript into clean, manageable segments.

    Operations:
    - Merge segments with the same text intention (yt-dlp subtitle chunks)
    - Fix overlapping or negative durations
    - Strip whitespace
    - Remove empty segments
    """
    if not segments:
        return []

    segments = _remove_empty(segments)
    segments = _merge_short(segments)
    segments = _fix_overlaps(segments)
    return segments


def _remove_empty(segments: list[Segment]) -> list[Segment]:
    return [s for s in segments if s.text.strip()]


def _merge_short(segments: list[Segment], min_duration: float = 1.5) -> list[Segment]:
    """Merge very short segments into their neighbors for readability."""
    if len(segments) < 2:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]
        # If this segment is short, merge with next if possible
        if current.end - current.start < min_duration and i + 1 < len(segments):
            nxt = segments[i + 1]
            current = current.model_copy(update={
                "end": nxt.end,
                "text": current.text + " " + nxt.text,
            })
            i += 2
        else:
            i += 1
        merged.append(current)

    return merged


def _fix_overlaps(segments: list[Segment]) -> list[Segment]:
    """Ensure no overlapping or negative durations."""
    fixed = []
    for seg in segments:
        if seg.start > seg.end:
            seg = seg.model_copy(update={"end": seg.start + 1.0})
        if fixed and seg.start < fixed[-1].end:
            seg = seg.model_copy(update={"start": fixed[-1].end})
        if seg.end <= seg.start:
            seg = seg.model_copy(update={"end": seg.start + 1.0})
        fixed.append(seg)
    return fixed
