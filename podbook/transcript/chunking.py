"""Chunking strategy — split transcripts at sentence/paragraph boundaries for LLM processing.

Never splits mid-sentence. Prefers topic/paragraph boundaries.
"""

from __future__ import annotations

import re

from podbook.models import Segment


def chunk_by_words(
    segments: list[Segment],
    target_size: int = 3000,
    max_size: int = 5000,
) -> list[list[Segment]]:
    """Split segments into chunks of roughly target_size words each.

    Boundaries are chosen at sentence endings within the target window,
    never mid-sentence.
    """
    if not segments:
        return []

    # Accumulate all sentences with positions
    sentences: list[tuple[int, int, str]] = []  # (seg_idx, char_pos, text)
    for i, seg in enumerate(segments):
        for sent in _split_sentences(seg.text):
            sentences.append((i, 0, sent))

    # Greedy pack sentences into chunks
    chunks: list[list[Segment]] = []
    current_chunk: list[Segment] = []
    current_words = 0
    current_seg_indices: set[int] = set()

    # Build chunks sentence-by-sentence
    sent_buf: list[str] = []
    pending_seg_indices: set[int] = set()

    for seg_idx, _, sent_text in sentences:
        sent_words = len(sent_text.split())

        if current_words + sent_words > max_size and sent_buf:
            # Flush current chunk
            chunks.append(_build_segment_chunk(sent_buf, pending_seg_indices, segments))
            sent_buf = []
            pending_seg_indices = set()
            current_words = 0

        sent_buf.append(sent_text)
        pending_seg_indices.add(seg_idx)
        current_words += sent_words

    # Flush remaining
    if sent_buf:
        chunks.append(_build_segment_chunk(sent_buf, pending_seg_indices, segments))

    return chunks


def chunk_by_segments(
    segments: list[Segment],
    target_segments: int = 20,
    max_segments: int = 40,
) -> list[list[Segment]]:
    """Split segments into chunks of roughly target_segments count.

    Simpler alternative to word-based chunking. Splits at natural pauses
    (gaps > 2 seconds between segments).
    """
    if not segments:
        return []

    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    count = 0

    for i, seg in enumerate(segments):
        current.append(seg)
        count += 1

        should_split = count >= max_segments or (
            count >= target_segments
            and i + 1 < len(segments)
            and segments[i + 1].start - seg.end > 2.0
        )

        if should_split:
            chunks.append(current)
            current = []
            count = 0

    if current:
        chunks.append(current)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences without destroying abbreviations."""
    pattern = r'(?<=[.!?])\s+(?=[A-Z"])'
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def _build_segment_chunk(
    sent_buf: list[str],
    seg_indices: set[int],
    source_segments: list[Segment],
) -> list[Segment]:
    """Reconstruct minimal segments from buffered sentences."""
    if not seg_indices:
        # Find which segments contributed
        combined = " ".join(sent_buf)
        # Fallback: create one segment
        start = source_segments[0].start if source_segments else 0.0
        end = source_segments[-1].end if source_segments else 0.0
        return [Segment(start=start, end=end, text=combined)]

    indices = sorted(seg_indices)
    start = source_segments[indices[0]].start
    end = source_segments[indices[-1]].end
    return [Segment(start=start, end=end, text=" ".join(sent_buf))]
