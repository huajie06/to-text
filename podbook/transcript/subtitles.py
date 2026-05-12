"""Subtitle extraction and parsing."""

from __future__ import annotations

import re
from pathlib import Path

from podbook.models import Segment

_INLINE_TAG_RE = re.compile(r"<[^>]+>")


def parse_srt(path: Path) -> list[Segment]:
    """Parse an SRT subtitle file into segments."""
    text = path.read_text(encoding="utf-8-sig")
    blocks = text.strip().replace("\r\n", "\n").split("\n\n")
    segments = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # lines[0] is index, lines[1] is timestamp, lines[2:] is text
        start, end = _parse_srt_timestamp(lines[1])
        clean_lines = [_INLINE_TAG_RE.sub("", l).strip() for l in lines[2:]]
        text = " ".join(l for l in clean_lines if l).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))

    return _dedup_rolling_captions(segments)


def parse_vtt(path: Path) -> list[Segment]:
    """Parse a WebVTT subtitle file into segments."""
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.strip().split("\n")

    segments = []
    current_start = current_end = 0.0
    current_text: list[str] = []
    in_cues = False  # skip header region before first cue

    for line in lines:
        line = line.strip()
        if "-->" in line:
            in_cues = True
            if current_text:
                segments.append(
                    Segment(
                        start=current_start,
                        end=current_end,
                        text=" ".join(current_text).strip(),
                    )
                )
                current_text = []
            current_start, current_end = _parse_vtt_timestamp(line)
        elif in_cues and line and not line.isdigit():
            clean = _INLINE_TAG_RE.sub("", line).strip()
            if clean:
                current_text.append(clean)

    if current_text:
        segments.append(
            Segment(
                start=current_start,
                end=current_end,
                text=" ".join(current_text).strip(),
            )
        )

    return _dedup_rolling_captions(segments)


def _dedup_rolling_captions(segments: list[Segment]) -> list[Segment]:
    """Strip YouTube rolling-caption prefix duplication.

    YouTube auto-captions repeat the previous cumulative text as a prefix in
    each new cue. Strip that prefix to leave only the new words per segment.
    """
    if not segments:
        return segments

    result: list[Segment] = []
    prev_text = ""

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if prev_text and text.startswith(prev_text):
            new_text = text[len(prev_text):].strip()
            if new_text:
                result.append(Segment(start=seg.start, end=seg.end, text=new_text))
                prev_text = new_text
            # else pure duplicate — drop it
        else:
            result.append(seg)
            prev_text = text

    return result


def _parse_srt_timestamp(line: str) -> tuple[float, float]:
    """Parse '00:01:23,456 --> 00:01:25,789'."""
    start_str, _, end_str = line.partition("-->")
    return _srt_to_seconds(start_str.strip()), _srt_to_seconds(end_str.strip())


def _srt_to_seconds(ts: str) -> float:
    """Convert '00:01:23,456' to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_vtt_timestamp(line: str) -> tuple[float, float]:
    """Parse '00:01:23.456 --> 00:01:25.789 [optional cue settings]'."""
    start_str, _, end_str = line.partition("-->")
    # Drop any cue settings after the end timestamp (e.g. 'align:start position:0%')
    return _vtt_to_seconds(start_str.strip()), _vtt_to_seconds(end_str.strip().split()[0])


def _vtt_to_seconds(ts: str) -> float:
    """Convert '00:01:23.456' or '01:23.456' to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
        s, ms = rest.split(".")
    else:
        h = "0"
        m, rest = parts
        s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
