"""Subtitle extraction and parsing."""

from __future__ import annotations

from pathlib import Path

from podbook.models import Segment


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
        text = " ".join(lines[2:]).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))

    return segments


def parse_vtt(path: Path) -> list[Segment]:
    """Parse a WebVTT subtitle file into segments."""
    text = path.read_text(encoding="utf-8-sig")
    # Remove WEBVTT header
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("WEBVTT"):
        lines = lines[1:]

    segments = []
    current_start = current_end = 0.0
    current_text: list[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.isdigit():
            continue
        if "-->" in line:
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
        else:
            current_text.append(line)

    if current_text:
        segments.append(
            Segment(
                start=current_start,
                end=current_end,
                text=" ".join(current_text).strip(),
            )
        )

    return segments


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
    """Parse '00:01:23.456 --> 00:01:25.789'."""
    start_str, _, end_str = line.partition("-->")
    return _vtt_to_seconds(start_str.strip()), _vtt_to_seconds(end_str.strip())


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
