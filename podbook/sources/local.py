"""Local file source — handle local audio and video files."""

from __future__ import annotations

from pathlib import Path

from podbook.models import Segment, Transcript


def extract_local(path: str | Path) -> Transcript:
    """Extract metadata from a local audio or video file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    return Transcript(
        source_url=str(p.resolve()),
        source_title=p.stem,
        segments=[],
    )
