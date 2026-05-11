"""YouTube source — subtitle download and audio extraction via yt-dlp."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from podbook.models import Segment, Transcript


def extract_youtube(url: str, cache_dir: Path) -> Transcript:
    """Extract transcript from a YouTube URL.

    Tries manual subtitles first, then auto-generated, then falls back
    to audio download for later transcription.
    """
    info = _fetch_info(url)
    title = info.get("title", "Untitled")
    duration = info.get("duration")
    language = info.get("language")
    description = info.get("description")
    channel = info.get("uploader") or info.get("channel")
    tags = info.get("tags")

    # Try subtitles first
    segments = _download_subtitles(url, cache_dir)
    return Transcript(
        source_url=url,
        source_title=title,
        duration=duration,
        language=language,
        description=description,
        channel=channel,
        tags=tags,
        segments=segments or [],
    )


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from a YouTube URL. Returns path to the audio file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "%(title)s.%(ext)s"

    subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(output_path),
            "--no-playlist",
            url,
        ],
        check=True,
        capture_output=True,
    )

    # Find the downloaded file
    wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wav_files:
        raise FileNotFoundError(f"No audio file found in {output_dir}")
    return wav_files[0]


def _fetch_info(url: str) -> dict:
    result = subprocess.run(
        [
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _download_subtitles(url: str, cache_dir: Path) -> list[Segment] | None:
    """Download subtitles as structured segments. Returns None if unavailable."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try JSON3 format (includes timing) — prefer manual, fall back to auto
    for sub_type in ("", "--write-auto-sub"):
        args = [
            "yt-dlp",
            "--skip-download",
            "--write-sub",
            "--sub-format", "json3",
            "--sub-lang", "en",
            "--convert-subs", "json3",
            "--output", str(cache_dir / "%(title)s"),
            "--no-playlist",
        ]
        if sub_type:
            args.insert(3, sub_type)

        try:
            subprocess.run(args + [url], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            continue

        # Look for .json3 subtitle file
        json3_files = sorted(
            cache_dir.glob("*.json3"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if json3_files:
            return _parse_json3(json3_files[0])

    return None


def _parse_json3(path: Path) -> list[Segment]:
    data = json.loads(path.read_text())
    events = data.get("events", [])
    segments = []

    for ev in events:
        segs = ev.get("segs", [])
        if not segs:
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text:
            segments.append(
                Segment(
                    start=start,
                    end=start + dur if dur > 0 else start + 5.0,
                    text=text,
                )
            )
    return segments
