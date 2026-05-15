"""YouTube source — subtitle download and audio extraction via yt-dlp."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from podbook.models import Segment, Transcript
from podbook.transcript.subtitles import parse_vtt


def extract_youtube(url: str, cache_dir: Path, subs: bool = True) -> Transcript:
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

    segments = _download_subtitles(url, cache_dir) if subs else None
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
    """Download audio from a YouTube URL. Returns path to the audio file.

    Skips download if a WAV file already exists in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        return existing[0]

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

    # Try manual subtitles first, then auto-generated; convert to VTT for timing
    for write_auto in (False, True):
        args = [
            "yt-dlp",
            "--skip-download",
            "--write-sub",
            "--sub-lang", "en",
            "--convert-subs", "vtt",
            "--output", str(cache_dir / "%(title)s"),
            "--no-playlist",
        ]
        if write_auto:
            args.insert(3, "--write-auto-sub")

        try:
            subprocess.run(args + [url], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            continue

        vtt_files = sorted(
            cache_dir.glob("*.vtt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if vtt_files:
            return parse_vtt(vtt_files[0])

    return None
