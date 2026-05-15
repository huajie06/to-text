"""Standalone script: merge diarization.json into transcript.json.

Reads transcript.json and diarization.json from a source cache directory,
aligns them by time overlap, and writes diarization_transcript.json with
speaker labels populated.

Strategy:
  For each transcript segment, check for ANY time overlap with each
  diarization speaker independently:

  - Overlaps only SPEAKER_00  → "SPEAKER_00"
  - Overlaps only SPEAKER_01  → "SPEAKER_01"
  - Overlaps BOTH             → "SPEAKER_00_SPEAKER_01" (interjection)
  - Overlaps NEITHER           → longest-duration speaker (fallback)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def _overlaps_any(s_start: float, s_end: float, segments: list) -> bool:
    """Check if a time range overlaps with ANY of the given diarization segments."""
    for ds, de, _ in segments:
        if max(0.0, min(s_end, de) - max(s_start, ds)) > 0:
            return True
    return False


def merge_diarization(
    transcript_path: Path,
    diarization_path: Path,
    output_path: Path | None = None,
    separator: str = "_",
) -> list[dict]:
    """Align diarization speaker segments to transcript segments.

    Parameters
    ----------
    transcript_path, diarization_path:
        Paths to the input JSON files.
    output_path:
        If set, write the merged result here.
    separator:
        Separator between speaker IDs when a segment overlaps both
        (default "_" → "SPEAKER_00_SPEAKER_01").

    Returns the list of segments with speaker labels populated.
    """
    transcript = json.loads(transcript_path.read_text())
    diarization = json.loads(diarization_path.read_text())

    # Group diarization segments by speaker ID
    by_speaker: dict[str, list] = defaultdict(list)
    for ds, de, spk in diarization:
        by_speaker[spk].append((ds, de, spk))

    # Sort speakers by total duration (longest = fallback)
    speaker_duration = {
        spk: sum(de - ds for ds, de, _ in segs)
        for spk, segs in by_speaker.items()
    }
    sorted_speakers = sorted(speaker_duration, key=lambda s: -speaker_duration[s])
    fallback = sorted_speakers[0]

    for seg in transcript["segments"]:
        s_start, s_end = seg["start"], seg["end"]

        overlapping = [
            spk for spk in sorted_speakers
            if _overlaps_any(s_start, s_end, by_speaker[spk])
        ]

        if not overlapping:
            seg["speaker"] = fallback
        elif len(overlapping) == 1:
            seg["speaker"] = overlapping[0]
        else:
            seg["speaker"] = separator.join(overlapping)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(transcript, indent=2))
        counts = Counter(s["speaker"] for s in transcript["segments"])
        print(f"Wrote {output_path}")
        print(f"Speaker distribution: {dict(counts)}")

    return transcript["segments"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge diarization into transcript")
    parser.add_argument("source_dir", type=Path, help="Source cache directory (e.g. output/{hash}-{slug})")
    parser.add_argument("--output", "-o", type=Path, help="Output path (default: <source_dir>/diarization_transcript.json)")
    parser.add_argument("--separator", default="_", help="Separator for multi-speaker label (default: '_')")
    args = parser.parse_args()

    transcript_path = args.source_dir / "transcript.json"
    diarization_path = args.source_dir / "diarization.json"

    if not transcript_path.exists():
        parser.error(f"Not found: {transcript_path}")
    if not diarization_path.exists():
        parser.error(f"Not found: {diarization_path}")

    output_path = args.output or (args.source_dir / "diarization_transcript.json")

    merge_diarization(transcript_path, diarization_path, output_path, separator=args.separator)
