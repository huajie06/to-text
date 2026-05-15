"""Speaker diarization via pyannote.audio."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from podbook.models import Segment


def diarize_audio(
    audio_path: Path,
    hf_token: str,
    *,
    source_dir: Path,
) -> list[tuple[float, float, str]]:
    """Run pyannote speaker diarization. Returns (start, end, speaker_id) tuples.

    Results are cached in source_dir/diarization.json — pyannote on CPU is slow
    (~1x realtime), so re-running on the same audio reads from cache.

    Requires pyannote.audio>=3.1 and a HuggingFace token with access to:
      - pyannote/speaker-diarization-3.1
      - pyannote/segmentation-3.0
    Accept terms at https://hf.co/pyannote/speaker-diarization-3.1 first.
    """
    cache_path = source_dir / "diarization.json"
    if cache_path.exists():
        return [tuple(x) for x in json.loads(cache_path.read_text())]  # type: ignore[return-value]

    from pyannote.audio import Pipeline  # type: ignore[import]

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    output = pipeline(str(audio_path))
    serialized = output.serialize()

    result: list[tuple[float, float, str]] = [
        (turn["start"], turn["end"], turn["speaker"])
        for turn in serialized.get("diarization", [])
    ]
    cache_path.write_text(json.dumps(result))
    return result


def _overlaps_any(s_start: float, s_end: float, windows: list[tuple[float, float, str]]) -> bool:
    """Check if a time range overlaps with ANY of the given windows."""
    for w_start, w_end, _ in windows:
        if max(0.0, min(s_end, w_end) - max(s_start, w_start)) > 0:
            return True
    return False


def assign_speakers(
    segments: list[Segment],
    diarization: list[tuple[float, float, str]],
    separator: str = "_",
) -> list[Segment]:
    """Tag each segment with diarization speaker(s) by any-overlap alignment.

    For each transcript segment, checks for ANY time overlap with each
    diarization speaker independently:

    - Overlaps only one speaker  → tagged with that speaker ID
    - Overlaps multiple speakers → combined via separator (e.g. SPEAKER_00_SPEAKER_01)
    - Overlaps none              → falls back to the longest-duration speaker

    The any-overlap approach avoids the max-overlap problem where short
    interjections from one speaker are swallowed by the dominant speaker's
    longer windows.
    """
    if not diarization:
        return segments

    # Group diarization windows by speaker ID
    by_speaker: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for ds, de, spk in diarization:
        by_speaker[spk].append((ds, de, spk))

    # Sort speakers by total duration (longest = fallback for unmatched)
    speaker_duration = {
        spk: sum(de - ds for ds, de, _ in segs)
        for spk, segs in by_speaker.items()
    }
    sorted_speakers = sorted(speaker_duration, key=lambda s: -speaker_duration[s])
    fallback = sorted_speakers[0]

    updated: list[Segment] = []
    for seg in segments:
        overlapping = [
            spk for spk in sorted_speakers
            if _overlaps_any(seg.start, seg.end, by_speaker[spk])
        ]

        if not overlapping:
            speaker = fallback
        elif len(overlapping) == 1:
            speaker = overlapping[0]
        else:
            speaker = separator.join(overlapping)

        updated.append(seg.model_copy(update={"speaker": speaker}))

    return updated
