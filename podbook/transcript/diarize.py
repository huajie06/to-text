"""Speaker diarization via pyannote.audio."""

from __future__ import annotations

import json
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
        use_auth_token=hf_token,
    )
    diarization = pipeline(str(audio_path))

    result: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    cache_path.write_text(json.dumps(result))
    return result


def assign_speakers(
    segments: list[Segment],
    diarization: list[tuple[float, float, str]],
) -> list[Segment]:
    """Tag each segment with the diarization speaker that overlaps it most.

    Uses maximum-overlap alignment: for each whisper/VTT segment, find the
    diarization window (start, end, SPEAKER_XX) with the largest time overlap.
    Segments with no overlap keep their existing speaker value.
    """
    updated: list[Segment] = []
    for seg in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for d_start, d_end, speaker in diarization:
            overlap = max(0.0, min(seg.end, d_end) - max(seg.start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        updated.append(seg.model_copy(update={"speaker": best_speaker or seg.speaker}))
    return updated
