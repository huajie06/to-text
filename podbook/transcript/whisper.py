"""Local transcription via faster-whisper (CTranslate2)."""

from __future__ import annotations

from pathlib import Path

from podbook.models import Segment


def transcribe(
    audio_path: Path,
    model_name: str = "base",
    language: str | None = None,
) -> list[Segment]:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to a 16kHz mono WAV file.
        model_name: Whisper model size or HuggingFace model ID
                    (e.g. 'base', 'small', 'deepdml/faster-whisper-large-v3-turbo').
        language: Language code hint (e.g. 'en').

    Returns:
        List of transcript segments with timing.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments = []

    result, info = model.transcribe(
        str(audio_path),
        language=language,
    )

    for seg in result:
        segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            )
        )

    return segments


def ensure_wav(audio_path: Path, output_path: Path | None = None) -> Path:
    """Ensure audio is in 16kHz mono WAV format for transcription.

    Converts via ffmpeg if the file is not already 16kHz mono WAV.
    """
    import json
    import subprocess

    # Probe actual format
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if audio_streams:
        s = audio_streams[0]
        is_16k = int(s.get("sample_rate", 0)) == 16000
        is_mono = int(s.get("channels", 0)) == 1
        is_wav = audio_path.suffix.lower() == ".wav"
        if is_16k and is_mono and is_wav:
            return audio_path

    dest = output_path or audio_path.with_suffix(".16k.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-i", str(audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest
