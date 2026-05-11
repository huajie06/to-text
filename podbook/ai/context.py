"""Build rich context strings from transcript metadata for LLM prompts."""

from __future__ import annotations

from podbook.models import Transcript


def build_context(transcript: Transcript) -> str:
    """Build a context string from transcript metadata.

    Used as a prefix in LLM prompts so the model understands the
    podcast format, topic, and participants.
    """
    parts: list[str] = []

    parts.append(f'This is a transcript from a podcast episode titled "{transcript.source_title or "Untitled"}".')

    if transcript.channel:
        parts.append(f"The podcast/show is called \"{transcript.channel}\".")

    if transcript.duration:
        parts.append(f"The episode is {_format_duration(transcript.duration)} long.")

    if transcript.language:
        parts.append(f"The language is {transcript.language}.")

    if transcript.tags:
        parts.append(f"Topics/tags: {', '.join(transcript.tags[:15])}.")

    if transcript.description:
        desc = transcript.description[:500].strip()
        parts.append(f"Episode description: {desc}")

    # Speaker guess from description (common patterns)
    speaker_info = _guess_speakers(transcript)
    if speaker_info:
        parts.append(speaker_info)

    return "\n".join(parts)


def build_speaker_context(transcript: Transcript) -> str:
    """Extract speaker hints from metadata and transcript patterns."""
    hints: list[str] = []

    if transcript.description:
        # Look for "Host: Name" or "Guest: Name" patterns
        import re
        hosts = re.findall(r'(?:[Hh]ost(?:ed)?\s*(?:by)?:?\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', transcript.description)
        guests = re.findall(r'(?:[Gg]uest(?:s)?:?\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', transcript.description)
        if hosts:
            hints.append(f"Likely host(s): {', '.join(hosts[:3])}")
        if guests:
            hints.append(f"Likely guest(s): {', '.join(guests[:5])}")

    return "\n".join(hints) if hints else ""


def _guess_speakers(transcript: Transcript) -> str:
    """Guess number and identity of speakers from metadata."""
    desc = (transcript.description or "") + " " + " ".join(transcript.tags or [])

    lower = desc.lower()
    # Interview indicators
    interview_words = ["interview", "guest", "joined by", "welcomes", "in conversation with"]
    debate_words = ["debate", "vs.", "versus", "face off", "head-to-head"]
    panel_words = ["panel", "roundtable", "round table", "moderated by"]

    if any(w in lower for w in panel_words):
        return "This appears to be a panel discussion with multiple speakers."
    if any(w in lower for w in debate_words):
        return "This appears to be a debate or head-to-head discussion with 2+ speakers."
    if any(w in lower for w in interview_words):
        return "This appears to be an interview format with a host and at least one guest."

    return ""


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"
