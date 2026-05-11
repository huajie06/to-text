"""Webpage source — parse podcast pages for audio, transcripts, RSS feeds."""

from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from podbook.models import Segment, Transcript


def extract_webpage(url: str, cache_dir: Path) -> Transcript:
    """Parse a podcast webpage for embedded audio, transcripts, or RSS feeds."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": "PodBook/0.1"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = _find_title(soup)
    audio_url = _find_audio(soup, url)
    rss_url = _find_rss(soup, url)
    transcript_text = _find_transcript(soup)

    segments: list[Segment] = []
    if transcript_text:
        segments = _text_to_segments(transcript_text)

    return Transcript(
        source_url=url,
        source_title=title,
        segments=segments,
    )


def _find_title(soup: BeautifulSoup) -> str | None:
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"]
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def _find_audio(soup: BeautifulSoup, base_url: str) -> str | None:
    # <audio> tag
    audio = soup.find("audio")
    if audio and audio.get("src"):
        return _resolve_url(audio["src"], base_url)

    # <source> inside audio
    source = soup.find("source", type=re.compile(r"^audio/"))
    if source and source.get("src"):
        return _resolve_url(source["src"], base_url)

    # og:audio
    og_audio = soup.find("meta", property="og:audio")
    if og_audio and og_audio.get("content"):
        return og_audio["content"]

    # Links ending in mp3
    for link in soup.find_all("a", href=True):
        if link["href"].endswith(".mp3"):
            return _resolve_url(link["href"], base_url)

    return None


def _find_rss(soup: BeautifulSoup, base_url: str) -> str | None:
    for link in soup.find_all("link", type="application/rss+xml", href=True):
        return _resolve_url(link["href"], base_url)
    return None


def _find_transcript(soup: BeautifulSoup) -> str | None:
    # Common transcript container patterns
    for cls in ("transcript", "episode-transcript", "podcast-transcript", "show-notes"):
        div = soup.find("div", class_=re.compile(cls, re.I))
        if div:
            return div.get_text(separator="\n").strip()

    # Look for a <section> with "transcript" in its text
    for section in soup.find_all("section"):
        heading = section.find(["h1", "h2", "h3", "h4"])
        if heading and "transcript" in heading.get_text().lower():
            return section.get_text(separator="\n").strip()

    return None


def _text_to_segments(text: str) -> list[Segment]:
    """Convert plain text to rough segments by paragraph breaks."""
    paragraphs = text.strip().split("\n\n")
    segments = []
    t = 0.0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = len(para.split())
        duration = max(words / 150.0 * 60.0, 1.0)  # rough estimate at 150 wpm
        segments.append(Segment(start=t, end=t + duration, text=para))
        t += duration
    return segments


def _resolve_url(href: str, base_url: str) -> str:
    from urllib.parse import urljoin

    return urljoin(base_url, href)
