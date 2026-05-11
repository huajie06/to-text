"""RSS feed source — parse podcast feeds for episodes and audio."""

from __future__ import annotations

from pathlib import Path

import feedparser
import requests

from podbook.models import Segment, Transcript


def extract_rss(url: str, cache_dir: Path) -> Transcript:
    """Parse an RSS feed and extract the first episode's metadata."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": "PodBook/0.1"})
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    if not feed.entries:
        raise ValueError("RSS feed contains no entries")

    entry = feed.entries[0]
    title = entry.get("title", feed.feed.get("title", "Untitled"))

    # Try to find audio enclosure
    audio_url = None
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/") or link.get("rel") == "enclosure":
            audio_url = link.get("href")
            break

    return Transcript(
        source_url=url,
        source_title=title,
        segments=[],
    )


def get_feed_entries(url: str) -> list[dict]:
    """Return all entries from an RSS feed for browsing/selection."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": "PodBook/0.1"})
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return [
        {
            "title": e.get("title", "Untitled"),
            "url": e.get("link", ""),
            "published": e.get("published", ""),
        }
        for e in feed.entries
    ]
