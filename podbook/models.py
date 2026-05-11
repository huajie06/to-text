"""Canonical data models for the PodBook pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel


class SourceType(str, Enum):
    YOUTUBE = "youtube"
    PODCAST_PAGE = "podcast_page"
    RSS = "rss"
    LOCAL_AUDIO = "local_audio"
    LOCAL_VIDEO = "local_video"


class Segment(BaseModel):
    """A single transcript segment — the canonical, immutable unit."""

    speaker: str | None = None
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """The canonical transcript — immutable once produced."""

    source_url: str | None = None
    source_title: str | None = None
    duration: float | None = None
    date: datetime | None = None
    language: str | None = None
    segments: list[Segment]

    # Rich metadata for AI context
    description: str | None = None
    channel: str | None = None
    tags: list[str] | None = None


class Chapter(BaseModel):
    """A cleaned, readable chapter derived from transcript segments."""

    title: str
    segments: list[Segment]


class EbookConfig(BaseModel):
    """Configuration for ebook generation."""

    title: str
    author: str = "PodBook"
    publisher: str | None = None
    language: str = "en"
    cover_path: Path | None = None
