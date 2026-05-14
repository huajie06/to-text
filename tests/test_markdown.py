"""Tests for markdown generation."""

from datetime import datetime

from podbook.ebook.markdown import generate_markdown, segments_to_markdown
from podbook.models import Chapter, EbookConfig, Segment, SourceType, Transcript


def make_transcript(segments=None, title="Test Episode", channel="Test Channel") -> Transcript:
    if segments is None:
        segments = [
            Segment(start=0.0, end=5.0, text="First segment."),
            Segment(start=5.0, end=10.0, text="Second segment."),
        ]
    return Transcript(
        source_url="https://example.com/episode",
        source_type=SourceType.PODCAST_PAGE,
        source_title=title,
        channel=channel,
        segments=segments,
        duration=600.0,
        date=datetime(2024, 1, 15),
    )


class TestGenerateMarkdown:
    def test_title_in_output(self):
        tr = make_transcript(title="My Great Podcast")
        md = generate_markdown(tr)
        assert "# My Great Podcast" in md

    def test_segments_in_output(self):
        tr = make_transcript()
        md = generate_markdown(tr)
        assert "First segment." in md
        assert "Second segment." in md

    def test_key_takeaways_section(self):
        tr = make_transcript()
        md = generate_markdown(tr, key_takeaways=["Insight one", "Insight two"])
        assert "## Key Takeaways" in md
        assert "- Insight one" in md

    def test_chapters_section(self):
        tr = make_transcript()
        chapters = [
            Chapter(title="Chapter One", segments=[]),
            Chapter(title="Chapter Two", segments=[]),
        ]
        md = generate_markdown(tr, chapters=chapters)
        assert "## Chapters" in md
        assert "Chapter One" in md

    def test_summary_section(self):
        tr = make_transcript()
        md = generate_markdown(tr, final_summary="This was a great episode.")
        assert "# Final Summary" in md
        assert "This was a great episode." in md

    def test_speaker_labels(self):
        segs = [
            Segment(start=0.0, end=5.0, speaker="Host", text="Hello, welcome."),
            Segment(start=5.0, end=10.0, speaker="Guest", text="Thanks for having me."),
        ]
        tr = make_transcript(segments=segs)
        md = generate_markdown(tr)
        assert "**Host:**" in md
        assert "**Guest:**" in md

    def test_metadata_included(self):
        tr = make_transcript()
        md = generate_markdown(tr)
        assert "2024-01-15" in md
        assert "Duration:" in md


class TestSegmentsToMarkdown:
    def test_basic(self):
        segs = [Segment(start=0.0, end=5.0, text="Hello world.")]
        md = segments_to_markdown(segs)
        assert "Hello world." in md

    def test_speaker_prefix(self):
        segs = [Segment(start=0.0, end=5.0, speaker="Alice", text="Hi there.")]
        md = segments_to_markdown(segs)
        assert "**Alice:** Hi there." in md
