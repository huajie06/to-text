"""Tests for EPUB generation utilities."""

import tempfile
from pathlib import Path

from podbook.ebook.epub import _split_on_h1, _extract_title, _markdown_to_html_body, generate_epub
from podbook.models import EbookConfig


class TestSplitOnH1:
    def test_splits_two_chapters(self):
        md = "# Chapter One\n\nSome text.\n\n# Chapter Two\n\nMore text."
        parts = _split_on_h1(md)
        assert len(parts) == 2
        assert parts[0].startswith("# Chapter One")
        assert parts[1].startswith("# Chapter Two")

    def test_no_h1(self):
        md = "## Not a chapter\n\nJust some text."
        parts = _split_on_h1(md)
        assert len(parts) == 1

    def test_empty(self):
        assert _split_on_h1("") == []

    def test_leading_content_before_h1(self):
        md = "Front matter\n\n# Chapter One\n\nContent."
        parts = _split_on_h1(md)
        # Front matter + chapter one
        assert any("Front matter" in p for p in parts)
        assert any("# Chapter One" in p for p in parts)


class TestExtractTitle:
    def test_extracts_h1(self):
        md = "# My Title\n\nSome content."
        assert _extract_title(md) == "My Title"

    def test_no_h1_returns_none(self):
        assert _extract_title("## Not H1\n\nContent.") is None

    def test_empty(self):
        assert _extract_title("") is None


class TestMarkdownToHtmlBody:
    def test_heading(self):
        html = _markdown_to_html_body("# Hello")
        assert "<h1>" in html
        assert "Hello" in html

    def test_paragraph(self):
        html = _markdown_to_html_body("A simple paragraph.")
        assert "<p>" in html
        assert "A simple paragraph." in html

    def test_bold(self):
        html = _markdown_to_html_body("**bold text**")
        assert "<strong>" in html or "bold text" in html

    def test_list(self):
        html = _markdown_to_html_body("- item one\n- item two")
        assert "<li>" in html
        assert "item one" in html

    def test_multiple_headings(self):
        md = "## Section A\n\nText.\n\n### Subsection\n\nMore text."
        html = _markdown_to_html_body(md)
        assert "<h2>" in html
        assert "<h3>" in html


class TestGenerateEpub:
    def test_generates_file(self):
        md = "# Test Episode\n\nDate: 2024-01-01\n\n## Content\n\nHello world."
        config = EbookConfig(title="Test Episode", author="Test Author")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.epub"
            result = generate_epub(md, out, config)
            assert result == out
            assert out.exists()
            assert out.stat().st_size > 0

    def test_epub_file_is_zip(self):
        md = "# Episode\n\nSome content here."
        config = EbookConfig(title="Episode")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ep.epub"
            generate_epub(md, out, config)
            # EPUB files are ZIP archives starting with PK
            with open(out, "rb") as f:
                header = f.read(2)
            assert header == b"PK"
