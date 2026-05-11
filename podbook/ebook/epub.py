"""EPUB generation via ebooklib."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ebooklib import epub

from podbook.models import EbookConfig


def generate_epub(
    markdown_text: str,
    output_path: Path,
    config: EbookConfig | None = None,
) -> Path:
    """Generate an EPUB file from markdown text.

    Returns the path to the generated EPUB file.
    """
    if config is None:
        config = EbookConfig(title="Untitled")

    book = epub.EpubBook()

    # Metadata
    book.set_identifier(str(uuid4()))
    book.set_title(config.title)
    book.set_language(config.language)
    book.add_author(config.author)
    if config.publisher:
        book.add_metadata("DC", "publisher", config.publisher)
    book.add_metadata("DC", "date", datetime.now().strftime("%Y-%m-%d"))

    # CSS
    style = epub.EpubItem(
        uid="style",
        file_name="style/default.css",
        media_type="text/css",
        content="\n".join([
            "body { font-family: serif; line-height: 1.6; margin: 0 1em; }",
            "h1 { margin-top: 2em; }",
            "h2 { margin-top: 1.5em; }",
            "p { margin: 0.7em 0; }",
        ]),
    )
    book.add_item(style)

    # Convert markdown to HTML-ish chapters (split on H1)
    chapters_md = _split_on_h1(markdown_text)
    epub_chapters = []
    spine = ["nav"]

    for i, md_text in enumerate(chapters_md):
        title = _extract_title(md_text) or f"Chapter {i + 1}"
        html_content = _markdown_to_html_body(md_text)

        ch = epub.EpubHtml(
            title=title,
            file_name=f"chapter_{i:03d}.xhtml",
            lang=config.language,
        )
        ch.content = (
            f'<html xmlns="http://www.w3.org/1999/xhtml" lang="{config.language}">'
            f"<head><title>{title}</title>"
            f'<link rel="stylesheet" type="text/css" href="style/default.css"/>'
            f"</head><body>{html_content}</body></html>"
        )
        book.add_item(ch)
        epub_chapters.append(ch)
        spine.append(ch)

    book.toc = epub_chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return output_path


def _split_on_h1(md_text: str) -> list[str]:
    """Split markdown on H1 headings."""
    import re

    parts = re.split(r"(?=^# )", md_text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _extract_title(md_text: str) -> str | None:
    """Extract the first H1 heading from markdown text."""
    for line in md_text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _markdown_to_html_body(md_text: str) -> str:
    """Convert markdown to HTML body content (basic, no external dep)."""
    import re

    html_lines: list[str] = []
    in_paragraph = False

    for line in md_text.split("\n"):
        stripped = line.strip()

        # Headings
        if stripped.startswith("###### "):
            html_lines.append(f"<h6>{stripped[7:]}</h6>")
            continue
        if stripped.startswith("##### "):
            html_lines.append(f"<h5>{stripped[6:]}</h5>")
            continue
        if stripped.startswith("#### "):
            html_lines.append(f"<h4>{stripped[5:]}</h4>")
            continue
        if stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
            continue
        if stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
            continue
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
            continue

        # Lists
        if stripped.startswith("- "):
            if in_paragraph:
                html_lines.append("</p>")
                in_paragraph = False
            html_lines.append(f"<li>{_inline_markdown(stripped[2:])}</li>")
            continue

        # Bold inline (handled later for full paragraphs)
        # Empty line = paragraph break
        if not stripped:
            if in_paragraph:
                html_lines.append("</p>")
                in_paragraph = False
            continue

        # Regular paragraph
        if not in_paragraph:
            html_lines.append("<p>")
            in_paragraph = True
        else:
            html_lines.append("<br/>")
        html_lines.append(_inline_markdown(stripped))

    if in_paragraph:
        html_lines.append("</p>")

    return "\n".join(html_lines)


def _inline_markdown(text: str) -> str:
    """Convert inline markdown (**bold**, *italic*) to HTML."""
    import re

    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text
