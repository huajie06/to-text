"""CLI entry point for PodBook."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress

from podbook.models import SourceType

app = typer.Typer(
    name="podbook",
    help="Convert podcasts and videos into readable ebooks.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def build(
    source: Annotated[
        str,
        typer.Argument(help="URL or local file path to the podcast/video source."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output directory for generated files.",
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens",
            help="Hard limit on LLM token usage for this run.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Estimate costs without making LLM calls.",
        ),
    ] = False,
    force_transcribe: Annotated[
        bool,
        typer.Option(
            "--force-transcribe",
            help="Skip subtitle check, always transcribe locally.",
        ),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option(
            "--cleanup",
            help="Run LLM cleanup pass on the transcript.",
        ),
    ] = False,
    enrich: Annotated[
        bool,
        typer.Option(
            "--enrich",
            help="Run LLM enrichment (chapters, takeaways, summary).",
        ),
    ] = False,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="LLM provider: openai, ollama.",
        ),
    ] = "ollama",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model name override (e.g. gpt-4o-mini, llama3.2).",
        ),
    ] = None,
    fraction: Annotated[
        float | None,
        typer.Option(
            "--fraction",
            help="Process only this fraction of the transcript (0.0–1.0). For testing.",
            min=0.0,
            max=1.0,
        ),
    ] = None,
) -> None:
    """Build an ebook from a podcast URL or local audio/video file."""
    from podbook.pipeline import run_pipeline

    src = _detect_source(source)
    console.print(f"[bold]Source:[/] {src}")
    console.print(f"[bold]Type:[/] {_detect_source_type(source).value}")
    if fraction is not None:
        console.print(f"[bold]Fraction:[/] {fraction:.1%}")

    run_pipeline(
        source=source,
        source_type=_detect_source_type(source),
        output_dir=output,
        max_tokens=max_tokens,
        dry_run=dry_run,
        force_transcribe=force_transcribe,
        cleanup=cleanup,
        enrich=enrich,
        provider=provider,
        model=model,
        fraction=fraction,
    )


@app.command()
def transcript(
    source: Annotated[
        str,
        typer.Argument(help="URL or local file path to the podcast/video source."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output path for the transcript JSON.",
        ),
    ] = None,
    force_transcribe: Annotated[
        bool,
        typer.Option(
            "--force-transcribe",
            help="Skip subtitle check, always transcribe locally.",
        ),
    ] = False,
) -> None:
    """Generate a transcript from a podcast source and save as JSON."""
    console.print("[bold]Transcript generation[/]")


@app.command()
def epub(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to a transcript JSON or markdown file.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output path for the EPUB file.",
        ),
    ] = None,
) -> None:
    """Generate an EPUB from an existing transcript file."""
    console.print(f"[bold]EPUB generation from:[/] {transcript_path}")


def _detect_source(source: str) -> str:
    """Detect and return a human-readable source label."""
    st = _detect_source_type(source)
    labels = {
        SourceType.YOUTUBE: "YouTube",
        SourceType.PODCAST_PAGE: "Podcast Page",
        SourceType.RSS: "RSS Feed",
        SourceType.LOCAL_AUDIO: "Local Audio File",
        SourceType.LOCAL_VIDEO: "Local Video File",
    }
    return labels[st]


def _detect_source_type(source: str) -> SourceType:
    """Detect the source type from a URL or file path."""
    s = source.strip()

    # Local files
    if s.startswith(".") or s.startswith("/") or s.startswith("~"):
        path = Path(s)
        if path.suffix.lower() in (".mp3", ".wav", ".m4a", ".opus", ".ogg", ".flac"):
            return SourceType.LOCAL_AUDIO
        if path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return SourceType.LOCAL_VIDEO
        return SourceType.LOCAL_AUDIO  # best guess

    # YouTube
    if "youtube.com" in s or "youtu.be" in s:
        return SourceType.YOUTUBE

    # RSS
    if s.lower().endswith((".xml", ".rss")) or "/feed" in s.lower() or "/rss" in s.lower():
        return SourceType.RSS

    # Default: podcast webpage
    return SourceType.PODCAST_PAGE


if __name__ == "__main__":
    app()
