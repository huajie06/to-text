"""CLI entry point for PodBook."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
load_dotenv()

import typer
from rich.console import Console

from podbook.models import SourceType

app = typer.Typer(
    name="podbook",
    help="Convert podcasts and videos into readable ebooks.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Manage the PodBook cache.")
app.add_typer(cache_app, name="cache")

console = Console()

_PROVIDERS = "ollama, openai, claude, deepseek"


@app.command()
def build(
    source: Annotated[
        str,
        typer.Argument(help="URL or local file path to the podcast/video source."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for generated files."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Hard limit on LLM token usage for this run."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Estimate costs without making LLM calls."),
    ] = False,
    force_transcribe: Annotated[
        bool,
        typer.Option("--force-transcribe", help="Skip subtitle check, always transcribe locally."),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option("--cleanup", help="Run LLM cleanup pass on the transcript."),
    ] = False,
    enrich: Annotated[
        bool,
        typer.Option("--enrich", help="Run LLM enrichment (chapters, takeaways, summary)."),
    ] = False,
    provider: Annotated[
        str,
        typer.Option("--provider", help=f"LLM provider: {_PROVIDERS}."),
    ] = "ollama",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name override (e.g. gpt-4o-mini, claude-haiku-4-5-20251001)."),
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
    speakers: Annotated[
        bool,
        typer.Option(
            "--speakers",
            help="Label speakers in the transcript (auto-enabled with --cleanup).",
        ),
    ] = False,
    force_diarize: Annotated[
        bool,
        typer.Option(
            "--force-diarize",
            help="Download audio and run pyannote diarization even when subtitles are available. Requires HUGGINGFACE_TOKEN.",
        ),
    ] = False,
) -> None:
    """Build an ebook from a podcast URL or local audio/video file."""
    from podbook.pipeline import run_pipeline

    src_type = _detect_source_type(source)
    console.print(f"[bold]Source:[/] {_source_label(src_type)}")
    if fraction is not None:
        console.print(f"[bold]Fraction:[/] {fraction:.1%}")

    run_pipeline(
        source=source,
        source_type=src_type,
        output_dir=output,
        max_tokens=max_tokens,
        dry_run=dry_run,
        force_transcribe=force_transcribe,
        cleanup=cleanup,
        enrich=enrich,
        provider=provider,
        model=model,
        fraction=fraction,
        label_speakers=speakers,
        force_diarize=force_diarize,
    )


@app.command()
def transcript(
    source: Annotated[
        str,
        typer.Argument(help="URL or local file path to the podcast/video source."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for the transcript JSON."),
    ] = None,
    force_transcribe: Annotated[
        bool,
        typer.Option("--force-transcribe", help="Skip subtitle check, always transcribe locally."),
    ] = False,
) -> None:
    """Extract a transcript and save it as JSON (no EPUB generated)."""
    from podbook.pipeline import _resolve_source_dir, extract_transcript, _slugify

    src_type = _detect_source_type(source)
    console.print(f"[bold]Extracting transcript from:[/] {_source_label(src_type)}")

    output_dir = output.parent if output and output.suffix else (output or Path("output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = _resolve_source_dir(source, output_dir)

    tr = extract_transcript(
        source=source,
        source_type=src_type,
        source_dir=source_dir,
        force_transcribe=force_transcribe,
    )

    if not tr.segments:
        console.print("[bold red]Error:[/] No segments produced.")
        raise typer.Exit(1)

    out_path = output or output_dir / f"{_slugify(tr.source_title or 'transcript')}.transcript.json"
    out_path.write_text(tr.model_dump_json(indent=2))
    console.print(f"[green]✓[/] {len(tr.segments)} segments → {out_path}")


@app.command()
def epub(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to a transcript JSON file.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for the EPUB file."),
    ] = None,
) -> None:
    """Generate an EPUB directly from an existing transcript JSON."""
    from podbook.ebook.epub import generate_epub
    from podbook.ebook.markdown import generate_markdown
    from podbook.models import EbookConfig, Transcript

    console.print(f"[bold]Generating EPUB from:[/] {transcript_path}")

    try:
        tr = Transcript.model_validate_json(transcript_path.read_text())
    except Exception as exc:
        console.print(f"[bold red]Error:[/] Could not parse transcript: {exc}")
        raise typer.Exit(1)

    md_text = generate_markdown(tr)
    out_path = output or transcript_path.with_suffix(".epub")
    generate_epub(
        md_text,
        out_path,
        EbookConfig(
            title=tr.source_title or "Untitled",
            author=tr.channel or "PodBook",
            language=tr.language or "en",
        ),
    )
    console.print(f"[green]✓[/] EPUB written to {out_path}")


# ── Cache subcommands ──────────────────────────────────────────────

@cache_app.command("list")
def cache_list(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-d", help="Output directory to inspect."),
    ] = Path("output"),
) -> None:
    """List all cached artifacts in the output directory."""
    from rich.table import Table

    source_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and "-" in d.name],
        key=lambda d: d.name,
    )
    if not source_dirs and not any(output_dir.glob("*.md")) and not any(output_dir.glob("*.epub")):
        console.print("[yellow]No cache directory found.[/]")
        raise typer.Exit(0)

    table = Table(title=f"Cache: {output_dir}")
    table.add_column("Source", style="dim")
    table.add_column("Type", style="dim")
    table.add_column("File")
    table.add_column("Size")

    any_found = False
    for sd in source_dirs:
        source_label = sd.name
        for f in sorted(sd.iterdir()):
            if f.is_file():
                file_type = f.suffix.upper().lstrip(".")
                if file_type == "JSON":
                    file_type = "Transcript"
                elif file_type in ("WAV", "MP3"):
                    file_type = "Audio"
                elif file_type in ("JSON3", "SRT", "VTT"):
                    file_type = "Subtitles"
                table.add_row(source_label, file_type, f.name, _human_size(f.stat().st_size))
                any_found = True

    for f in sorted(output_dir.glob("*.md")):
        table.add_row("—", "Markdown", f.name, _human_size(f.stat().st_size))
        any_found = True

    for f in sorted(output_dir.glob("*.epub")):
        table.add_row("—", "EPUB", f.name, _human_size(f.stat().st_size))
        any_found = True

    if any_found:
        console.print(table)
    else:
        console.print("[dim]Cache is empty.[/]")


@cache_app.command("clear")
def cache_clear(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-d", help="Output directory to clear."),
    ] = Path("output"),
    type_: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Only clear this type: audio, transcript, subtitle, all."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Clear cached artifacts from the output directory."""
    source_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and "-" in d.name],
        key=lambda d: d.name,
    )
    if not source_dirs:
        console.print("[yellow]No cache directory found.[/]")
        raise typer.Exit(0)

    type_patterns: dict[str, list[str]] = {
        "audio": ["*.wav", "*.mp3", "*.m4a"],
        "transcript": ["transcript.json"],
        "subtitle": ["*.json3", "*.srt", "*.vtt"],
    }

    selected = type_ or "all"
    if selected == "all":
        patterns = [p for ps in type_patterns.values() for p in ps]
    elif selected in type_patterns:
        patterns = type_patterns[selected]
    else:
        console.print(f"[red]Unknown type '{selected}'. Use: audio, transcript, subtitle, all[/]")
        raise typer.Exit(1)

    targets = []
    for sd in source_dirs:
        for pat in patterns:
            targets.extend(sd.glob(pat))

    if not targets:
        console.print("[dim]Nothing to clear.[/]")
        raise typer.Exit(0)

    console.print(f"Will delete {len(targets)} file(s):")
    for f in targets:
        console.print(f"  [dim]{f}[/]")

    if not yes:
        typer.confirm("Proceed?", abort=True)

    for f in targets:
        f.unlink()
    console.print(f"[green]✓[/] Cleared {len(targets)} file(s).")


# ── Helpers ───────────────────────────────────────────────────────

def _detect_source_type(source: str) -> SourceType:
    s = source.strip()
    if s.startswith(".") or s.startswith("/") or s.startswith("~"):
        path = Path(s)
        if path.suffix.lower() in (".mp3", ".wav", ".m4a", ".opus", ".ogg", ".flac"):
            return SourceType.LOCAL_AUDIO
        if path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return SourceType.LOCAL_VIDEO
        return SourceType.LOCAL_AUDIO
    if "youtube.com" in s or "youtu.be" in s:
        return SourceType.YOUTUBE
    if s.lower().endswith((".xml", ".rss")) or "/feed" in s.lower() or "/rss" in s.lower():
        return SourceType.RSS
    return SourceType.PODCAST_PAGE


def _source_label(src_type: SourceType) -> str:
    return {
        SourceType.YOUTUBE: "YouTube",
        SourceType.PODCAST_PAGE: "Podcast Page",
        SourceType.RSS: "RSS Feed",
        SourceType.LOCAL_AUDIO: "Local Audio File",
        SourceType.LOCAL_VIDEO: "Local Video File",
    }[src_type]


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n //= 1024
    return f"{n:.0f} TB"


if __name__ == "__main__":
    app()
