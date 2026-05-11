"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from rich.console import Console
from rich.table import Table

from podbook.models import SourceType, Transcript

console = Console()


def run_pipeline(
    *,
    source: str,
    source_type: SourceType,
    output_dir: Path | None = None,
    max_tokens: int | None = None,
    dry_run: bool = False,
    force_transcribe: bool = False,
    cleanup: bool = False,
    enrich: bool = False,
    provider: str = "ollama",
    model: str | None = None,
    fraction: float | None = None,
) -> Path | None:
    """Run the full PodBook pipeline.

    Returns the path to the generated EPUB file, or None in dry-run mode.
    """
    output_dir = output_dir or Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _cache_key(source)

    # ── Dry-run: inspect cache and estimate ───────────────────────
    if dry_run:
        console.print("[bold]DRY RUN[/] — no expensive operations will be performed")
        console.print()
        _inspect_cache(cache_dir, cache_key, source, source_type)
        console.print()
        if cleanup or enrich:
            _show_cost_estimate(source, source_type, cleanup, enrich, provider, model, max_tokens)
        else:
            console.print("[dim]No AI passes requested. Use --cleanup or --enrich for LLM estimates.[/]")
        console.print()
        console.print("[dim]Run without --dry-run to execute the full pipeline.[/]")
        return None

    # ── Phase 1: Extract transcript ──────────────────────────────
    console.print("[dim]──────────────────────────────────────────[/]")
    console.print("[bold]Phase 1:[/] Extracting transcript...")

    transcript = _extract_transcript(
        source=source,
        source_type=source_type,
        cache_dir=cache_dir,
        force_transcribe=force_transcribe,
    )

    if not transcript.segments:
        console.print(
            "[bold red]Error:[/] No transcript segments produced. "
            "Check the source or use --force-transcribe."
        )
        raise SystemExit(1)

    console.print(
        f"  [green]✓[/] {len(transcript.segments)} segments "
        f"({_format_duration(_total_duration(transcript))})"
    )
    if transcript.description:
        console.print(f"  [dim]Source:[/] {transcript.channel or 'Unknown'}")
        desc_preview = transcript.description[:120].replace("\n", " ")
        console.print(f"  [dim]Description:[/] {desc_preview}...")

    # Cache the transcript as JSON
    _save_transcript_cache(cache_dir, cache_key, transcript)

    # ── Preprocessing: classify + filter segments ────────────────
    from podbook.transcript.preprocess import (
        classify_segments,
        filter_content,
        content_ratio,
        label_summary,
        SegmentLabel,
    )

    labeled = classify_segments(transcript.segments)
    summary_stats = label_summary(labeled)
    ratio = content_ratio(labeled)

    if summary_stats:
        stats_str = ", ".join(
            f"{label.value}: {count}" for label, count in sorted(summary_stats.items(), key=lambda x: -x[1])
        )
        console.print(f"  [dim]Segment labels: {stats_str}[/]")

    # Filter to content-only for AI processing
    content_segments = filter_content(labeled)
    removed = len(transcript.segments) - len(content_segments)
    if removed > 0:
        console.print(
            f"  [yellow]Filtered {removed} non-content segments "
            f"(ads, promos, meta) — {len(content_segments)} content segments remain[/]"
        )
    transcript = transcript.model_copy(update={"segments": content_segments})

    # Fractional processing (for testing)
    if fraction is not None and fraction < 1.0:
        cutoff = int(len(transcript.segments) * fraction)
        transcript = transcript.model_copy(update={"segments": transcript.segments[:cutoff]})
        console.print(
            f"  [dim]Processing {fraction:.1%} of transcript: "
            f"{len(transcript.segments)} segments[/]"
        )

    if not transcript.segments:
        console.print("[bold red]Error:[/] No content segments after filtering.")
        raise SystemExit(1)

    # ── Phase 2: AI passes (optional) ────────────────────────────
    llm = None
    token_spent = 0

    if cleanup or enrich:
        llm = _get_provider(provider, model)
        console.print("[dim]──────────────────────────────────────────[/]")
        console.print(
            f"[bold]Phase 2:[/] AI processing "
            f"([dim]{llm.name}/{getattr(llm, 'model', 'default')}[/])"
        )

    chapters = None
    takeaways = None
    summary = None
    cleaned_segments = None

    if cleanup and llm is not None:
        console.print("  [bold]Pass 2a:[/] Cleaning transcript...")
        from podbook.ai.cleanup import cleanup_transcript

        cleaned_segments, usage = cleanup_transcript(transcript, llm)
        token_spent += usage.total
        console.print(
            f"  [green]✓[/] Cleanup complete "
            f"({usage.input_tokens:,} in / {usage.output_tokens:,} out)"
        )
        if max_tokens and token_spent >= max_tokens:
            console.print("[yellow]  Token budget reached, stopping AI passes[/]")

    if enrich and llm is not None:
        if max_tokens and token_spent >= max_tokens:
            console.print("  [yellow]Skipping enrichment — token budget exhausted[/]")
        else:
            source_transcript = transcript
            if cleaned_segments:
                source_transcript = transcript.model_copy(update={"segments": cleaned_segments})

            console.print("  [bold]Pass 2b:[/] Generating chapters...")
            from podbook.ai.summarize import generate_chapters

            chapters, usage = generate_chapters(source_transcript, llm)
            token_spent += usage.total
            console.print(
                f"  [green]✓[/] {len(chapters)} chapters "
                f"({usage.input_tokens:,} in / {usage.output_tokens:,} out)"
            )

            hit_cap = max_tokens and token_spent >= max_tokens

            console.print("  [bold]Pass 2c:[/] Key takeaways...")
            if not hit_cap:
                from podbook.ai.summarize import generate_takeaways

                takeaways, usage = generate_takeaways(source_transcript, llm)
                token_spent += usage.total
                console.print(
                    f"  [green]✓[/] {len(takeaways)} takeaways "
                    f"({usage.input_tokens:,} in / {usage.output_tokens:,} out)"
                )
                hit_cap = max_tokens and token_spent >= max_tokens

            console.print("  [bold]Pass 2d:[/] Final summary...")
            if not hit_cap:
                from podbook.ai.summarize import generate_summary

                summary, usage = generate_summary(source_transcript, llm)
                token_spent += usage.total
                console.print(
                    f"  [green]✓[/] Summary "
                    f"({usage.input_tokens:,} in / {usage.output_tokens:,} out)"
                )

        if token_spent > 0:
            console.print(f"  [bold]Total tokens spent:[/] {token_spent:,}")
            if max_tokens:
                console.print(f"  [dim]Budget: {max_tokens:,}[/]")

    # ── Phase 3: Generate markdown ───────────────────────────────
    console.print("[dim]──────────────────────────────────────────[/]")
    console.print("[bold]Phase 3:[/] Generating markdown...")

    from podbook.ebook.markdown import generate_markdown

    md_text = generate_markdown(
        transcript,
        chapters=chapters,
        key_takeaways=takeaways,
        final_summary=summary,
    )
    md_path = output_dir / f"{_slugify(transcript.source_title or 'transcript')}.md"
    md_path.write_text(md_text)
    console.print(f"  [green]✓[/] Markdown written to {md_path}")

    # ── Phase 4: Generate EPUB ───────────────────────────────────
    console.print("[dim]──────────────────────────────────────────[/]")
    console.print("[bold]Phase 4:[/] Generating EPUB...")

    from podbook.ebook.epub import generate_epub
    from podbook.models import EbookConfig

    epub_config = EbookConfig(
        title=transcript.source_title or "Untitled",
        author=transcript.channel or "PodBook",
        language=transcript.language or "en",
    )
    epub_path = output_dir / f"{_slugify(transcript.source_title or 'transcript')}.epub"
    generate_epub(md_text, epub_path, epub_config)
    console.print(f"  [green]✓[/] EPUB written to {epub_path}")

    # ── Done ─────────────────────────────────────────────────────
    console.print("[dim]──────────────────────────────────────────[/]")
    console.print(f"[bold green]Done![/] EPUB: {epub_path}")
    if token_spent > 0:
        console.print(f"  Total LLM tokens: {token_spent:,}")

    return epub_path


# ═══════════════════════════════════════════════════════════════════
# Cache inspection (for --dry-run)
# ═══════════════════════════════════════════════════════════════════

def _inspect_cache(
    cache_dir: Path,
    cache_key: str,
    source: str,
    source_type: SourceType,
) -> None:
    """Inspect what's in the cache and report status."""
    table = Table(title=f"Cache status for: {source[:80]}")
    table.add_column("Artifact", style="dim")
    table.add_column("Status")
    table.add_column("Path")

    # Check audio cache
    audio_files = list(cache_dir.glob(f"*.wav")) + list(cache_dir.glob(f"*.mp3"))
    audio_cached = any(_cache_key(str(f)) == cache_key for f in audio_files) if audio_files else False
    table.add_row(
        "Audio download",
        "[green]cached[/]" if audio_files else "[yellow]not cached[/]",
        str(audio_files[0])[:60] if audio_files else "—",
    )

    # Check transcript cache
    transcript_cache = cache_dir / f"{cache_key}.transcript.json"
    if transcript_cache.exists():
        try:
            data = json.loads(transcript_cache.read_text())
            segs = len(data.get("segments", []))
            table.add_row(
                "Transcript",
                f"[green]cached ({segs} segments)[/]",
                str(transcript_cache)[:60],
            )
        except (json.JSONDecodeError, KeyError):
            table.add_row("Transcript", "[red]corrupted[/]", str(transcript_cache)[:60])
    else:
        table.add_row("Transcript", "[yellow]not cached[/]", "—")

    # Check subtitle cache
    sub_files = list(cache_dir.glob("*.json3")) + list(cache_dir.glob("*.srt")) + list(cache_dir.glob("*.vtt"))
    if sub_files:
        table.add_row("Subtitles", "[green]cached[/]", str(sub_files[0])[:60])
    else:
        table.add_row("Subtitles", "[dim]unavailable[/]", "—")

    # Check markdown cache
    md_files = list(cache_dir.parent.glob("*.md"))
    slug_matches = [f for f in md_files if cache_key[:16] in f.name or cache_key[:16] in str(f)]
    if slug_matches:
        table.add_row("Markdown", "[green]cached[/]", str(slug_matches[0])[:60])

    # Check EPUB cache
    epub_files = list(cache_dir.parent.glob("*.epub"))
    if epub_files:
        table.add_row("EPUB", "[green]cached[/]", str(epub_files[0])[:60])

    console.print(table)


def _show_cost_estimate(
    source: str,
    source_type: SourceType,
    cleanup: bool,
    enrich: bool,
    provider: str,
    model: str | None,
    max_tokens: int | None,
) -> None:
    """Estimate token usage and cost for AI passes."""
    console.print("[bold]Cost Estimate[/]")

    # Do a lightweight metadata fetch to estimate transcript size
    try:
        info = _fetch_light_metadata(source, source_type)
    except Exception:
        info = {}

    duration = info.get("duration", 0)
    if duration:
        # Rough: 150 words/min → ~200 tokens/min for English speech
        est_tokens = int(duration / 60 * 200)
        console.print(f"  Duration: {_format_duration(duration)}")
        console.print(f"  Estimated transcript tokens: ~{est_tokens:,}")
    else:
        est_tokens = 25000  # default estimate for ~1h podcast
        console.print(f"  Estimated transcript tokens: ~{est_tokens:,} (unknown duration)")

    table = Table(title="Estimated AI costs")
    table.add_column("Pass")
    table.add_column("Input tokens")
    table.add_column("Output tokens")
    table.add_column("Est. cost (GPT-4o-mini)")
    table.add_column("Est. cost (Haiku 4.5)")

    if cleanup:
        # Chunked: 7 chunks × ~4.2K input + output
        in_tok = est_tokens + 500
        out_tok = int(est_tokens * 0.9)
        cost_4o = f"${in_tok * 0.15 / 1_000_000 + out_tok * 0.6 / 1_000_000:.4f}"
        cost_haiku = f"${in_tok * 0.25 / 1_000_000 + out_tok * 1.25 / 1_000_000:.4f}"
        table.add_row("Cleanup", f"{in_tok:,}", f"{out_tok:,}", cost_4o, cost_haiku)

    if enrich:
        in_tok = est_tokens + 500
        out_tok = 1500
        cost_4o = f"${in_tok * 0.15 / 1_000_000 + out_tok * 0.6 / 1_000_000:.4f}"
        cost_haiku = f"${in_tok * 0.25 / 1_000_000 + out_tok * 1.25 / 1_000_000:.4f}"
        table.add_row("Chapters", f"{in_tok:,}", f"500", cost_4o, cost_haiku)
        table.add_row("Takeaways", f"{in_tok:,}", f"500", cost_4o, cost_haiku)
        table.add_row("Summary", f"{in_tok:,}", f"500", cost_4o, cost_haiku)

    console.print(table)

    if max_tokens:
        console.print(f"  Budget: {max_tokens:,} tokens")
    if provider == "ollama":
        console.print("  [dim]Using local Ollama — no API cost[/]")


def _fetch_light_metadata(source: str, source_type: SourceType) -> dict:
    """Lightweight metadata fetch (no download)."""
    if source_type == SourceType.YOUTUBE:
        import subprocess
        import json

        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", source],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return json.loads(result.stdout)
    return {}


# ═══════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════

def _cache_key(source: str) -> str:
    """Generate a stable cache key from a source URL/path."""
    return hashlib.sha256(source.encode()).hexdigest()[:32]


def _save_transcript_cache(cache_dir: Path, cache_key: str, transcript: Transcript) -> None:
    """Save transcript as JSON in the cache."""
    cache_path = cache_dir / f"{cache_key}.transcript.json"
    cache_path.write_text(transcript.model_dump_json(indent=2))


def _load_transcript_cache(cache_dir: Path, cache_key: str) -> Transcript | None:
    """Load transcript from cache if available."""
    cache_path = cache_dir / f"{cache_key}.transcript.json"
    if cache_path.exists():
        try:
            return Transcript.model_validate_json(cache_path.read_text())
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════
# Pipeline phases
# ═══════════════════════════════════════════════════════════════════

def _extract_transcript(
    source: str,
    source_type: SourceType,
    cache_dir: Path,
    force_transcribe: bool = False,
) -> Transcript:
    """Extract transcript from source, preferring subtitles over transcription."""
    from podbook.transcript.normalize import normalize

    cache_key = _cache_key(source)

    # Check cache first (unless force_transcribe)
    if not force_transcribe:
        cached = _load_transcript_cache(cache_dir, cache_key)
        if cached and cached.segments:
            console.print("  [dim](using cached transcript)[/]")
            return cached

    transcript: Transcript | None = None

    if source_type == SourceType.YOUTUBE:
        from podbook.sources.youtube import download_audio, extract_youtube

        transcript = extract_youtube(source, cache_dir)

        if not transcript.segments or force_transcribe:
            console.print("  No subtitles found, downloading audio for transcription...")
            # Check audio cache
            audio_files = list(cache_dir.glob("*.wav")) + list(cache_dir.glob("*.16k.wav"))
            if audio_files and not force_transcribe:
                audio_path = audio_files[0]
                console.print("  [dim](using cached audio)[/]")
            else:
                audio_path = download_audio(source, cache_dir)
            transcript = _transcribe_audio(audio_path, transcript)

    elif source_type == SourceType.PODCAST_PAGE:
        from podbook.sources.webpage import extract_webpage

        transcript = extract_webpage(source, cache_dir)
        if not transcript.segments:
            console.print("[yellow]Warning:[/] No transcript found on page.")

    elif source_type == SourceType.RSS:
        from podbook.sources.rss import extract_rss

        transcript = extract_rss(source, cache_dir)
        if not transcript.segments:
            console.print("[yellow]Warning:[/] RSS feed processed but no audio downloaded yet.")

    elif source_type in (SourceType.LOCAL_AUDIO, SourceType.LOCAL_VIDEO):
        from podbook.sources.local import extract_local

        transcript = extract_local(source)
        if not transcript.segments or force_transcribe:
            audio_path = Path(source)
            transcript = _transcribe_audio(audio_path, transcript)

    if transcript and transcript.segments:
        transcript = transcript.model_copy(update={"segments": normalize(transcript.segments)})

    return transcript or Transcript(source_url=source, segments=[])


def _transcribe_audio(audio_path: Path, transcript: Transcript) -> Transcript:
    """Transcribe audio and update the transcript with segments."""
    from podbook.transcript.whisper import ensure_wav, transcribe

    console.print("  Transcribing with whisper.cpp (model=base)...")
    wav_path = ensure_wav(audio_path)
    segments = transcribe(wav_path, model_name="base")
    console.print(f"  [green]✓[/] Transcription produced {len(segments)} segments")

    return transcript.model_copy(update={"segments": segments})


def _get_provider(provider: str, model: str | None):
    """Create an LLM provider instance."""
    if provider == "openai":
        from podbook.ai.providers.openai import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-4o-mini")

    if provider == "ollama":
        from podbook.ai.providers.ollama import OllamaProvider

        return OllamaProvider(model=model or "llama3.2")

    raise ValueError(f"Unknown provider: {provider}")


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def _total_duration(transcript: Transcript) -> float:
    if transcript.duration:
        return transcript.duration
    if transcript.segments:
        return transcript.segments[-1].end
    return 0.0


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _slugify(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:100]
