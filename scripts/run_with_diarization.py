"""Run cleanup + enrich passes using diarization_transcript.json as input.

Loads the any-overlap merged transcript (with SPEAKER_XX labels),
maps IDs to real names via LLM, runs all AI passes with speaker
labels embedded in the prompt text, and writes a differentiated
output file for comparison.

Usage:
    uv run python scripts/run_with_diarization.py output/{hash}-{slug} --provider deepseek
"""

from __future__ import annotations

from pathlib import Path

from podbook.ai.cleanup import cleanup_transcript
from podbook.ai.providers.base import LLMProvider
from podbook.ai.speakers import map_speaker_ids
from podbook.ai.summarize import generate_chapters, generate_takeaways, generate_summary
from podbook.ebook.markdown import generate_markdown
from podbook.models import Transcript


def _get_provider(provider_name: str) -> LLMProvider:
    import os
    if provider_name == "deepseek":
        from podbook.ai.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    elif provider_name == "openai":
        from podbook.ai.providers.openai import OpenAIProvider
        return OpenAIProvider()
    elif provider_name == "claude":
        from podbook.ai.providers.anthropic import ClaudeProvider
        return ClaudeProvider()
    else:
        from podbook.ai.providers.ollama import OllamaProvider
        return OllamaProvider()


def _embed_speakers(transcript: Transcript) -> Transcript:
    """Prepend speaker label to each segment's text so the LLM sees who's speaking."""
    updated = []
    for seg in transcript.segments:
        prefix = f"[{seg.speaker}] " if seg.speaker else ""
        updated.append(seg.model_copy(update={"text": f"{prefix}{seg.text}"}))
    return transcript.model_copy(update={"segments": updated})


def _slugify(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:100]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run AI passes on diarization_transcript.json")
    parser.add_argument("source_dir", type=Path, help="Source cache directory (e.g. output/{hash}-{slug})")
    parser.add_argument("--input", default="diarization_transcript.json", help="Input transcript file (default: diarization_transcript.json)")
    parser.add_argument("--provider", default="deepseek", help="LLM provider")
    parser.add_argument("--output-suffix", default="with-diarization", help="Suffix for output filename (before provider)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    input_path = args.source_dir / args.input
    if not input_path.exists():
        parser.error(f"Not found: {input_path}")

    transcript = Transcript.model_validate_json(input_path.read_text())
    n_speakers = len(set(s.speaker for s in transcript.segments if s.speaker))
    print(f"Loaded {len(transcript.segments)} segments, {n_speakers} unique speaker labels")

    provider = _get_provider(args.provider)
    print(f"Provider: {provider.name}/{provider.model}")

    # Step 1: Map speaker IDs to real names (e.g. SPEAKER_00 → Joe Rogan)
    print("\n1. Mapping speaker IDs to names...")
    transcript, speaker_usage = map_speaker_ids(transcript, provider)
    speakers_seen = set(s.speaker for s in transcript.segments if s.speaker)
    print(f"   Speakers: {', '.join(sorted(speakers_seen))}")
    print(f"   ({speaker_usage.input_tokens:,} in / {speaker_usage.output_tokens:,} out)")

    # Step 2: Embed speaker names into segment text for the LLM to see
    print("\n2. Embedding speaker labels into prompt text...")
    transcript = _embed_speakers(transcript)

    # Step 3: Cleanup pass
    print("3. Cleaning transcript...")
    cleaned, cleanup_usage = cleanup_transcript(transcript, provider)
    print(f"   ({cleanup_usage.input_tokens:,} in / {cleanup_usage.output_tokens:,} out)")

    # Step 4: Enrichment passes
    transcript_clean = transcript.model_copy(update={"segments": cleaned})

    print("4. Generating chapters...")
    chapters, ch_usage = generate_chapters(transcript_clean, provider)
    print(f"   {len(chapters)} chapters ({ch_usage.input_tokens:,} in / {ch_usage.output_tokens:,} out)")

    print("5. Generating takeaways...")
    takeaways, tk_usage = generate_takeaways(transcript_clean, provider)
    print(f"   {len(takeaways)} takeaways ({tk_usage.input_tokens:,} in / {tk_usage.output_tokens:,} out)")

    print("6. Generating summary...")
    summary, sm_usage = generate_summary(transcript_clean, provider)
    print(f"   ({sm_usage.input_tokens:,} in / {sm_usage.output_tokens:,} out)")

    total = (speaker_usage.total + cleanup_usage.total +
             ch_usage.total + tk_usage.total + sm_usage.total)
    print(f"\nTotal tokens: {total:,}")

    # Step 5: Strip embedded speaker prefixes from cleaned text to avoid double-rendering
    # (markdown renderer adds its own **Speaker:** from seg.speaker)
    import re as _re
    output_segments = []
    for seg in transcript_clean.segments:
        cleaned_text = _re.sub(r'^\[.*?\]\s*', '', seg.text)
        output_segments.append(seg.model_copy(update={"text": cleaned_text}))
    transcript_output = transcript_clean.model_copy(update={"segments": output_segments})

    # Step 6: Generate markdown
    slug = _slugify(transcript.source_title or "transcript")
    output_path = Path("output") / f"{slug}-{args.output_suffix}-{args.provider}.md"
    md_text = generate_markdown(transcript_output, chapters=chapters, key_takeaways=takeaways, final_summary=summary)
    output_path.write_text(md_text)
    print(f"\nMarkdown → {output_path}")


if __name__ == "__main__":
    main()
