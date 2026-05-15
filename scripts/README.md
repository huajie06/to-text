# Scripts

## `merge_diarization.py`

Standalone merge of `transcript.json` + `diarization.json` → `diarization_transcript.json`.

Aligns whisper segments to pyannote diarization windows using **any-overlap** alignment with multi-speaker labels for overlapped regions.

```bash
uv run python scripts/merge_diarization.py output/{hash}-{slug}
```

Output: `output/{hash}-{slug}/diarization_transcript.json`

## `run_with_diarization.py`

Experiment: feed `diarization_transcript.json` (with speaker labels) into the LLM cleanup + enrichment passes. The speaker names are embedded into the segment text so the LLM can see who's speaking during cleanup, chapters, takeaways, and summary.

This tests whether diarization-based speaker labels in the prompt text produce better quality output than the default pipeline (which only includes speaker hints in metadata context).

```bash
uv run python scripts/run_with_diarization.py output/{hash}-{slug} --provider deepseek
```

Output: `output/{slug}-with-diarization-{provider}.md`

### How it works

1. Loads `diarization_transcript.json` (has `SPEAKER_00`, `SPEAKER_00_SPEAKER_01` labels)
2. Maps IDs to real names via `map_speaker_ids()` LLM call
3. Prepends `[Speaker Name]` to each segment's text
4. Runs cleanup, chapters, takeaways, summary passes (LLM sees speaker names inline)
5. Strips `[Speaker Name]` prefixes from output segments (markdown renderer adds its own)
6. Writes a differentiated `.md` file for side-by-side comparison

### Compare outputs

| Run | File |
|---|---|
| Default pipeline (LLM-only speakers) | `output/{slug}-deepseek.md` |
| Diarization pipeline (full pipeline) | `examples/example2/{slug}-diarize.md` |
| Diarization transcript → LLM passes | `output/{slug}-with-diarization-deepseek.md` |
