# PodBook Next Steps (2026-05-13)

## Done today

- Removed glossary feature entirely (summarize.py, markdown.py, pipeline.py, CLI, tests)
- Created `transcript/diarize.py` — pyannote diarization + max-overlap segment alignment
- Added `map_speaker_ids()` to `ai/speakers.py` — maps SPEAKER_XX → real names via one LLM call
- Integrated diarization into pipeline Phase 1.5 (audio path → diarize; VTT path → LLM fallback)
- Added `--force-diarize` CLI flag and `diarize` optional dep in pyproject.toml

## Remaining from plan

### 1. Markdown collapse — consecutive same-speaker segments (highest ROI, do first)

In `ebook/markdown.py`, the segment rendering loop emits a `**Speaker:**` header for every segment.
Change it to emit a new paragraph only on speaker change, concatenating same-speaker segments with a space.
No capitalization logic — just space-join.

Affects three rendering sites in `generate_markdown`:
- Chapter content with mapped segments (line ~64)
- Chapter outline fallback (line ~76)
- Raw transcript fallback (line ~84)
Also fix `segments_to_markdown` the same way.

### 4. Prompt caching audit

Verify Claude cache hits are actually firing. Quick check:
```bash
# Add a debug log line in providers/anthropic.py to print cache_read_input_tokens per call
# Run cleanup on a real episode and confirm 2nd+ chunks show cache_read > 0
```
For DeepSeek: check `prompt_tokens_details.cached_tokens` in the OpenAI provider response.

### 6. Cleanup chunk size test

Current chunk size: ~3000 words. The plan suggests testing 1200 words.
- Run cleanup on one cached transcript at 3k vs 1.2k
- Compare drift rate (segments lost/merged) and total tokens
- Decision: is smaller chunks worth the extra API calls?

## pyannote first-run checklist

Before testing on a real episode:
- [ ] `uv sync --extra diarize`
- [ ] Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1
- [ ] Accept model terms at https://hf.co/pyannote/segmentation-3.0
- [ ] `export HUGGINGFACE_TOKEN=hf_...`
- [ ] Test: `podbook build --speakers <youtube-url-with-two-hosts>`
- [ ] Check `output/<hash>/diarization.json` was created and cached
- [ ] Verify speaker names in output markdown

For VTT-path force: `podbook build --speakers --force-diarize <url>`

## Open questions

- **Chunk size for cleanup** — 3k or 1.2k? Run the A/B first.
- **Speaker mapping confidence** — currently falls back to Host / Guest N. Good enough for now.
- **Diarization on non-YouTube audio sources** — untested; should work since it just needs a WAV path.
