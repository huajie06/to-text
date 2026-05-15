# Diarization Merge Enhancements

The current `assign_speakers()` in `diarize.py` uses **any-overlap alignment**:
- Whisper segments (~1–3s) checked for any time overlap with each pyannote `(start, end, SPEAKER_XX)` window independently
- Overlaps both speakers → combined label (`SPEAKER_00_SPEAKER_01`)
- These become `Joe_Rogan_Theo_Von` in the final markdown — ugly but honest

Three enhancement directions:

---

## 1. Split whisper segments at diarization boundaries (recommended first step)

Instead of giving a 2s segment a combined label, split it at the timestamp where pyannote detected a speaker change.

**Before (current):**
```
[10.9-12.8] SPEAKER_00_SPEAKER_01: "who's a bad, pick a bad guy out of there?"
```

**After:**
```
[10.9-11.3] SPEAKER_01: "who's a bad,"
[11.3-12.8] SPEAKER_00: "pick a bad guy out of there?"
```

**Implementation sketch (`assign_speakers` in `diarize.py`):**

```python
def _collect_boundaries(diarization):
    """Collect all speaker-change timestamps from diarization."""
    boundaries = set()
    for ds, de, spk in diarization:
        boundaries.add(ds)
        boundaries.add(de)
    return sorted(b for b in boundaries if b > 0)

def assign_speakers(segments, diarization, separator="_"):
    # Pre-compute speaker per micro-window from diarization
    # Split each whisper segment at any internal boundary
    for seg in segments:
        internal = [b for b in boundaries if seg.start < b < seg.end]
        if not internal:
            # single speaker — normal assignment
        else:
            # split seg into sub-segments at each boundary
            for sub_start, sub_end in pairwise([seg.start] + internal + [seg.end]):
                speaker = speaker_at_time(sub_start, sub_end, diarization)
                new_segments.append(Segment(...))
```

**Tradeoffs:**
+ No combined labels in final markdown
+ ~30 lines of code, low risk
+ More accurate speaker attribution
- More segments (potentially 2–3x)
- Boundary accuracy limited by pyannote (can be off by ~200ms)
- Need to handle very short sub-segments (< 0.3s) — merge into neighbor

---

## 2. Embedding-based speaker clustering (best accuracy)

Pyannote produces speaker embeddings (d-vectors) per diarization window. Instead of time-alignment, use the embeddings directly:

1. Run whisper → get segments
2. Run pyannote diarization → get speaker embeddings per window
3. For each whisper segment, extract its audio crop and run through pyannote's embedding model
4. Cluster whisper segments by cosine similarity to diarization embeddings

This would correctly attribute short interjections even when fully time-overlapped with the dominant speaker, because the voice characteristics are different even if the timestamps overlap.

**Tradeoffs:**
+ Accuracy — works even when speakers fully overlap (interruptions, cross-talk)
+ No combined labels needed
- Adds a full forward pass through the embedding model per segment
- Clustering hyperparameters (threshold, number of speakers) need tuning
- Complex implementation — ~200 lines of new code in a separate module
- Embedding extraction from pyannote internals is not a public API

---

## 3. Voice-activity-based segmentation (foundational)

Use a VAD model (e.g. `silero-vad`) to produce pause-aware segment boundaries before transcription:

1. Run VAD on the audio → get `(start, end)` voice activity windows
2. Feed those window timestamps as "prompts" to whisper for segment alignment
3. Or: use VAD windows as post-processing to merge/split whisper segments

VAD detects silence gaps between speech, which naturally aligns with speaker turn boundaries. Multi-speaker detection is a harder VAD problem, but single-speaker VAD still captures turn-taking pauses.

**Tradeoffs:**
+ Segments naturally align with speaker turns
+ Reduces the granularity mismatch at source
- VAD misses overlapping speech
- VAD on this audio might produce very different boundaries than pyannote
- Significant refactor of the transcription pipeline
- `silero-vad` adds ~50MB dependency

---

## Recommendation

| Path | Effort | Impact | Risk |
|---|---|---|---|
| #1 Boundary splitting | ~30 lines | High (fixes combined labels) | Low |
| #2 Embedding clustering | ~200 lines | Highest (accurate attribution) | Medium |
| #3 VAD segmentation | ~150 lines across modules | High (natural alignment) | Medium-high |

**Start with #1.** It directly fixes the rendering problem (`Joe_Rogan_Theo_Von`), is simple to implement, and you can verify correctness by checking that split boundaries match human-perceptible speaker changes. If the boundaries turn out too noisy (e.g. splitting mid-word), fall back to the current combined-label approach for that segment.
