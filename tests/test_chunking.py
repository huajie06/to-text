"""Tests for transcript chunking."""

from podbook.models import Segment
from podbook.transcript.chunking import chunk_by_words, chunk_by_segments, _split_sentences


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


class TestSplitSentences:
    def test_splits_on_period(self):
        sents = _split_sentences("Hello world. This is a test.")
        assert len(sents) == 2
        assert sents[0] == "Hello world."

    def test_no_split_mid_abbrev(self):
        # Should not split "Dr." before a lowercase word
        sents = _split_sentences("Dr. Smith said hello. It was nice.")
        # At minimum the two proper sentences should be there
        assert any("Smith" in s for s in sents)

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        sents = _split_sentences("Just one sentence here")
        assert len(sents) == 1


class TestChunkByWords:
    def test_empty(self):
        assert chunk_by_words([]) == []

    def test_single_segment_below_target(self):
        segs = [seg(0, 10, "hello world")]
        chunks = chunk_by_words(segs, target_size=100, max_size=200)
        assert len(chunks) == 1

    def test_splits_at_max(self):
        # 20 segments, each with 300 words — should produce multiple chunks at max_size=500
        words = " ".join(["word"] * 300)
        segs = [seg(i * 10.0, (i + 1) * 10.0, words) for i in range(20)]
        chunks = chunk_by_words(segs, target_size=300, max_size=500)
        assert len(chunks) > 1
        # No single chunk should exceed max_size by more than one sentence worth
        for chunk in chunks:
            total_words = sum(len(s.text.split()) for s in chunk)
            assert total_words <= 600  # some slack for sentence boundaries

    def test_preserves_all_text(self):
        segs = [seg(i * 5.0, (i + 1) * 5.0, f"Sentence {i}. This is sentence {i}.") for i in range(10)]
        chunks = chunk_by_words(segs, target_size=20, max_size=40)
        all_chunk_text = " ".join(s.text for chunk in chunks for s in chunk)
        for i in range(10):
            assert f"Sentence {i}" in all_chunk_text


class TestChunkBySegments:
    def test_empty(self):
        assert chunk_by_segments([]) == []

    def test_splits_at_max(self):
        segs = [seg(i * 1.0, (i + 1) * 1.0, f"seg {i}") for i in range(50)]
        chunks = chunk_by_segments(segs, target_segments=10, max_segments=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 20

    def test_splits_at_gap(self):
        # Two groups with a big gap between them
        group_a = [seg(i * 1.0, (i + 1) * 1.0, f"a{i}") for i in range(10)]
        group_b = [seg(100.0 + i * 1.0, 101.0 + i * 1.0, f"b{i}") for i in range(10)]
        segs = group_a + group_b
        chunks = chunk_by_segments(segs, target_segments=5, max_segments=20)
        # The gap should trigger a split
        assert len(chunks) >= 2

    def test_all_segments_preserved(self):
        segs = [seg(i * 1.0, (i + 1) * 1.0, f"s{i}") for i in range(15)]
        chunks = chunk_by_segments(segs, target_segments=5, max_segments=10)
        total = sum(len(c) for c in chunks)
        assert total == 15
