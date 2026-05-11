"""Tests for transcript normalization."""

from podbook.models import Segment
from podbook.transcript.normalize import normalize, _merge_short, _fix_overlaps, _remove_empty


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


class TestRemoveEmpty:
    def test_removes_blank(self):
        segs = [seg(0, 1, "hello"), seg(1, 2, "   "), seg(2, 3, "world")]
        assert [s.text for s in _remove_empty(segs)] == ["hello", "world"]

    def test_empty_list(self):
        assert _remove_empty([]) == []


class TestMergeShort:
    def test_merges_below_threshold(self):
        segs = [seg(0, 1.0, "hello"), seg(1.0, 5.0, "world")]
        result = _merge_short(segs, min_duration=1.5)
        assert len(result) == 1
        assert "hello" in result[0].text
        assert "world" in result[0].text

    def test_keeps_long_segments(self):
        segs = [seg(0, 3.0, "hello"), seg(3.0, 6.0, "world")]
        result = _merge_short(segs, min_duration=1.5)
        assert len(result) == 2

    def test_empty_list(self):
        assert _merge_short([]) == []


class TestFixOverlaps:
    def test_fixes_overlap(self):
        segs = [seg(0, 5.0, "a"), seg(3.0, 8.0, "b")]
        result = _fix_overlaps(segs)
        assert result[1].start >= result[0].end

    def test_fixes_negative_duration(self):
        segs = [seg(5.0, 3.0, "bad")]
        result = _fix_overlaps(segs)
        assert result[0].end > result[0].start

    def test_no_change_on_clean(self):
        segs = [seg(0, 2.0, "a"), seg(2.0, 4.0, "b")]
        result = _fix_overlaps(segs)
        assert result[0].start == 0
        assert result[1].start == 2.0


class TestNormalize:
    def test_full_pipeline(self):
        segs = [
            seg(0, 0.5, "um"),      # short
            seg(0.5, 3.0, "hello"), # normal
            seg(3.0, 2.0, "oops"),  # negative duration
            seg(2.0, 5.0, "world"), # overlap after fix
            seg(5.0, 5.5, ""),      # empty
        ]
        result = normalize(segs)
        # Empty segment gone, no overlaps, durations positive
        assert all(s.text.strip() for s in result)
        assert all(s.end >= s.start for s in result)
        for i in range(1, len(result)):
            assert result[i].start >= result[i - 1].end

    def test_empty_input(self):
        assert normalize([]) == []
