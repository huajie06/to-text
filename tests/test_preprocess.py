"""Tests for transcript segment classification."""

import pytest
from podbook.models import Segment
from podbook.transcript.preprocess import (
    classify_segments,
    filter_content,
    content_ratio,
    SegmentLabel,
)


def seg(text: str, start: float = 0.0, end: float = 5.0) -> Segment:
    return Segment(start=start, end=end, text=text)


class TestAdDetection:
    def test_sponsor_by(self):
        labeled = classify_segments([seg("This episode is sponsored by Acme Corp.")])
        assert labeled[0][1] == SegmentLabel.AD

    def test_brought_to_you(self):
        labeled = classify_segments([seg("Brought to you by our partners at GreatCo.")])
        assert labeled[0][1] == SegmentLabel.AD

    def test_promo_code(self):
        labeled = classify_segments([seg("Use promo code PODCAST for 20% off.")])
        assert labeled[0][1] == SegmentLabel.AD

    def test_free_trial(self):
        labeled = classify_segments([seg("Sign up for a free trial today.")])
        assert labeled[0][1] == SegmentLabel.AD

    def test_normal_content_not_ad(self):
        labeled = classify_segments([seg("The study found a significant effect on brain function.")])
        assert labeled[0][1] == SegmentLabel.CONTENT


class TestSelfPromoDetection:
    def test_like_and_subscribe(self):
        labeled = classify_segments([seg("Please like and subscribe to the channel.")])
        assert labeled[0][1] == SegmentLabel.SELF_PROMO

    def test_leave_review(self):
        labeled = classify_segments([seg("Leave a 5-star review on Apple Podcasts.")])
        assert labeled[0][1] == SegmentLabel.SELF_PROMO

    def test_share_episode(self):
        labeled = classify_segments([seg("Share this episode with a friend.")])
        assert labeled[0][1] == SegmentLabel.SELF_PROMO


class TestMetaDetection:
    def test_thanks_for_listening(self):
        labeled = classify_segments([seg("Thanks for listening to today's episode.")])
        assert labeled[0][1] == SegmentLabel.META

    def test_coming_up(self):
        labeled = classify_segments([seg("Coming up on today's show, we'll discuss AI.")])
        assert labeled[0][1] == SegmentLabel.META

    def test_welcome_back(self):
        labeled = classify_segments([seg("Welcome back to another episode of the show.")])
        assert labeled[0][1] == SegmentLabel.META


class TestFillerDetection:
    def test_heavy_filler(self):
        labeled = classify_segments([seg("Um uh you know like um I mean sort of like okay.")])
        assert labeled[0][1] == SegmentLabel.FILLER

    def test_light_filler_still_content(self):
        labeled = classify_segments([
            seg("You know, the interesting thing about neural networks is how they learn.")
        ])
        assert labeled[0][1] == SegmentLabel.CONTENT


class TestFilterContent:
    def test_keeps_only_content(self):
        segs = [
            seg("Great substantive point about science."),
            seg("This episode is sponsored by Acme."),
            seg("Please subscribe to the channel."),
        ]
        labeled = classify_segments(segs)
        content = filter_content(labeled)
        assert len(content) == 1
        assert "science" in content[0].text

    def test_empty_input(self):
        assert filter_content([]) == []


class TestContentRatio:
    def test_all_content(self):
        segs = [seg("fact one"), seg("fact two"), seg("fact three")]
        labeled = classify_segments(segs)
        assert content_ratio(labeled) == pytest.approx(1.0)

    def test_empty(self):
        assert content_ratio([]) == 1.0
