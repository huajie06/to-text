"""Segment classification — detect ads, sponsors, meta-talk, and filler content.

Runs before chunking/LLM to filter out non-content segments, reducing token waste
and preventing the LLM from cleaning up ad copy as if it were podcast content.
"""

from __future__ import annotations

import re
from enum import Enum

from podbook.models import Segment


class SegmentLabel(str, Enum):
    CONTENT = "content"
    AD = "ad"               # sponsor message, product placement
    SELF_PROMO = "self_promo"  # subscribe, like, review, channel promo
    META = "meta"           # intro teaser, outro, "coming up", "closing tradition"
    FILLER = "filler"       # heavy filler words, cross-talk fragments
    UNKNOWN = "unknown"


# ── Pattern sets ──────────────────────────────────────────────────

AD_TRIGGERS = [
    r"sponsor(?:ed|s)?\s+(?:by|this)",
    r"brought\s+to\s+you\s+by",
    r"free\s+trial",
    r"use\s+(?:my|our|the)\s+link",
    r"discount\s+code",
    r"promo\s+code",
    r"sign\s*up\s+(?:now|today|for)",
    r"no\s+credit\s+card",
    r"money[-\s]back\s+guarantee",
    r"limited\s+time\s+offer",
    r"exclusive\s+(?:offer|deal|discount)",
    r"head\s+to\s+\w+\.com",
    r"visit\s+\w+\.com",
    r"check\s+out\s+\w+\.com",
    r"download\s+(?:our|the)\s+(?:app|free)",
    r"offer\s+code",
]

SELF_PROMO_TRIGGERS = [
    r"(?:please|don't\s+forget\s+to)\s*(?:hit\s+)?(?:\w+\s+)?subscrib",
    r"subscrib(?:e|ing|ed)\s+(?:button|to\s+(?:this|the|our|my)\s+(?:channel|show|podcast))",
    r"like\s+and\s+subscrib",
    r"leave\s+(?:a|us\s+a)\s+(?:5[-\s]?star\s+)?review",
    r"rate\s+(?:us|this|the)\s+(?:show|podcast)",
    r"keep\s+(?:this|the)\s+(?:show|podcast|channel)\s+(?:free|going)",
    r"help(?:ing)?\s+(?:us|me|the\s+team)\s+(?:keep|grow|continue)",
    r"hit\s+the\s+(?:bell|notification)",
    r"share\s+(?:this|the)\s+(?:episode|video|podcast)",
    r"tell\s+(?:your\s+)?(?:friends|family)\s+about",
    r"word\s+of\s+mouth",
    r"(?:significant|big|huge)\s+landmark",
    r"favor\s+(?:to|for)\s+(?:our|the|your)",
    r"check\s+(?:right\s+)?now\s+if\s+you",
]

META_TRIGGERS = [
    r"let'?s\s+get\s+(?:on\s+with|into|started)",
    r"without\s+further\s+ado",
    r"coming\s+up\s+(?:on|in|after)",
    r"we'?ll\s+be\s+right\s+back",
    r"stay\s+(?:tuned|with\s+us)",
    r"welcome\s+(?:back\s+)?to\s+(?:the|another)",
    r"thanks?\s+(?:for|so\s+much)\s+(?:for\s+)?(?:watching|listening|tuning|joining)",
    r"closing\s+tradition",
    r"question\s+(?:that\s+)?has\s+been\s+left\s+for",
    r"where\s+(?:do|can|should)\s+people\s+find\s+you",
    r"(?:link|linked)\s+(?:below|in\s+the\s+description)",
    r"(?:the\s+)?algorithm\s+(?:says|recommends|knows)",
    r"(?:this\s+video|check\s+this)\s+(?:is\s+the\s+perfect|out)",
    r"(?:you('?d| would)\s+like|you\s+might\s+love)",
    r"next\s+(?:episode|week|time|video)",
    r"(?:please\s+)?(?:check|hit|smash)\s+(?:out\s+)?(?:this|the)\s+(?:video|link|button)",
    r"i'?ll\s+(?:link|put|leave)\s+(?:all\s+of\s+)?that\s+(?:below|in\s+the\s+description)",
]

FILLER_TRIGGERS = [
    # Segments that are >50% filler by word count
]

# Words that skew filler ratio
FILLER_WORDS = {
    "um", "uh", "er", "ah", "mm", "hmm", "like", "you know",
    "i mean", "sort of", "kind of", "right", "okay", "so",
    "basically", "literally", "actually", "anyway",
}


def classify_segments(segments: list[Segment]) -> list[tuple[Segment, SegmentLabel]]:
    """Classify each segment as content, ad, self-promo, meta, or filler.

    Returns list of (segment, label) tuples. The original segments are unmodified.
    """
    labeled: list[tuple[Segment, SegmentLabel]] = []

    for seg in segments:
        text = seg.text.lower().strip()
        label = _classify_one(text)
        labeled.append((seg, label))

    # Second pass: contextual fixes
    labeled = _contextual_fixes(labeled)

    return labeled


def filter_content(
    labeled: list[tuple[Segment, SegmentLabel]],
    keep: set[SegmentLabel] | None = None,
) -> list[Segment]:
    """Return only segments matching the given labels.

    Default: keep only CONTENT segments.
    """
    if keep is None:
        keep = {SegmentLabel.CONTENT}

    return [seg for seg, label in labeled if label in keep]


def content_ratio(labeled: list[tuple[Segment, SegmentLabel]]) -> float:
    """Return the fraction of segments labeled as CONTENT."""
    if not labeled:
        return 1.0
    content_count = sum(1 for _, label in labeled if label == SegmentLabel.CONTENT)
    return content_count / len(labeled)


def label_summary(labeled: list[tuple[Segment, SegmentLabel]]) -> dict[SegmentLabel, int]:
    """Count segments per label."""
    counts: dict[SegmentLabel, int] = {}
    for _, label in labeled:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _classify_one(text: str) -> SegmentLabel:
    """Classify a single segment."""
    # Check ad triggers first (highest priority)
    for pattern in AD_TRIGGERS:
        if re.search(pattern, text):
            return SegmentLabel.AD

    # Self-promo
    for pattern in SELF_PROMO_TRIGGERS:
        if re.search(pattern, text):
            return SegmentLabel.SELF_PROMO

    # Meta
    for pattern in META_TRIGGERS:
        if re.search(pattern, text):
            return SegmentLabel.META

    # Filler check: if >50% of words are filler words
    words = [w.strip() for w in text.lower().split()]
    if words:
        filler_count = sum(1 for w in words if w in FILLER_WORDS)
        if filler_count / len(words) > 0.5:
            return SegmentLabel.FILLER

    return SegmentLabel.CONTENT


def _contextual_fixes(
    labeled: list[tuple[Segment, SegmentLabel]],
) -> list[tuple[Segment, SegmentLabel]]:
    """Apply contextual fixes to classification.

    - If a CONTENT segment is surrounded by 2+ AD or SELF_PROMO segments,
      it's likely part of the ad block.
    - First few segments of the transcript are often intro/meta.
    - Last few segments are often outro/meta.
    """
    if not labeled:
        return labeled

    result = list(labeled)
    n = len(result)

    # Expand ad/promo blocks: if a CONTENT segment has AD/SELF_PROMO on both
    # sides within a 2-segment window, reclassify it
    for i in range(1, n - 1):
        seg, cur_label = result[i]
        if cur_label != SegmentLabel.CONTENT:
            continue

        prev_labels = {result[j][1] for j in range(max(0, i - 2), i)}
        next_labels = {result[j][1] for j in range(i + 1, min(n, i + 3))}

        ad_types = {SegmentLabel.AD, SegmentLabel.SELF_PROMO}
        if (prev_labels & ad_types) and (next_labels & ad_types):
            result[i] = (seg, SegmentLabel.SELF_PROMO)

    # First 3% of transcript: if labeled CONTENT but feels like intro
    intro_end = max(1, int(n * 0.03))
    for i in range(intro_end):
        seg, label = result[i]
        if label == SegmentLabel.CONTENT:
            text = seg.text.lower()
            intro_signals = [
                "today we're going to talk about", "we're going to cover",
                "coming up", "in this episode",
                "we'll get through that", "what else have we got",
            ]
            if any(sig in text for sig in intro_signals):
                result[i] = (seg, SegmentLabel.META)

    # Last 5% of transcript: if labeled CONTENT but feels like outro
    outro_start = int(n * 0.95)
    for i in range(outro_start, n):
        seg, label = result[i]
        if label == SegmentLabel.CONTENT:
            text = seg.text.lower()
            outro_signals = [
                "thank you", "thanks for", "really appreciate",
                "hope to", "see you next", "tune in next",
            ]
            if any(sig in text for sig in outro_signals):
                result[i] = (seg, SegmentLabel.META)

    return result
