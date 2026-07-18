"""AdaptiveLoadPacer: composite step-1 pacing + evidence-based rebasing.

The pacer fixes the two remaining miscalibration modes from issue #2556:
a bar frozen (ETA climbing) through archive extraction, and weights that
budget for a download/decode that never happens (cached archive, cached
embedded pkl) or happens at a very different speed than calibrated.
"""

from __future__ import annotations

import pytest

from vtscore.concurrency.progress import ProgressTracker
from vtscore.datasets.stages import _common
from vtscore.datasets.stages._common import AdaptiveLoadPacer

_OVERALL_EXTRAS = {"step": None, "total_steps": None, "overall": None, "eta_seconds": None}


class _Clock:
    """Deterministic stand-in for time.monotonic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(_common.time, "monotonic", c)
    return c


def _make(terms):
    tracker = ProgressTracker(extra_fields=dict(_OVERALL_EXTRAS))
    pacer = AdaptiveLoadPacer(tracker, terms)
    return tracker, pacer


def _overall(tracker) -> float:
    return tracker.get()["overall"]


TERMS = {"download": 10.0, "extract": 5.0, "load": 5.0, "embed": 20.0, "finalize": 10.0}


def test_extraction_advances_the_bar_after_download(clock):
    tracker, pacer = _make(TERMS)
    pacer.update("downloading", "dl", 0, 100, step=1)
    clock.advance(1.0)
    pacer.update("downloading", "dl", 100, 100, step=1)
    after_download = _overall(tracker)
    assert after_download > 0

    # Extraction reports on a different scale; the bar must keep moving.
    clock.advance(0.5)
    pacer.update("extracting", "unpack", 0, 400, step=1)
    clock.advance(1.0)
    pacer.update("extracting", "unpack", 200, 400, step=1)
    mid_extract = _overall(tracker)
    assert mid_extract > after_download
    clock.advance(1.0)
    pacer.update("extracting", "unpack", 400, 400, step=1)
    assert _overall(tracker) > mid_extract


def test_cached_archive_does_not_leap_the_bar(clock):
    # A cache-backed load never fires download/extract: the first status is
    # already "loading". The acquire slice must collapse instead of the bar
    # leaping straight to its end (the Caltech-101 0 -> 49% jump).
    tracker, pacer = _make(TERMS)
    pacer.update("loading", "Loading cached dataset...", 0, 0, step=2)
    assert _overall(tracker) == pytest.approx(0.0, abs=1e-9)
    pacer.update("loading", "Loading cached dataset...", 1, 2, step=2)
    # load's rebased share: 5 / (5 + 20 + 10) of the whole bar; half done.
    assert _overall(tracker) == pytest.approx(0.5 * 5 / 35, rel=1e-6)


def test_skipping_straight_to_embed_rebases_over_remaining_phases(clock):
    tracker, pacer = _make(TERMS)
    pacer.update("embedding", "embed", 10, 100, step=3)
    # embed's share of the remaining bar: 20 / (20 + 10).
    assert _overall(tracker) == pytest.approx(0.1 * 20 / 30, rel=1e-6)


def test_full_phase_sequence_is_monotone_and_ends_at_one(clock):
    tracker, pacer = _make(TERMS)
    seen = []

    def push(status, cur, tot, step):
        pacer.update(status, status, cur, tot, step=step)
        seen.append(_overall(tracker))
        clock.advance(0.7)

    push("downloading", 0, 100, 1)
    push("downloading", 60, 100, 1)
    push("extracting", 1, 10, 1)
    push("extracting", 10, 10, 1)
    push("loading", 0, 0, 2)
    push("loading", 3, 4, 2)
    push("embedding", 50, 100, 3)
    push("embedding", 100, 100, 3)
    pacer.update("loading", "finalize", 500, 1000, step=4)
    seen.append(_overall(tracker))
    pacer.update("loading", "finalize", 1000, 1000, step=4)
    seen.append(_overall(tracker))

    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(1.0, abs=1e-6)


def test_observed_rate_shrinks_an_overweighted_download(clock):
    # The model predicts a 100s download (10MB/s prior for a 1000MB archive),
    # but the link is 10x faster. Once the pacer trusts the observed rate the
    # acquire slice shrinks toward its real share.
    terms = {"download": 100.0, "extract": 5.0, "load": 5.0, "embed": 20.0, "finalize": 10.0}
    tracker, pacer = _make(terms)
    pacer.update("downloading", "dl", 0, 1000, step=1)
    clock.advance(5.0)  # 500MB in 5s -> projected 10s total, not 100s
    pacer.update("downloading", "dl", 500, 1000, step=1)
    assert pacer._terms["download"] == pytest.approx(10.0)
    clock.advance(5.0)
    pacer.update("downloading", "dl", 1000, 1000, step=1)
    # With the corrected term, a finished download consumed ~10/50 of the bar,
    # nowhere near the ~71% the stale prior would have claimed.
    assert _overall(tracker) < 0.35


def test_multi_archive_alternation_stays_monotone(clock):
    tracker, pacer = _make(TERMS)
    seen = []
    for _ in range(3):  # three download->extract cycles (e.g. TUT's zips)
        pacer.update("downloading", "dl", 0, 100, step=1)
        clock.advance(1.0)
        pacer.update("downloading", "dl", 100, 100, step=1)
        seen.append(_overall(tracker))
        pacer.update("extracting", "unpack", 0, 10, step=1)
        clock.advance(1.0)
        pacer.update("extracting", "unpack", 10, 10, step=1)
        seen.append(_overall(tracker))
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    assert seen[-1] < 1.0  # step 1 never claims the whole bar

    pacer.update("embedding", "embed", 100, 100, step=3)
    assert _overall(tracker) >= seen[-1]


def test_step1_counts_pass_through_unchanged(clock):
    # The UI formats download bytes (e.g. "0.86/1.14GB") from current/total;
    # the pacer must not replace them with a synthetic scale.
    tracker, pacer = _make(TERMS)
    pacer.update("downloading", "dl", 12345, 67890, step=1)
    snap = tracker.get()
    assert snap["current"] == 12345
    assert snap["total"] == 67890


def test_finalize_progress_composes_with_pacer(clock):
    from vtscore.datasets.stages._common import FinalizeProgress

    tracker, pacer = _make(TERMS)
    pacer.update("embedding", "embed", 100, 100, step=3)
    at_finalize_start = _overall(tracker)
    fin = FinalizeProgress(pacer)
    fin.begin("registry")
    fin.update("loading", "Saving to registry…", 1, 2)
    assert _overall(tracker) > at_finalize_start
    fin.begin("projection")
    fin.update("loading", "Projecting…", 1, 1)
    assert _overall(tracker) == pytest.approx(1.0, abs=1e-6)


def test_unmapped_status_passes_through(clock):
    tracker, pacer = _make(TERMS)
    pacer.update("someday-status", "x", 1, 2, step=None)
    snap = tracker.get()
    assert snap["overall"] is None
    assert snap["current"] == 1


def test_slow_phase_grows_its_span_and_the_bar_follows(clock):
    # GTZAN live run (2026-07-18): decode ran ~11x its calibrated term while
    # its bar span stayed tiny, so the rate-extrapolated overall ETA ballooned
    # to ~55min for a 3min load. The pacer must re-estimate the *current*
    # phase's term from its observed pace, not only the download's.
    terms = {"download": 0.0, "extract": 0.0, "load": 10.0, "embed": 60.0, "finalize": 5.0}
    tracker, pacer = _make(terms)
    pacer.update("loading", "decode", 0, 1000, step=2)
    static_share = 10.0 / 75.0
    # Half the files done, but ten times slower than the term predicted: the
    # projected phase total (100s) must displace the calibrated 10s, growing
    # load's share of the bar well past its static allocation.
    clock.advance(50.0)
    pacer.update("loading", "decode", 500, 1000, step=2)
    halfway = _overall(tracker)
    assert halfway > static_share
    # Finishing the phase still lands short of 100% — embed+finalize remain.
    clock.advance(50.0)
    pacer.update("loading", "decode", 1000, 1000, step=2)
    assert halfway < _overall(tracker) < 1.0
