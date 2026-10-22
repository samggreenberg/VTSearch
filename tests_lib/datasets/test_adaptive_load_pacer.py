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


# Static-profile fallback: unitless pseudo-times (the CPU profile's ratios),
# not seconds. Only their ratios are meaningful.
PSEUDO_TERMS = {"download": 0.1875, "extract": 0.0625, "load": 0.10, "embed": 0.55, "finalize": 0.10}


def _make_pseudo(terms):
    tracker = ProgressTracker(extra_fields=dict(_OVERALL_EXTRAS))
    pacer = AdaptiveLoadPacer(tracker, terms, calibrated=False)
    return tracker, pacer


def test_pseudo_terms_keep_prior_ratios_when_a_phase_is_observed(clock):
    # Regression for #2615: with static fallback terms, a 30s observed model
    # load was written (in seconds) over its 0.10 pseudo-term, so its share of
    # the remaining bar became 30/(30+0.55+0.10) ~ 98% — the bar leapt to ~1.0
    # entering embed and the whole-job ETA pinned near zero for the entire
    # (dominant) embed phase. An observation must fix the scale, not the
    # ratios: embed keeps its prior ratio to the observed load term.
    tracker, pacer = _make_pseudo(PSEUDO_TERMS)
    pacer.update("loading", "model", 0, 100, step=2)
    clock.advance(30.0)
    pacer.update("loading", "model", 50, 100, step=2)  # projected 60s total
    # Load's share of the whole bar must stay near its prior ratio
    # (0.10 / 0.75 of the post-acquire remainder), not swallow it.
    assert pacer._terms["load"] == pytest.approx(60.0)
    assert pacer._terms["embed"] == pytest.approx(60.0 * 0.55 / 0.10)
    assert pacer._terms["finalize"] == pytest.approx(60.0 * 0.10 / 0.10)
    clock.advance(30.0)
    pacer.update("loading", "model", 100, 100, step=2)
    pacer.update("embedding", "e", 0, 200, step=3)
    # Entering embed, most of the bar must still be ahead of us: embed's prior
    # dominates the profile, so consumed stays well below ~50%.
    assert _overall(tracker) < 0.5


def test_pseudo_terms_unbudgeted_phase_does_not_grab_the_bar(clock):
    # A phase with a zero prior gives no seconds-per-unit scale; the pacer
    # must keep pacing by the remaining priors rather than hand the phase the
    # whole remaining span.
    terms = dict(PSEUDO_TERMS, load=0.0)
    tracker, pacer = _make_pseudo(terms)
    pacer.update("loading", "model", 0, 100, step=2)
    clock.advance(30.0)
    pacer.update("loading", "model", 50, 100, step=2)
    assert pacer._terms["embed"] == pytest.approx(0.55)
    pacer.update("embedding", "e", 0, 200, step=3)
    assert _overall(tracker) < 0.5


def test_calibrated_terms_still_replace_only_the_observed_phase(clock):
    # Seconds-mode behavior is unchanged: the observed phase's term is
    # replaced by its projection and the others keep their measured values.
    tracker, pacer = _make(TERMS)
    pacer.update("loading", "decode", 0, 100, step=2)
    clock.advance(10.0)
    pacer.update("loading", "decode", 50, 100, step=2)  # projected 20s vs 5s term
    assert pacer._terms["load"] == pytest.approx(20.0)
    assert pacer._terms["embed"] == pytest.approx(TERMS["embed"])
    assert pacer._terms["finalize"] == pytest.approx(TERMS["finalize"])


def test_conversion_after_load_keeps_the_bar_moving(clock):
    # #5176: a converted demo loads its source media (status "loading") and then
    # converts it (status "converting"), both on step 2. Sharing one phase, the
    # load drove its fraction to 1 and the whole conversion then ran on a parked
    # bar. Conversion gets its own sub-range of step 2, after the load.
    tracker, pacer = _make(TERMS)
    pacer.update("loading", "Loading 600 images", 600, 600, step=2)
    after_load = _overall(tracker)
    clock.advance(1.0)
    pacer.update("converting", "Converting 600 items", 0, 600, step=2)
    assert _overall(tracker) == pytest.approx(after_load)
    clock.advance(1.0)
    pacer.update("converting", "Converting", 300, 600, step=2)
    mid_convert = _overall(tracker)
    clock.advance(1.0)
    pacer.update("converting", "Converting", 600, 600, step=2)
    end_convert = _overall(tracker)
    assert after_load < mid_convert < end_convert < 1.0
    # Unpriced by the cost model, the conversion is budgeted like the embed
    # phase after it, so half of it covers a real stretch of the bar.
    assert pacer._terms["convert"] == pytest.approx(TERMS["embed"])
    assert mid_convert - after_load > 0.1
    assert tracker.get()["step"] == 2


def test_converted_load_sequence_is_monotone_and_ends_at_one(clock):
    # The full converted-demo order: load the sources, convert them, load the
    # target's embedding model (step 2 again), embed the outputs, finalize.
    tracker, pacer = _make(TERMS)
    seen, steps = [], []

    def push(status, cur, tot, step):
        pacer.update(status, status, cur, tot, step=step)
        seen.append(_overall(tracker))
        steps.append(tracker.get()["step"])
        clock.advance(0.7)

    push("downloading", 50, 100, 1)
    push("downloading", 100, 100, 1)
    push("loading", 300, 600, 2)
    push("loading", 600, 600, 2)
    push("converting", 200, 600, 2)
    push("converting", 600, 600, 2)
    push("loading", 0, 0, 2)
    push("loading", 50, 107, 2)
    push("loading", 107, 107, 2)
    push("embedding", 350, 700, 3)
    push("embedding", 700, 700, 3)
    pacer.update("loading", "finalize", 1000, 1000, step=4)
    seen.append(_overall(tracker))
    steps.append(tracker.get()["step"])

    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    assert steps == sorted(steps)
    assert seen[-1] == pytest.approx(1.0, abs=1e-6)


def test_a_load_that_never_converts_keeps_its_pacing(clock):
    # The convert term is seeded only when a conversion starts, so a plain
    # load's phases split the bar exactly as before.
    tracker, pacer = _make(TERMS)
    pacer.update("loading", "decode", 1, 2, step=2)
    assert pacer._terms["convert"] == 0.0
    assert _overall(tracker) == pytest.approx(0.5 * 5 / 35, rel=1e-6)


def test_pseudo_terms_conversion_moves_the_bar(clock):
    # Static-profile terms (no cost-model row: e.g. the face type) take the same
    # embed-sized prior, so the observation keeps a sane ratio.
    tracker, pacer = _make_pseudo(PSEUDO_TERMS)
    pacer.update("loading", "images", 10, 10, step=2)
    after_load = _overall(tracker)
    pacer.update("converting", "c", 0, 100, step=2)
    clock.advance(30.0)
    pacer.update("converting", "c", 50, 100, step=2)
    mid = _overall(tracker)
    clock.advance(30.0)
    pacer.update("converting", "c", 100, 100, step=2)
    assert after_load < mid < _overall(tracker) < 1.0
