"""Planted-answer tests for Test mode's test sample (#4527; ``vtscore/docs/packages/training.md``).

A synthetic ranking with a known per-band precision and a known count of
positives below the line, seeded.  Pinned here: the bands on both sides of
the line, the rounds (uniform, without replacement, a census of a small
band, each pick's band recorded), the joint estimators (exact once everything
is labelled; the ranges hold the planted truth at the stated level over many
seeds; the line re-estimated at every edge from the same draws; the
model-assisted count below the line and its flagged tail), the allocation
rule (the band holding the line first, then the band with the most
uncertainty by size; the walk below the line, to its budget with a class
model and to its first dry band without one), the phase machine and its
stops (width, budget, exhaustion, dry run, and *nothing to test* on an empty
line), and the ``test`` vote flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.datasets.vote_provenance import FLOWS, normalize_provenance
from vtscore.training.thresholds import (
    ABOVE,
    BELOW,
    CHECK_MIN_PICKS,
    DEFAULT_BUDGETS,
    PHASE_DONE,
    PHASE_MATCHES,
    PHASE_MISSES,
    PHASE_NOTHING,
    PHASES,
    STOP_BUDGET,
    STOP_DRY_RUN,
    STOP_EXHAUSTED,
    STOP_REASONS,
    STOP_WIDTH,
    TEST_ALPHA,
    TEST_PROVENANCE,
    Estimate,
    LineTest,
    LineBudgets,
    band_edges,
    fbeta_score,
    found_words,
    line_bands,
    line_phase,
)


def _planted(
    n: int = 600,
    line: int = 64,
    above_rate: float = 0.75,
    below: tuple[int, ...] = (2, 1, 1, 0, 0, 0, 0),
    seed: int = 42,
) -> tuple[list[int], set[int], np.ndarray]:
    """A ranking of *n* ids whose line keeps the top *line*: ``(ids, positives, posteriors)``.

    Above the line each band holds *above_rate* of its items as positives
    (rounded, at random ranks within the band); below it the *b*-th band
    holds ``below[b]`` positives.  The posteriors are the planted per-band
    rates, so the model is calibrated band by band and the unreached tail's
    point is exact.
    """
    rng = np.random.default_rng(seed)
    ids = list(range(1, n + 1))
    positives: set[int] = set()
    post = np.zeros(n)
    for lo, hi in zip(band_edges(line), band_edges(line)[1:]):
        count = int(round(above_rate * (hi - lo)))
        chosen = rng.choice(np.arange(lo, hi), size=count, replace=False)
        positives.update(int(ids[r]) for r in chosen)
        post[lo:hi] = count / (hi - lo)
    edges = band_edges(n - line)
    for b, (lo, hi) in enumerate(zip(edges, edges[1:])):
        count = below[b] if b < len(below) else 0
        chosen = rng.choice(np.arange(line + lo, line + hi), size=count, replace=False)
        positives.update(int(ids[r]) for r in chosen)
        post[line + lo : line + hi] = count / (hi - lo)
    return ids, positives, post


def _run(test: LineTest, positives: set[int], *, rounds: int | None = None) -> LineTest:
    """Vote every round until the test is done (or *rounds* rounds): a pick is a match iff it is planted."""
    taken = 0
    while rounds is None or taken < rounds:
        picks = test.draw()
        if not picks:
            break
        test.record({cid: cid in positives for cid in picks})
        taken += 1
    return test


def _truth(ids: list[int], positives: set[int], line: int, beta: float = 1.0) -> tuple[float, float, float]:
    tp = sum(1 for cid in ids[:line] if cid in positives)
    return tp / line, tp / len(positives), fbeta_score(tp, line, len(positives), beta)


class TestTheBands:
    def test_the_spot_checks_bands_above_the_line_and_the_same_doubling_below_it(self):
        bands = line_bands(600, 64)
        above = [(b.lo, b.hi) for b in bands if b.side == ABOVE]
        below = [(b.lo, b.hi) for b in bands if b.side == BELOW]
        assert above == [(0, 8), (8, 16), (16, 32), (32, 64)]
        assert below == [(64, 72), (72, 80), (80, 96), (96, 128), (128, 192), (192, 320), (320, 576), (576, 600)]
        assert [b.index for b in bands] == list(range(12))

    def test_either_side_may_be_empty(self):
        assert all(b.side == BELOW for b in line_bands(100, 0))
        assert all(b.side == ABOVE for b in line_bands(100, 100))
        assert line_bands(0, 0) == ()
        assert [(b.lo, b.hi) for b in line_bands(100, 150)] == [(0, 8), (8, 16), (16, 32), (32, 64), (64, 100)]

    def test_a_ranking_with_repeated_ids_or_a_beta_off_the_range_is_refused(self):
        with pytest.raises(ValueError, match="distinct"):
            LineTest.start([1, 1, 2], 2, 1.0)
        with pytest.raises(ValueError, match="beta"):
            LineTest.start([1, 2, 3], 2, 9.0)
        with pytest.raises(ValueError, match="align"):
            LineTest.start([1, 2, 3], 2, 1.0, posteriors=[0.1, 0.2])


class TestTheRounds:
    def test_a_round_is_five_uniform_picks_from_one_band_without_replacement(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        picks = test.draw()
        assert len(picks) == CHECK_MIN_PICKS == DEFAULT_BUDGETS.picks_per_round
        assert len(set(picks)) == len(picks)
        assert test.band == 3 and all(test.pick_band[cid] == 3 for cid in picks)  # the band holding the line first
        assert all(32 < cid <= 64 for cid in picks)
        assert test.draw() == picks, "a pending round is dealt again, not redrawn"

    def test_a_label_on_an_item_not_dealt_is_refused_and_a_partial_round_waits(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        picks = test.draw()
        with pytest.raises(ValueError, match="not this round"):
            test.record({999: True})
        assert test.record({picks[0]: True, picks[1]: False}) is False
        assert test.pending == picks[2:] and test.rounds == 0
        assert test.record({cid: False for cid in picks[2:]}) is True
        assert test.rounds == 1 and test.pending == () and test.band is None

    def test_the_down_key_takes_back_a_label_of_the_current_round(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        picks = test.draw()
        test.record({picks[0]: True})
        test.unrecord(picks[0])
        assert picks[0] not in test.labels and picks[0] in test.pending
        with pytest.raises(ValueError, match="not a label of this round"):
            test.unrecord(picks[1])

    def test_a_band_no_larger_than_a_round_is_censused_and_never_drawn_twice(self):
        ids, positives, post = _planted(n=40, line=5)
        test = LineTest.start(ids, 5, 1.0, posteriors=post, seed=42, budgets=LineBudgets(matches_width=0.01))
        picks = test.draw()
        assert sorted(picks) == [1, 2, 3, 4, 5]
        test.record({cid: cid in positives for cid in picks})
        assert test.exhausted(0) and test.band_range(0).lo == test.band_range(0).hi
        assert test.phase().matches_stop == STOP_EXHAUSTED  # a census has no width at all, but exhaustion is named

    def test_a_test_resumes_from_kept_picks_placed_by_rank(self):
        ids, positives, post = _planted()
        first = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42), positives, rounds=3)
        resumed = LineTest.start(ids, 64, 1.0, posteriors=post, seed=7, labels=first.labels)
        assert resumed.pick_band == first.pick_band
        assert resumed.estimates() == first.estimates()
        with pytest.raises(ValueError, match="not on this ranking"):
            LineTest.start(ids, 64, 1.0, labels={99999: True})


class TestTheEstimators:
    def test_a_fully_labelled_test_is_exact(self):
        ids, positives, post = _planted(n=48, line=16)
        test = LineTest.start(ids, 16, 1.0, posteriors=post, seed=42)
        for cid in ids:
            test.pick_band[cid] = test.band_of(cid) or 0
            test.labels[cid] = cid in positives
        test._invalidate()
        est = test.estimates()
        precision, recall, fb = _truth(ids, positives, 16)
        assert (est.precision.point, est.precision.lo, est.precision.hi) == pytest.approx((precision,) * 3)
        assert (est.recall.point, est.recall.lo, est.recall.hi) == pytest.approx((recall,) * 3)
        assert (est.fbeta.point, est.fbeta.lo, est.fbeta.hi) == pytest.approx((fb,) * 3)
        assert est.tail_from_model is False and est.tail_positives == 0.0
        assert est.labelled == 48

    def test_before_any_pick_the_precision_range_is_the_prior_and_the_tail_is_the_model(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        est = test.estimates()
        assert est.precision.lo < 0.2 and est.precision.hi > 0.8 and est.precision.point == pytest.approx(0.5, abs=0.02)
        assert est.tail_from_model is True
        assert est.tail_positives == pytest.approx(sum(post[64:]))
        assert est.positives_below.lo == est.positives_below.hi == pytest.approx(est.tail_positives)

    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_recall_and_fbeta_come_from_the_same_draws_as_precision(self, beta):
        ids, positives, post = _planted()
        test = _run(LineTest.start(ids, 64, beta, posteriors=post, seed=42), positives)
        est = test.estimates()
        assert est.beta == beta
        # The line's edge is the last band above it: its edge estimate is the line's own.
        edge = next(e for e in est.at_edges if e.count == 64)
        assert edge.side == ABOVE
        assert (edge.precision, edge.recall, edge.fbeta) == (est.precision, est.recall, est.fbeta)
        # F-beta at the point is the textbook one of the point counts, near enough for a ratio of draws.
        total = est.positives_above.point + est.positives_below.point
        assert est.fbeta.point == pytest.approx(fbeta_score(est.positives_above.point, 64, total, beta), abs=0.03)
        assert est.recall.point == pytest.approx(est.positives_above.point / total, abs=0.03)

    def test_the_same_draws_re_estimate_the_line_at_every_band_edge_on_both_sides(self):
        ids, positives, post = _planted()
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42), positives)
        est = test.estimates()
        assert [e.count for e in est.at_edges] == [8, 16, 32, 64, 72, 80, 96, 128, 192, 320, 576, 600]
        assert [e.side for e in est.at_edges] == [ABOVE] * 4 + [BELOW] * 8
        # Keeping everything ships every positive: recall 1, precision the prevalence.
        last = est.at_edges[-1]
        assert last.recall.point == pytest.approx(1.0) and last.recall.lo == pytest.approx(1.0)
        assert last.precision.hi < est.precision.lo, "a deeper line is less precise on this ranking"
        # Recall never falls as the line deepens.
        recalls = [e.recall.point for e in est.at_edges]
        assert recalls == sorted(recalls)

    def test_estimate_at_a_band_edge_is_the_edges_estimate(self):
        ids, positives, post = _planted()
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42), positives)
        for edge in test.estimates().at_edges:
            at = test.estimate_at(edge.count)
            assert (at.precision, at.recall, at.fbeta) == (edge.precision, edge.recall, edge.fbeta)

    def _top_heavy(self):
        """A line of 64 whose band 32-64 holds all its matches in its top half, and a model that says so."""
        n = 400
        ids = list(range(1, n + 1))
        positives = set(range(1, 33)) | set(range(33, 49))  # bands 0-2 all right; band 3 right only in 32-48
        post = np.zeros(n)
        post[:32] = 0.95
        post[32:48] = 0.9
        post[48:64] = 0.05
        post[64:] = 0.001
        return ids, positives, post

    def test_inside_a_band_the_split_follows_the_models_posteriors(self):
        """#4540: a shallower preset inside a top-heavy band reads near its truth, not at the band's average."""
        ids, positives, post = self._top_heavy()
        budgets = LineBudgets(matches_width=0.01, matches_picks=20, misses_picks=5)
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=budgets), positives)
        precision_at_48 = sum(1 for cid in ids[:48] if cid in positives) / 48
        assert precision_at_48 == 1.0
        by_model = test.estimate_at(48).precision
        flat = LineTest.start(ids, 64, 1.0, posteriors=None, seed=42, budgets=budgets, labels=test.labels)
        by_size = flat.estimate_at(48).precision
        assert by_model.point > by_size.point + 0.05, (by_model, by_size)
        assert by_model.holds(precision_at_48) or by_model.hi > 0.95

    def test_the_split_never_puts_more_matches_above_the_count_than_it_has_items(self):
        """A model sure of a band's top cannot read a preset as more than 100% right (the route caught 1.0037)."""
        ids, positives, post = self._top_heavy()
        post = post.copy()
        post[32:34] = 1.0  # all the model's mass on the band's first two items
        post[34:64] = 1e-6
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        for count in (33, 34, 36, 40, 48, 60):
            at = test.estimate_at(count)
            assert 0.0 <= at.precision.lo <= at.precision.hi <= 1.0, (count, at.precision)
            assert 0.0 <= at.recall.lo <= at.recall.hi <= 1.0, (count, at.recall)

    def test_a_pick_inside_the_band_counts_where_its_rank_falls(self):
        """A labelled right pick ranked above the count is counted as itself, not spread across the band."""
        ids, positives, post = self._top_heavy()
        test = LineTest.start(ids, 64, 1.0, posteriors=None, seed=42)
        # Label every item of band 3 (ranks 32-64): a census, so the split is exact.
        for cid in ids[32:64]:
            test.pick_band[cid] = 3
            test.labels[cid] = cid in positives
        test._invalidate()
        at = test.estimate_at(48)
        band3_above = sum(1 for cid in ids[32:48] if cid in positives)
        assert band3_above == 16
        # Bands 0-2 are unlabelled and drawn; band 3's contribution above 48 is exactly its 16 right picks.
        counts = test._base_draws()
        expected = (counts[0] + counts[1] + counts[2] + 16) / 48
        assert at.precision.point == pytest.approx(float(np.mean(expected)), rel=1e-9)

    def test_the_count_below_the_line_is_the_models_corrected_by_the_picks(self):
        ids, positives, post = _planted(below=(4, 0, 0, 0, 0, 0, 0))
        # A model that thinks the first band under the line is empty.
        post = post.copy()
        post[64:72] = 0.0
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=LineBudgets(matches_picks=20))
        _run(test, positives, rounds=4)
        assert test.phase().phase == PHASE_MISSES
        before = test.estimates().positives_below.point
        _run(test, positives, rounds=1)  # the first band under the line
        assert test.band_counts(4)[1] == 5 and test.band_counts(4)[2] >= 1
        after = test.estimates()
        assert after.positives_below.point > before, "the picks corrected a model that counted nothing there"
        assert after.tail_from_model is True and after.tail_positives == pytest.approx(sum(post[72:]))

    def test_without_a_model_the_tail_counts_nothing_and_is_flagged(self):
        ids, positives, _ = _planted()
        test = LineTest.start(ids, 64, 1.0, seed=42)
        est = test.estimates()
        assert est.tail_from_model is True and est.tail_positives == 0.0

    @pytest.mark.parametrize("above_rate", [0.3, 0.6, 0.9])
    def test_the_precision_range_holds_the_truth_at_the_stated_level_over_many_seeds(self, above_rate):
        """A 95% range on the matches phase's picks, with the phase stopped by its own rule."""
        held = 0
        seeds = 60
        for seed in range(seeds):
            ids, positives, post = _planted(above_rate=above_rate, seed=seed)
            budgets = LineBudgets(matches_width=0.20, matches_picks=40, misses_picks=0, draws=1500)
            test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=seed, budgets=budgets)
            _run(test, positives)
            assert test.phase().matches_stop in (STOP_WIDTH, STOP_BUDGET)
            precision, _, _ = _truth(ids, positives, 64)
            held += test.estimates().precision.holds(precision)
        assert held / seeds >= 1.0 - TEST_ALPHA - 0.05, f"{held} of {seeds} ranges held the truth"

    def test_the_recall_range_holds_the_truth_when_the_model_is_calibrated(self):
        held = 0
        seeds = 40
        for seed in range(seeds):
            ids, positives, post = _planted(seed=seed)
            budgets = LineBudgets(matches_width=0.20, draws=1500)
            test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=seed, budgets=budgets), positives)
            _, recall, _ = _truth(ids, positives, 64)
            held += test.estimates().recall.holds(recall)
        assert held / seeds >= 1.0 - TEST_ALPHA - 0.05, f"{held} of {seeds} ranges held the truth"

    def test_the_estimates_are_a_function_of_the_labels_not_their_order(self):
        ids, positives, post = _planted()
        a = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=1), positives, rounds=4)
        labels = dict(reversed(list(a.labels.items())))
        b = LineTest.start(ids, 64, 1.0, posteriors=post, seed=1, labels=labels)
        assert a.estimates() == b.estimates() and a.phase() == b.phase()


class TestTheFoundWords:
    @pytest.mark.parametrize(
        ("lo", "hi", "words"),
        [
            (0, 0.2, "few of them found"),
            (0.1, 0.4, "about a quarter of them found"),
            (0.3, 0.7, "about half of them found"),
            (0.6, 0.9, "about three quarters of them found"),
            (0.8, 1, "nearly all of them found"),
        ],
    )
    def test_a_range_reads_at_its_midpoint(self, lo, hi, words):
        assert found_words(Estimate((lo + hi) / 2, lo, hi)) == words

    def test_the_cuts_are_the_frontends(self):
        """``foundWords`` in ``frontend/src/app/utils/line-balance.ts`` cuts at 15, 37.5, 62.5 and 87.5 percent."""
        assert found_words(0.149) == "few of them found"
        assert found_words(0.15) == found_words(0.374) == "about a quarter of them found"
        assert found_words(0.375) == found_words(0.624) == "about half of them found"
        assert found_words(0.625) == found_words(0.874) == "about three quarters of them found"
        assert found_words(0.875) == found_words(1.0) == "nearly all of them found"


class TestTheAllocationRule:
    def test_above_the_line_every_band_once_from_the_line_upward(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        order = []
        for _ in range(4):
            picks = test.draw()
            order.append(test.pick_band[picks[0]])
            test.record({cid: cid in positives for cid in picks})
        assert order == [3, 2, 1, 0]

    def test_then_the_band_that_holds_the_most_uncertainty_by_size(self):
        """After the first pass, the largest mixed band (32 items, 3 of 5 right) beats the small settled ones."""
        ids, positives, post = _planted(above_rate=1.0)
        # The top three bands are all positives, the band holding the line is half and half.
        positives = {cid for cid in positives if cid <= 32} | set(range(33, 49))
        budgets = LineBudgets(matches_width=0.01, matches_picks=40)  # room past the first pass
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=budgets)
        _run(test, positives, rounds=4)
        assert test.phase().phase == PHASE_MATCHES
        shrink = {b.index: test.expected_shrink(b.index) for b in test.above}
        assert max(shrink, key=lambda b: shrink[b]) == 3 and test.next_band() == 3
        assert shrink[3] > 0.0

    def test_a_band_with_nothing_left_to_draw_shrinks_nothing_and_is_skipped(self):
        ids, positives, post = _planted(n=200, line=8)
        test = LineTest.start(ids, 8, 1.0, posteriors=post, seed=42, budgets=LineBudgets(matches_width=0.01))
        _run(test, positives, rounds=1)  # 5 of 8
        _run(test, positives, rounds=1)  # the other 3: a census
        assert test.exhausted(0) and test.expected_shrink(0) == 0.0
        assert test.phase().matches_stop == STOP_EXHAUSTED and test.phase().phase == PHASE_MISSES

    def test_below_the_line_the_first_band_under_it_first_then_one_band_deeper_a_round(self):
        ids, positives, post = _planted(
            below=(8, 8, 1, 0, 0, 0, 0)
        )  # the first two bands under the line are all matches
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=LineBudgets(matches_picks=20))
        _run(test, positives, rounds=4)
        assert test.phase().phase == PHASE_MISSES and test.misses_walk() == (4, None)
        _run(test, positives, rounds=1)
        assert test.misses_walk() == (5, None)
        _run(test, positives, rounds=1)
        assert test.misses_walk() == (6, None)
        assert sorted(test.pick_band[cid] for cid in test.labels if test.pick_band[cid] >= 4) == [4] * 5 + [5] * 5

    def test_with_a_class_model_a_dry_band_does_not_end_the_walk(self):
        """#4523: a walk that stops early leaves the tail to the model's point, and its recall range rarely holds."""
        ids, positives, post = _planted(below=(0, 0, 0, 0, 0, 0, 0))
        budgets = LineBudgets(matches_picks=20, misses_picks=30)
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=budgets)
        _run(test, positives, rounds=5)
        assert test.band_counts(4) == (8, 5, 0) and test.band_mass(4) == 0.0
        assert test.misses_walk() == (5, None) and test.phase().phase == PHASE_MISSES
        _run(test, positives)
        assert test.phase().misses_stop == STOP_BUDGET and test.phase().picks_below == 30

    def test_without_a_class_model_a_dry_band_is_a_dry_run_that_ends_the_walk(self):
        """With no model every band's mass is nothing, so the first band with no match ends the walk."""
        ids, positives, _ = _planted(below=(0, 0, 0, 0, 0, 0, 0))
        test = LineTest.start(ids, 64, 1.0, seed=42, budgets=LineBudgets(matches_picks=20, misses_width=0.01))
        _run(test, positives, rounds=5)
        assert test.band_counts(4) == (8, 5, 0) and test.band_mass(4) == 0.0
        assert test.misses_walk() == (None, STOP_DRY_RUN)
        assert test.phase().phase == PHASE_DONE and test.phase().misses_stop == STOP_DRY_RUN

    def test_without_a_class_model_a_band_that_turns_up_a_match_keeps_the_walk_going(self):
        ids, positives, _ = _planted(below=(8, 8, 1, 0, 0, 0, 0))
        test = LineTest.start(ids, 64, 1.0, seed=42, budgets=LineBudgets(matches_picks=20, misses_width=0.01))
        _run(test, positives, rounds=5)
        assert test.band_counts(4)[2] == 5
        assert test.misses_walk() == (5, None)

    def test_the_walk_is_exhausted_past_the_last_band_and_with_nothing_below_the_line(self):
        ids, positives, post = _planted(n=80, line=64, below=(8, 8))
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=LineBudgets(matches_picks=20))
        _run(test, positives)
        assert test.phase().misses_stop == STOP_EXHAUSTED and test.phase().phase == PHASE_DONE
        whole = LineTest.start(ids[:64], 64, 1.0, seed=42, budgets=LineBudgets(matches_picks=20))
        assert whole.misses_walk() == (None, STOP_EXHAUSTED)
        _run(whole, positives)
        assert whole.phase().misses_stop == STOP_EXHAUSTED and whole.phase().picks_below == 0


class TestThePhaseMachine:
    def test_the_vocabulary(self):
        assert PHASES == ("score", "matches", "misses", "done", "nothing")
        assert STOP_REASONS == ("width", "budget", "exhausted", "dry_run")

    def test_nothing_to_test_when_the_line_keeps_fewer_items_than_one_round(self):
        ids, positives, post = _planted()
        for line in (0, 4):
            test = LineTest.start(ids, line, 1.0, posteriors=post, seed=42)
            assert test.nothing_to_test and test.phase().phase == PHASE_NOTHING and test.phase().done
            assert test.draw() == () and test.next_band() is None
            assert test.as_dict()["estimates"] is None
        assert not LineTest.start(ids, 5, 1.0, posteriors=post).nothing_to_test

    def test_matches_then_misses_then_done(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        seen = [test.phase().phase]
        while test.draw():
            test.record({cid: cid in positives for cid in test.pending})
            if test.phase().phase != seen[-1]:
                seen.append(test.phase().phase)
        assert seen == [PHASE_MATCHES, PHASE_MISSES, PHASE_DONE]
        assert test.phase().done and test.next_band() is None

    def test_the_matches_phase_ends_on_width_once_every_band_above_the_line_has_a_round(self):
        """#4539: the pooled prior can meet a loose target on one band's picks; a band no pick has seen still waits."""
        ids, positives, post = _planted(above_rate=1.0)
        budgets = LineBudgets(matches_width=0.7)  # a target the prior alone does not meet, but one round does
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=budgets)
        before = test.phase()
        assert before.phase == PHASE_MATCHES and before.matches_width is not None and before.matches_width > 0.7
        _run(test, positives, rounds=1)
        early = test.phase()
        assert early.phase == PHASE_MATCHES and early.matches_stop is None, "three bands above the line unseen"
        _run(test, positives, rounds=len(test.above) - 1)
        report = test.phase()
        assert all(test.audited(b.index) for b in test.above)
        assert report.matches_stop == STOP_WIDTH and report.phase == PHASE_MISSES
        assert report.matches_width is not None and report.matches_width <= 0.7

    def test_the_matches_phase_ends_on_budget(self):
        ids, positives, post = _planted(above_rate=0.5)
        budgets = LineBudgets(matches_width=0.01, matches_picks=15)
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=budgets)
        _run(test, positives, rounds=2)
        assert test.phase().phase == PHASE_MATCHES
        _run(test, positives, rounds=1)
        assert test.phase().matches_stop == STOP_BUDGET and test.phase().picks_above == 15

    def test_the_matches_phase_ends_on_exhaustion(self):
        ids, positives, post = _planted(n=100, line=8, above_rate=0.5)
        budgets = LineBudgets(matches_width=0.01, matches_picks=400)
        test = _run(LineTest.start(ids, 8, 1.0, posteriors=post, seed=42, budgets=budgets), positives, rounds=2)
        assert test.phase().matches_stop == STOP_EXHAUSTED and test.phase().picks_above == 8

    def test_with_a_class_model_the_misses_phase_ends_on_its_budget_or_the_bands_running_out(self):
        """#4523: the recall range's width never stops a walk with a model; the walk runs to its budget."""
        ids, positives, post = _planted(below=(8, 8, 8, 8, 0, 0, 0))
        width = LineBudgets(matches_picks=20, misses_width=1.0, misses_picks=30)  # met by any range: not a stop
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=width), positives)
        assert test.phase().misses_stop == STOP_BUDGET and test.phase().picks_below == 30
        budget = LineBudgets(matches_picks=20, misses_picks=10)
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=budget), positives)
        assert test.phase().misses_stop == STOP_BUDGET and test.phase().picks_below == 10
        walk = LineBudgets(matches_picks=20, misses_picks=400)  # past the dry bands, to the corpus's end
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42, budgets=walk), positives)
        assert test.phase().misses_stop == STOP_EXHAUSTED and test.phase().picks_below == 5 * len(test.below)

    def test_without_a_class_model_the_misses_phase_also_ends_on_width_and_a_dry_run(self):
        ids, positives, _ = _planted(below=(8, 8, 8, 8, 0, 0, 0))
        width = LineBudgets(matches_picks=20, misses_width=0.5)
        test = _run(LineTest.start(ids, 64, 1.0, seed=42, budgets=width), positives)
        assert test.phase().misses_stop == STOP_WIDTH and test.phase().picks_below == 5
        budget = LineBudgets(matches_picks=20, misses_width=0.001, misses_picks=10)
        test = _run(LineTest.start(ids, 64, 1.0, seed=42, budgets=budget), positives)
        assert test.phase().misses_stop == STOP_BUDGET and test.phase().picks_below == 10
        walk = LineBudgets(matches_picks=20, misses_width=0.001, misses_picks=400)
        test = _run(LineTest.start(ids, 64, 1.0, seed=42, budgets=walk), positives)
        assert test.phase().misses_stop == STOP_DRY_RUN and test.phase().picks_below == 25

    def test_the_defaults_are_the_values_4523_and_4540_priced(self):
        """0.20 / 20 above the line (#4540's split budget), 40 picks below it (#4523)."""
        assert (DEFAULT_BUDGETS.matches_width, DEFAULT_BUDGETS.matches_picks) == (0.20, 20)
        assert DEFAULT_BUDGETS.misses_picks == 40 and DEFAULT_BUDGETS.picks_per_round == CHECK_MIN_PICKS == 5

    def test_the_phase_is_a_pure_function_of_the_sample_and_the_budgets(self):
        ids, positives, post = _planted()
        test = _run(LineTest.start(ids, 64, 1.0, posteriors=post, seed=42), positives, rounds=6)
        assert line_phase(test) == test.phase()
        tighter = LineTest.start(
            ids, 64, 1.0, posteriors=post, labels=test.labels, budgets=LineBudgets(matches_width=0.01, matches_picks=40)
        )
        assert tighter.phase().phase == PHASE_MATCHES and test.phase().phase != PHASE_MATCHES

    def test_the_budgets_are_validated(self):
        with pytest.raises(ValueError, match="width"):
            LineBudgets(matches_width=0.0)
        with pytest.raises(ValueError, match="budget"):
            LineBudgets(matches_picks=0)
        with pytest.raises(ValueError, match="round"):
            LineBudgets(picks_per_round=0)
        with pytest.raises(ValueError, match="alpha"):
            LineBudgets(alpha=1.0)


class TestTheWireShape:
    def test_as_dict_carries_the_phase_the_round_the_bands_and_the_estimates(self):
        ids, positives, post = _planted()
        test = LineTest.start(ids, 64, 1.0, posteriors=post, seed=42)
        picks = test.draw()
        d = test.as_dict()
        assert d["phase"] == PHASE_MATCHES and d["round"] == 1 and d["picks"] == list(picks)
        assert d["band"] == {"index": 3, "side": ABOVE, "lo": 33, "hi": 64}
        assert d["line_count"] == 64 and d["size"] == 600 and d["beta"] == 1.0
        assert len(d["bands"]) == 12 and d["bands"][0]["range"] is None
        assert set(d["estimates"]) >= {"precision", "recall", "fbeta", "found", "at_edges", "tail_from_model"}
        assert d["budgets"] == DEFAULT_BUDGETS.as_dict()
        assert d["class_model"] is True and LineTest.start(ids, 64, 1.0, seed=42).as_dict()["class_model"] is False
        _run(test, positives)
        d = test.as_dict()
        assert d["phase"] == PHASE_DONE and d["picks"] == [] and d["band"] is None
        assert d["report"]["matches_stop"] in STOP_REASONS and d["report"]["misses_stop"] in STOP_REASONS
        assert d["bands"][3]["labelled"] >= 5 and d["bands"][3]["range"]["labelled"] == d["bands"][3]["labelled"]


class TestTheVoteFlow:
    def test_a_test_vote_has_its_own_flow(self):
        assert "test" in FLOWS and TEST_PROVENANCE == {"flow": "test"}
        assert normalize_provenance(TEST_PROVENANCE) == {"v": 1, "flow": "test"}
