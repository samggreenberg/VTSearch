"""The eval harness's typed-query runs score the withheld half as a typed-query detector's Test does (#4758).

Until the label quota, the app gives a detector made from a typed query that
query's own set, from the first vote.  "The Eval Default Arm IS the App"
(CLAUDE.md): so a default run that opens on a typed query writes a row at every
click, ``detector_tier`` is ``"query"`` until the quota and ``"trained"`` from
it, and the query rows are the text sort's set on the withheld half.  The
Train side never reads Test, so the trajectory is the arm before's
(``query_tier=False``: the Goods' centroid from the first Good).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.detectors.label_quota import TIER_CENTROID, TIER_QUERY, TIER_TRAINED, label_quota
from vtscore.eval.voting_iterations import simulate_voting_iterations

DIM = 12


def _world(n_per_cat: int = 40, seed: int = 0) -> tuple[dict[int, dict], dict[int, float]]:
    """Two categories and a typed query's cosines that rank ``alpha`` near the top, with misses."""
    rng = np.random.default_rng(seed)
    centres = {"alpha": rng.standard_normal(DIM), "beta": rng.standard_normal(DIM)}
    query = centres["alpha"] + 1.2 * rng.standard_normal(DIM)
    query /= np.linalg.norm(query)
    clips: dict[int, dict] = {}
    seed_scores: dict[int, float] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat // 4 else "beta"
        vec = centres[cat] + 0.9 * rng.standard_normal(DIM)
        vec = (vec / np.linalg.norm(vec)).astype(np.float32)
        clips[i + 1] = {"id": i + 1, "embeddings": {"emb": vec}, "category": cat}
        seed_scores[i + 1] = float(vec @ query)
    return clips, seed_scores


def _run(**kwargs):
    clips, seed_scores = _world()
    picks: list[dict] = []
    rows = simulate_voting_iterations(
        clips,
        "alpha",
        seed=0,
        max_steps=20,
        calibrate_count=2,
        spot_check="off",
        pick_sink=picks,
        seed_scores=seed_scores,
        **kwargs,
    )
    return rows, picks


@pytest.fixture(scope="module")
def typed_run():
    """The quota's hand-over, before #4736's hold: the tiers this class pins."""
    return _run(query_hold=0)


@pytest.fixture(scope="module")
def before_run():
    return _run(query_tier=False)


class TestATypedQueryRun:
    def test_rows_start_at_the_first_vote(self, typed_run):
        rows, _picks = typed_run
        assert [r["t"] for r in rows] == list(range(1, 21))

    def test_each_row_is_the_query_until_the_quota_and_the_head_from_it(self, typed_run):
        rows, _picks = typed_run
        for r in rows:
            trained = label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED
            assert r["detector_tier"] == (TIER_TRAINED if trained else TIER_QUERY), r["t"]
        assert {r["detector_tier"] for r in rows} == {TIER_QUERY, TIER_TRAINED}, "the run crosses the quota"

    def test_the_query_rows_are_one_set_whatever_the_votes(self, typed_run):
        """The typed query's set does not move with the votes: every query row scores the same."""
        rows, _picks = typed_run
        query = [r for r in rows if r["detector_tier"] == TIER_QUERY]
        assert len(query) >= 2
        for col in ("precision", "recall", "fbeta", "n_flagged", "average_precision"):
            assert len({r[col] for r in query}) == 1, col
        assert all(math.isfinite(r["recall"]) for r in query)

    def test_the_trajectory_and_the_trained_rows_are_the_arm_befores(self, typed_run, before_run):
        rows, picks = typed_run
        old, old_picks = before_run
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in old_picks]
        old_by_t = {r["t"]: r for r in old}
        trained = [r for r in rows if r["detector_tier"] == TIER_TRAINED]
        assert trained
        for r in trained:
            for col in ("precision", "recall", "fbeta", "auroc", "acq_threshold"):
                o = old_by_t[r["t"]][col]
                assert r[col] == o or (math.isnan(r[col]) and math.isnan(o)), (r["t"], col)

    def test_the_arm_before_gives_the_centroid_from_the_first_good(self, before_run):
        rows, picks = before_run
        first_good = min(p["t"] for p in picks if p["picked_label"] == 1)
        assert rows[0]["t"] == first_good
        assert TIER_QUERY not in {r["detector_tier"] for r in rows}
        assert TIER_CENTROID in {r["detector_tier"] for r in rows}


HOLD = 12


@pytest.fixture(scope="module")
def held_run():
    """The app's hold (#4736) at a count this short run crosses."""
    return _run(query_hold=HOLD)


def _base(rows):
    return [r for r in rows if not r.get("gmm_variant")]


class TestTheQueryHold:
    """Past the quota the typed query's set stays the untagged row until the hold; the head's rows ride tagged."""

    def test_the_default_is_the_apps_hold(self):
        import inspect

        from vtscore.detectors.label_quota import QUERY_HOLD_LABELS

        assert inspect.signature(simulate_voting_iterations).parameters["query_hold"].default == QUERY_HOLD_LABELS

    def test_the_picks_are_the_hand_overs_at_the_quota(self, held_run, typed_run):
        _rows, picks = held_run
        _old, old_picks = typed_run
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in old_picks]

    def test_one_untagged_row_a_click_the_query_until_the_hold(self, held_run):
        rows, _picks = held_run
        base = _base(rows)
        assert [r["t"] for r in base] == list(range(1, 21))
        for r in base:
            held = r["n_good"] + r["n_bad"] < HOLD
            trained = label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED
            assert r["detector_tier"] == (TIER_TRAINED if trained and not held else TIER_QUERY), r["t"]
        held_past_quota = [
            r
            for r in base
            if r["detector_tier"] == TIER_QUERY and label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED
        ]
        assert held_past_quota, "the run meets the quota before the hold"
        assert all(r["app_trained"] == 0 for r in held_past_quota)

    def test_a_held_clicks_head_is_the_tagged_row_and_the_rest_are_the_hand_overs(self, held_run, typed_run):
        from vtscore.eval.voting_iterations import QUERY_HOLD_HEAD_TAG

        rows, _picks = held_run
        old_by_t = {r["t"]: r for r in _base(typed_run[0])}
        tagged = {r["t"]: r for r in rows if r.get("gmm_variant") == QUERY_HOLD_HEAD_TAG}
        assert tagged and max(tagged) < HOLD
        base_by_t = {r["t"]: r for r in _base(rows)}
        for t, old in old_by_t.items():
            new = tagged.get(t, base_by_t[t]) if old["detector_tier"] == TIER_TRAINED else base_by_t[t]
            assert new["detector_tier"] == old["detector_tier"], t
            for col in ("precision", "recall", "fbeta", "n_flagged"):
                assert new[col] == old[col] or (math.isnan(new[col]) and math.isnan(old[col])), (t, col)

    def test_the_held_rows_are_the_query_rows_set(self, held_run):
        rows, _picks = held_run
        query = [r for r in _base(rows) if r["detector_tier"] == TIER_QUERY]
        for col in ("precision", "recall", "fbeta", "n_flagged"):
            assert len({r[col] for r in query}) == 1, col


class TestOtherOpeningsAreUnchanged:
    def test_an_example_opening_gives_the_centroid(self):
        clips, _seed_scores = _world()
        rows = simulate_voting_iterations(
            clips, "alpha", seed=0, max_steps=12, calibrate_count=2, spot_check="off", seed_examples=1
        )
        assert TIER_QUERY not in {r["detector_tier"] for r in rows}

    def test_without_the_quota_there_is_no_query_tier(self):
        rows, _picks = _run(label_quota=False)
        assert {r["detector_tier"] for r in rows} == {TIER_TRAINED}


class TestTheCalibrationPath:
    def test_a_query_row_says_so_in_its_provenance(self):
        rows, _picks = _run(style="whole_image", emit_calibration_metrics=True)
        query = [r for r in rows if r["detector_tier"] == TIER_QUERY]
        assert query
        assert {r["threshold_provenance"] for r in query} == {"query"}
        assert all(r["threshold"] == 0.5 for r in query)


class TestTheQueryTestIsThePlantedAnswer:
    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_it_keeps_the_text_sorts_set_on_the_withheld_half(self, beta):
        """The typed query's cosines on the withheld half, cut where ``text_sort_cuts`` draws there."""
        from vtscore.eval.labels import media_is_positive
        from vtscore.eval.voting_iterations import _query_test
        from vtscore.training.thresholds import text_sort_cuts

        clips, seed_scores = _world()
        test_ids = list(range(2, 81, 2))
        _step, rows = _query_test(
            seed_scores, clips, test_ids, "alpha", 0, beta=beta, calibration_rows=False, line_beta=beta
        )
        cos = np.asarray([seed_scores[c] for c in test_ids])
        line = text_sort_cuts(list(cos), beta=beta).threshold
        kept = cos >= line - 1e-9
        pos = np.asarray([media_is_positive(clips[c], "alpha") for c in test_ids], dtype=bool)
        assert rows[0]["n_flagged"] == int(kept.sum())
        assert rows[0]["recall"] == pytest.approx(float((kept & pos).sum() / pos.sum()))


#: #4800's variants: the centroid at the app's rule over two corpora, and the typed query's set, at two balances.
VARIANTS = ["count_gap_floor_skip25_b1@1", "count_gap_floor_skip25_b1~unvoted@4", "query@1", "query@0.25"]


@pytest.fixture(scope="module")
def priced_run():
    return _run(query_hold=0, centroid_line_variants=VARIANTS)


def _tagged(rows, spec):
    from vtscore.eval.voting_iterations import centroid_line_tag

    rule, _, beta = spec.partition("@")
    return {r["t"]: r for r in rows if r.get("gmm_variant") == centroid_line_tag(rule, float(beta))}


class TestTheCentroidIsPricedUnderTheQuery:
    """#4800: a typed-query run prices the Goods' centroid and the query's set at other balances as tagged rows."""

    def test_the_variants_parse_and_the_query_takes_no_corpus(self):
        from vtscore.eval.voting_iterations import QUERY_LINE_VARIANT, parse_centroid_line_variants

        assert parse_centroid_line_variants("query@0.25,count@1") == [(QUERY_LINE_VARIANT, 0.25), ("count", 1.0)]
        with pytest.raises(ValueError, match="query@"):
            parse_centroid_line_variants("query~unvoted@1")

    def test_the_session_and_its_untagged_rows_are_the_unpriced_runs(self, priced_run, typed_run):
        rows, picks = priced_run
        old, old_picks = typed_run
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in old_picks]
        base = [r for r in rows if not str(r.get("gmm_variant") or "").startswith("centroid_line:")]
        assert len(base) == len(old)
        for a, b in zip(base, old, strict=True):
            assert (a["t"], a["detector_tier"], a["fbeta"], a["n_flagged"]) == (
                b["t"],
                b["detector_tier"],
                b["fbeta"],
                b["n_flagged"],
            )

    def test_the_centroid_rows_ride_the_query_clicks_that_hold_a_good(self, priced_run):
        rows, _picks = priced_run
        query_clicks = {r["t"]: r for r in _base(rows) if r["detector_tier"] == TIER_QUERY}
        with_good = {t for t, r in query_clicks.items() if r["n_good"] >= 1}
        assert with_good
        for spec in VARIANTS[:2]:
            tagged = _tagged(rows, spec)
            assert set(tagged) == with_good, spec
            assert {r["detector_tier"] for r in tagged.values()} == {TIER_CENTROID}
        for spec in VARIANTS[2:]:
            tagged = _tagged(rows, spec)
            assert set(tagged) == set(query_clicks), spec
            assert {r["detector_tier"] for r in tagged.values()} == {TIER_QUERY}
        assert {r["beta"] for r in _tagged(rows, "query@0.25").values()} == {0.25}

    def test_a_click_before_the_first_good_prices_the_query_alone(self):
        from vtscore.eval.voting_iterations import _centroid_line_variant_rows, parse_centroid_line_variants

        clips, seed_scores = _world()
        kw = {"region_voting": False, "region_aware": False, "style_obj": None, "voted": {5}}
        args = (parse_centroid_line_variants(VARIANTS), {"t": 1}, {}, clips, list(range(2, 81, 2)), "alpha", 0)
        rows = _centroid_line_variant_rows(*args, **kw, query_scores=seed_scores)
        assert [r["gmm_variant"] for r in rows] == ["centroid_line:query@1", "centroid_line:query@0.25"]
        assert _centroid_line_variant_rows(*args, **kw) == []

    def test_the_query_at_the_runs_balance_is_the_untagged_row(self, priced_run):
        rows, _picks = priced_run
        base = {r["t"]: r for r in _base(rows) if r["detector_tier"] == TIER_QUERY}
        tagged = _tagged(rows, "query@1")
        for t, r in base.items():
            for col in ("precision", "recall", "fbeta", "n_flagged"):
                assert tagged[t][col] == r[col] or (math.isnan(tagged[t][col]) and math.isnan(r[col])), (t, col)

    def test_the_centroid_at_the_apps_rule_is_what_the_arm_before_gives(self, priced_run, before_run):
        """The arm before (``query_tier=False``) gives the centroid on these clicks: its rows are the tagged ones."""
        rows, _picks = priced_run
        old = {r["t"]: r for r in _base(before_run[0]) if r["detector_tier"] == TIER_CENTROID}
        tagged = _tagged(rows, "count_gap_floor_skip25_b1@1")
        assert set(tagged) <= set(old)
        assert tagged
        for t, r in tagged.items():
            for col in ("precision", "recall", "fbeta", "n_flagged"):
                assert r[col] == pytest.approx(old[t][col], nan_ok=True), (t, col)


class TestAPricedCentroidIsTheCentroidTest:
    """#4800's one probe pass a click gives each variant the set :func:`_centroid_test` gives at that line."""

    SPECS = ["count_gap_floor_skip25_b1@1", "count_gap_floor_skip25_b1~unvoted@4", "midpoint@0.25", "count~pool@1"]

    @staticmethod
    def _worlds():
        from tests_lib.eval.test_find_corpus import _patch_medias
        from vtscore.eval.patch_styles import resolve_style

        clips, _seed_scores = _world()
        yield "single", clips, {"region_voting": False, "region_aware": False, "style_obj": None}
        patches = _patch_medias(16, 64, seed=2)
        yield "patch", patches, {"region_voting": True, "region_aware": True, "style_obj": resolve_style("max_patch")}

    @pytest.mark.parametrize("spec", SPECS)
    def test_each_variant_matches(self, spec, monkeypatch):
        import vtscore.eval.voting_iterations as vi

        monkeypatch.setattr(vi, "PROBE_CHUNK", 7)  # chunks that split the corpus mid-way
        for name, clips, geo in self._worlds():
            ids = sorted(clips)
            test_ids, goods, voted = ids[1::2], {ids[0]: None, ids[2]: None}, {ids[0], ids[2], ids[4]}
            (got,) = vi._centroid_line_variant_rows(
                vi.parse_centroid_line_variants(spec), {}, goods, clips, test_ids, "alpha", 0, voted=voted, **geo
            )
            rule, _, beta = spec.partition("@")
            rule, _, corpus = rule.partition("~")
            line_ids = {"pool": ids, "unvoted": [i for i in ids if i not in voted]}.get(corpus)
            _step, (want,), _cal = vi._centroid_test(
                goods, clips, test_ids, "alpha", 0, **geo, beta=float(beta), calibration_rows=False,
                line_rule=rule, line_beta=float(beta), line_ids=line_ids,
            )  # fmt: skip
            for col in ("n_flagged", "precision", "recall", "fbeta", "fpr", "fnr", "cost"):
                assert got[col] == want[col] or (math.isnan(got[col]) and math.isnan(want[col])), (name, col)
            for col in ("auroc", "average_precision"):
                assert got[col] == pytest.approx(want[col], abs=1e-9, nan_ok=True), (name, col)


def _missing_world():
    """A patch-grid cell whose typed query ranks two Goods first, then a run of Bads, then the other Goods."""
    from tests_lib.eval.test_find_corpus import _patch_medias

    clips = _patch_medias(16, 64, seed=5)
    pos = [c for c, m in clips.items() if m["category"] == "alpha"]
    neg = [c for c, m in clips.items() if m["category"] != "alpha"]
    order = [*pos[:2], *neg[:12], *pos[2:], *neg[12:]]
    seed_scores = {cid: 0.9 - 0.005 * i for i, cid in enumerate(order)}
    return clips, seed_scores


def _missing_run(**kwargs):
    clips, seed_scores = _missing_world()
    picks: list[dict] = []
    rows = simulate_voting_iterations(
        clips, "alpha", seed=1, max_steps=14, calibrate_count=2, spot_check="off", region_voting=True,
        pick_sink=picks, seed_scores=seed_scores, **kwargs,
    )  # fmt: skip
    return rows, picks


@pytest.fixture(scope="module")
def runs():
    """The app, the arm before #4800 (the query until the quota) and the arm before #4758 (the centroid)."""
    return _missing_run(), _missing_run(query_miss=False), _missing_run(query_tier=False)


class TestTheQueryMiss:
    """#4800: on a patch space, once the typed query has missed, Test gives the Goods' centroid until the quota."""

    def test_the_missed_votes_are_the_centroids_and_the_rest_the_querys(self, runs):
        from vtscore.detectors.label_quota import query_missed

        (rows, _picks), _before, _old = runs
        base = _base(rows)
        missed = [r for r in base if query_missed(r["n_good"], r["n_bad"], patch_space=True)]
        assert missed, "the world's query misses"
        assert {r["detector_tier"] for r in missed} == {TIER_CENTROID}
        under = [r for r in base if label_quota(r["n_good"], r["n_bad"]).tier != TIER_TRAINED and r not in missed]
        assert under and {r["detector_tier"] for r in under} == {TIER_QUERY}

    def test_the_arm_before_keeps_the_query_and_the_session(self, runs):
        (rows, picks), (before, before_picks), _old = runs
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in before_picks]
        assert TIER_CENTROID not in {r["detector_tier"] for r in _base(before)}

    def test_the_centroid_rows_are_the_centroids_own(self, runs):
        """The centroid there is the one the arm before #4758 gives on the same click."""
        (rows, _picks), _before, (old, _old_picks) = runs
        old_by_t = {r["t"]: r for r in _base(old)}
        for r in (r for r in _base(rows) if r["detector_tier"] == TIER_CENTROID):
            for col in ("precision", "recall", "fbeta", "n_flagged"):
                o = old_by_t[r["t"]][col]
                assert r[col] == o or (math.isnan(r[col]) and math.isnan(o)), (r["t"], col)


class TestAnOpeningThatIsNotACosine:
    """#5241: SAM3's detection scores open a run; its set is cut at a fixed score or the GMM midpoint, and can hold
    on a patch space through the run while the head trains, tagged, as on a single-vector space."""

    @pytest.mark.parametrize("line", ["fixed:0.5", "fixed:0.8", "gmm"])
    def test_the_line_keeps_the_scores_at_or_above_its_cut(self, line):
        from vtscore.eval.labels import media_is_positive
        from vtscore.eval.voting_iterations import _query_test
        from vtscore.training.thresholds import calculate_gmm_threshold

        clips, cosines = _world()
        scores = {c: 1.0 / (1.0 + math.exp(-12.0 * v)) for c, v in cosines.items()}  # a detector's [0, 1] scores
        test_ids = list(range(2, 81, 2))
        _step, rows = _query_test(scores, clips, test_ids, "alpha", 0, beta=1.0, calibration_rows=False, line=line)
        s = np.asarray([scores[c] for c in test_ids])
        cut = float(calculate_gmm_threshold(list(s))) if line == "gmm" else float(line.partition(":")[2])
        kept = s >= cut
        pos = np.asarray([media_is_positive(clips[c], "alpha") for c in test_ids], dtype=bool)
        assert rows[0]["n_flagged"] == int(kept.sum())
        assert rows[0]["recall"] == pytest.approx(float((kept & pos).sum() / pos.sum()))

    def test_an_unknown_line_is_refused(self):
        from vtscore.eval.voting_iterations import _query_test

        clips, scores = _world()
        with pytest.raises(ValueError, match="unknown query line"):
            _query_test(scores, clips, [2, 4, 6], "alpha", 0, beta=1.0, calibration_rows=False, line="median")

    def test_a_patch_space_holds_only_when_asked(self):
        rows, picks = _missing_run(query_miss=False, query_hold=100)
        held, held_picks = _missing_run(query_miss=False, query_hold=100, query_hold_patch=True)
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in held_picks]
        assert TIER_TRAINED in {r["detector_tier"] for r in _base(rows)}
        assert {r["detector_tier"] for r in _base(held)} == {TIER_QUERY}
        from vtscore.eval.voting_iterations import QUERY_HOLD_HEAD_TAG

        heads = [r for r in held if r.get("gmm_variant") == QUERY_HOLD_HEAD_TAG]
        trained = {r["t"]: r for r in _base(rows) if r["detector_tier"] == TIER_TRAINED}
        assert heads and {r["t"] for r in heads} == set(trained)
        for r in heads:
            assert r["fbeta"] == trained[r["t"]]["fbeta"] or (
                math.isnan(r["fbeta"]) and math.isnan(trained[r["t"]]["fbeta"])
            )
