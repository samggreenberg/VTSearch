"""The Goods' centroid's line (#4732): the rules it can be drawn by, and the harness rows that price them.

Below the label quota, Test gives the Goods' centroid, cut at the two-Gaussian midpoint of its corpus
cosines. On a rare target that midpoint splits the negatives' own bulk: on one FHIBE face cell the
centroid ranked all four withheld photos first of 5,439 and its line still kept about 2,500. These
tests pin the rules :func:`~vtscore.detectors.centroid_head.centroid_cut` can draw and the harness's
tagged rows that price each of them on the same session.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.detectors.centroid_head import CENTROID_COUNTS, CENTROID_LINE_RULE, CENTROID_LINE_RULES, centroid_cut
from vtscore.eval.voting_iterations import centroid_line_tag, parse_centroid_line_variants
from vtscore.training.thresholds import (
    _count_line,
    calculate_gmm_threshold,
    guarded_text_sort_threshold,
    text_sort_cuts,
)


def _rare_target(n_neg: int = 5000, n_pos: int = 4, seed: int = 0) -> np.ndarray:
    """Cosines of a rare target: one broad bulk of negatives and a few far matches."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.1, 0.08, n_neg), rng.normal(0.8, 0.03, n_pos)])


class TestTheRules:
    def test_the_app_draws_the_gated_standout_drop(self):
        assert CENTROID_LINE_RULE == "count_gap_floor_skip25_b1"
        for cos in (_rare_target(), _light_tailed_person(), _standout_person()):
            for beta in (0.25, 1.0, 4.0):
                assert centroid_cut(cos, beta=beta) == centroid_cut(cos, rule="count_gap_floor_skip25_b1", beta=beta)

    def test_each_rule_is_the_typed_querys_line_it_names(self):
        cos = _rare_target()
        scores = cos.tolist()
        assert centroid_cut(cos, rule="midpoint", beta=1.0) == pytest.approx(calculate_gmm_threshold(scores))
        assert centroid_cut(cos, rule="guarded", beta=0.25) == pytest.approx(guarded_text_sort_threshold(scores)[0])
        for beta in (0.25, 1.0, 4.0):
            assert centroid_cut(cos, rule="text", beta=beta) == pytest.approx(
                text_sort_cuts(scores, beta=beta).threshold
            )
            assert centroid_cut(cos, rule="count", beta=beta) == pytest.approx(_count_line(cos, beta))

    def test_on_a_rare_target_the_midpoint_keeps_the_bulk_and_the_count_keeps_the_matches(self):
        """The #4732 failure in miniature: the midpoint keeps thousands, the count line about the four."""
        cos = _rare_target()
        kept = {rule: int((cos >= centroid_cut(cos, rule=rule, beta=1.0)).sum()) for rule in CENTROID_LINE_RULES}
        assert kept["midpoint"] > 500
        assert 1 <= kept["count"] <= 12
        assert kept["text"] == kept["count"]

    def test_the_count_keeps_more_at_a_recall_balance(self):
        cos = _rare_target(n_pos=40)
        kept = [int((cos >= centroid_cut(cos, rule="count", beta=b)).sum()) for b in (0.25, 1.0, 4.0)]
        assert kept[0] < kept[1] < kept[2]

    def test_without_a_balance_the_balanced_rules_keep_the_guarded_line(self):
        cos = _rare_target()
        guarded = guarded_text_sort_threshold(cos.tolist())[0]
        assert centroid_cut(cos, rule="text") == pytest.approx(guarded)
        assert centroid_cut(cos, rule="count") == pytest.approx(guarded)

    def test_too_few_scores_for_a_count_keep_the_guarded_line(self):
        cos = _rare_target(n_neg=30, n_pos=2)
        assert centroid_cut(cos, rule="count", beta=1.0) == pytest.approx(guarded_text_sort_threshold(cos.tolist())[0])

    def test_refuses_an_unknown_rule(self):
        with pytest.raises(ValueError, match="centroid line rule"):
            centroid_cut(_rare_target(), rule="tail")


def _light_tailed_person(n_neg: int = 5000, n_pos: int = 3, seed: int = 0) -> np.ndarray:
    """A face centroid's cosines (#4759): a bounded bulk, and a few matches just past its edge.

    On FHIBE's FaceNet crops a person's withheld photos sit 3 to 4 bulk sigmas above the
    median, where a Gaussian bulk would put several images but this one puts almost none.
    """
    rng = np.random.default_rng(seed)
    bulk = rng.uniform(-0.3, 0.3, n_neg)  # sigma = 1.4826 x MAD = 0.22; nothing past z = 1.4
    return np.concatenate([bulk, rng.uniform(0.75, 0.85, n_pos)])


class TestTheCandidateCounts:
    """#4759's counts: what each keeps, and the shipped count they are priced against."""

    def test_the_count_line_is_its_estimate_kept_at_the_balance(self):
        from vtscore.training.thresholds import _count_estimate, _count_keep_line

        cos = _rare_target(n_pos=40)
        gauss, mirror = _count_estimate(cos), _count_estimate(cos, z=3.5, bulk="mirror")
        assert gauss is not None and mirror is not None
        for beta in (0.25, 1.0, 4.0):
            assert _count_line(cos, beta) == _count_keep_line(cos, beta, gauss)
            assert _count_line(cos, beta, z=3.5, bulk="mirror") == _count_keep_line(cos, beta, mirror)
        with pytest.raises(ValueError, match="bulk"):
            _count_estimate(cos, bulk="pareto")

    def test_a_gaussian_count_misses_matches_just_past_a_light_bulk(self):
        """The #4759 failure in miniature: 3 matches at z ~ 3.6, and a lower bar only subtracts more."""
        cos = _light_tailed_person()
        kept = {r: int((cos >= centroid_cut(cos, rule=r, beta=1.0)).sum()) for r in CENTROID_COUNTS}
        assert kept["count"] == 1
        assert kept["count_z3"] == 1
        assert kept["count_z3.5"] < 3
        assert kept["gap"] == kept["count_gap"] == kept["gap_floor"] == kept["count_gap_floor"] == 3

    def test_the_gap_floor_keeps_the_gap_at_every_balance(self):
        """A gap is a line: the floor keeps the 3 matches at beta 1/4 and 4 too, where a count is scaled."""
        cos = _light_tailed_person()
        kept = {
            (r, b): int((cos >= centroid_cut(cos, rule=r, beta=b)).sum())
            for r in ("gap", "gap_floor")
            for b in (0.25, 4.0)
        }
        assert kept[("gap_floor", 0.25)] == 3 and kept[("gap", 0.25)] == 1
        assert kept[("gap_floor", 4.0)] == 3 and kept[("gap", 4.0)] == 8

    @pytest.mark.parametrize("seed", range(8))
    def test_count_gap_floor_is_gap_floor_below_beta_1_and_count_gap_above(self, seed):
        """The identity #4759 reads count_gap_floor's rows by: the tagged runs never drew it."""
        rng = np.random.default_rng(seed)
        n_pos = int(rng.integers(1, 60))
        cos = _rare_target(n_pos=n_pos, seed=seed) if seed % 2 else _light_tailed_person(n_pos=min(n_pos, 8), seed=seed)
        for beta in (0.25, 0.5, 1.0):
            assert centroid_cut(cos, rule="count_gap_floor", beta=beta) == centroid_cut(
                cos, rule="gap_floor", beta=beta
            )
        for beta in (1.0, 2.0, 4.0):
            assert centroid_cut(cos, rule="count_gap_floor", beta=beta) == centroid_cut(
                cos, rule="count_gap", beta=beta
            )

    @pytest.mark.parametrize("rule", ["count_gap", "gap_floor", "count_gap_floor"])
    def test_the_gap_hybrids_never_keep_less_than_count(self, rule):
        for seed in range(5):
            cos = _rare_target(n_pos=40, seed=seed)
            for beta in (0.25, 1.0, 4.0):
                assert (cos >= centroid_cut(cos, rule=rule, beta=beta)).sum() >= (
                    cos >= centroid_cut(cos, rule="count", beta=beta)
                ).sum()

    @pytest.mark.parametrize("rule", CENTROID_COUNTS)
    def test_every_count_keeps_the_guarded_line_without_a_balance_or_scores(self, rule):
        cos = _rare_target()
        assert centroid_cut(cos, rule=rule) == pytest.approx(guarded_text_sort_threshold(cos.tolist())[0])
        few = _rare_target(n_neg=30, n_pos=2)
        assert centroid_cut(few, rule=rule, beta=1.0) == pytest.approx(guarded_text_sort_threshold(few.tolist())[0])

    def test_the_variant_spec_takes_the_candidates(self):
        assert parse_centroid_line_variants("count_gap@1,mirror_z3.5@4") == [("count_gap", 1.0), ("mirror_z3.5", 4.0)]


def _standout_person(n_neg: int = 5000, seed: int = 0) -> np.ndarray:
    """A person whose best photo stands far above their other three, all above the crowd (#4783)."""
    rng = np.random.default_rng(seed)
    bulk = rng.uniform(-0.3, 0.70, n_neg)
    return np.concatenate([bulk, [0.95, 0.80, 0.79, 0.78]])


class TestTheStandoutDrops:
    """#4783: the widest drop stops after a standout first photo; the candidates look past it."""

    def test_the_widest_drop_keeps_only_the_standout(self):
        from vtscore.detectors.centroid_head import _top_gap_count

        assert _top_gap_count(_standout_person()) == 1

    @pytest.mark.parametrize("drop", ["skip25", "skip50", "rank5"])
    def test_each_candidate_keeps_the_whole_person(self, drop):
        from vtscore.detectors.centroid_head import _top_gap_count

        assert _top_gap_count(_standout_person(), drop=drop) == 4

    @pytest.mark.parametrize("rule", ["count_gap_floor_skip25", "count_gap_floor_skip50", "count_gap_floor_rank5"])
    def test_the_rules_keep_the_person_at_every_balance(self, rule):
        cos = _standout_person()
        for beta in (0.25, 1.0, 4.0):
            assert int((cos >= centroid_cut(cos, rule=rule, beta=beta)).sum()) >= 4
        assert int((cos >= centroid_cut(cos, rule="count_gap_floor", beta=0.25)).sum()) == 1

    def test_a_skip_keeps_the_widest_drop_when_it_is_not_the_first(self):
        from vtscore.detectors.centroid_head import _top_gap_count

        cos = _light_tailed_person()
        assert _top_gap_count(cos, drop="skip25") == _top_gap_count(cos) == 3

    @pytest.mark.parametrize("seed", range(8))
    def test_the_gated_rule_is_the_floor_below_beta_1_and_the_skip_above(self, seed):
        """The identity #4783 reads its shipped rule's rows by: the runs carried the two halves."""
        rng = np.random.default_rng(seed)
        cos = _rare_target(n_pos=int(rng.integers(1, 60)), seed=seed) if seed % 2 else _standout_person(seed=seed)
        for beta in (0.25, 0.5, 0.99):
            assert centroid_cut(cos, rule="count_gap_floor_skip25_b1", beta=beta) == centroid_cut(
                cos, rule="count_gap_floor", beta=beta
            )
        for beta in (1.0, 2.0, 4.0):
            assert centroid_cut(cos, rule="count_gap_floor_skip25_b1", beta=beta) == centroid_cut(
                cos, rule="count_gap_floor_skip25", beta=beta
            )

    def test_the_gated_rule_keeps_the_standout_alone_at_a_precision_balance(self):
        cos = _standout_person()
        kept = {
            b: int((cos >= centroid_cut(cos, rule="count_gap_floor_skip25_b1", beta=b)).sum()) for b in (0.25, 1.0, 4.0)
        }
        assert kept[0.25] == 1
        assert kept[1.0] >= 4 and kept[4.0] >= 4

    def test_refuses_an_unknown_drop(self):
        from vtscore.detectors.centroid_head import _top_gap_count

        with pytest.raises(ValueError, match="drop"):
            _top_gap_count(_rare_target(), drop="steepest")


class TestTheHeadKeepsWhatTheLineKeeps:
    """The count line sits on a media's cosine; the float32 head must still keep that media."""

    def test_the_gap_centre_keeps_the_same_media(self):
        from vtscore.detectors.centroid_head import _gap_centre

        cos = _rare_target()
        for rule in CENTROID_LINE_RULES:
            line = centroid_cut(cos, rule=rule, beta=1.0)
            cut = _gap_centre(cos, line)
            assert np.array_equal(cos >= line, cos >= cut)
            assert not np.any(cos == cut)
        assert _gap_centre(cos, 99.0) == 99.0
        assert _gap_centre(cos, -99.0) == -99.0

    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_a_count_head_keeps_the_count(self, beta):
        from vtscore.detectors.centroid_head import fit_centroid_head, goods_centroid
        from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap

        rng = np.random.default_rng(1)
        dim, emb = 32, "test_embedder"
        centre = rng.standard_normal(dim)
        vecs = np.vstack([rng.standard_normal((3000, dim)), centre * 4 + rng.standard_normal((5, dim)) * 0.5])
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
        snap = {
            i: {"id": i, "media_type": "image", "embedder": emb, "embeddings": {emb: v}} for i, v in enumerate(vecs)
        }
        rows = scoring_rows_for_snap(snap, emb)
        goods = [vecs[-1]]
        cosines = rows.matrix.astype(np.float64) @ goods_centroid(goods).astype(np.float64)
        want = int((cosines >= centroid_cut(cosines, rule="count", beta=beta)).sum())
        head, threshold = fit_centroid_head(goods, lambda h: score_rows_with_model(h, rows)[0], rule="count", beta=beta)
        got = int((np.asarray(score_rows_with_model(head, rows)[0]) >= threshold).sum())
        assert got == want >= 1


class TestVariantSpec:
    def test_parses_rule_at_beta(self):
        assert parse_centroid_line_variants("count@0.25, text@1,midpoint@4") == [
            ("count", 0.25),
            ("text", 1.0),
            ("midpoint", 4.0),
        ]
        assert parse_centroid_line_variants(["guarded@1"]) == [("guarded", 1.0)]
        assert parse_centroid_line_variants(None) == parse_centroid_line_variants("") == []

    @pytest.mark.parametrize("bad", ["count", "count@", "count@0", "count@-1", "tail@1", "count@x", "count@1,count@1"])
    def test_refuses_a_malformed_variant(self, bad):
        with pytest.raises(ValueError):
            parse_centroid_line_variants(bad)

    def test_the_tag(self):
        assert centroid_line_tag("count", 0.25) == "centroid_line:count@0.25"
        assert centroid_line_tag("text", 1.0) == "centroid_line:text@1"


class TestInTheHarness:
    """A few-positive target: Test gives the centroid for the opening's first clicks."""

    VARIANTS = ["midpoint@1", "count@1", "count@0.25"]

    def _run(self, **kw):
        from tests_lib.eval.test_example_opening import _medias
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _medias(n_pos=6, n_neg=400),
            target_category="person",
            seed=3,
            dataset_name="stub",
            max_steps=12,
            atlas_min_node_size=8,
            spot_check="off",
            seed_examples=1,
            stratify_target=True,
            beta=1.0,
            pick_sink=picks,
            **kw,
        )
        return rows, picks

    @staticmethod
    def _centroid_rows(rows: list[dict], tag: str = "") -> dict[int, dict]:
        return {r["t"]: r for r in rows if r.get("detector_tier") == "centroid" and (r.get("gmm_variant") or "") == tag}

    def test_variants_are_tagged_rows_on_the_same_session(self):
        base_rows, base_picks = self._run()
        rows, picks = self._run(centroid_line_variants=self.VARIANTS)
        assert [p["picked_id"] for p in picks] == [p["picked_id"] for p in base_picks]
        untagged = [r for r in rows if not r.get("gmm_variant", "").startswith("centroid_line:")]
        assert len(untagged) == len(base_rows)
        for a, b in zip(untagged, base_rows, strict=True):
            assert (a["t"], a.get("fbeta"), a.get("n_above")) == (b["t"], b.get("fbeta"), b.get("n_above"))
        clicks = self._centroid_rows(rows)
        assert clicks, "the opening should give the centroid on some click"
        for spec in self.VARIANTS:
            rule, _, beta = spec.partition("@")
            tagged = self._centroid_rows(rows, centroid_line_tag(rule, float(beta)))
            assert set(tagged) == set(clicks)
            assert all(r["beta"] == float(beta) for r in tagged.values())

    def test_the_apps_rule_at_the_runs_balance_is_the_apps_row(self):
        rows, _ = self._run(centroid_line_variants=[f"{CENTROID_LINE_RULE}@1"])
        base = self._centroid_rows(rows)
        tagged = self._centroid_rows(rows, centroid_line_tag(CENTROID_LINE_RULE, 1.0))
        for t, r in base.items():
            assert tagged[t]["fbeta"] == pytest.approx(r["fbeta"])
            assert tagged[t]["precision"] == pytest.approx(r["precision"])

    def test_the_main_rule_moves_only_the_centroid_rows(self):
        base_rows, base_picks = self._run()
        rows, picks = self._run(centroid_line="count")
        assert [p["picked_id"] for p in picks] == [p["picked_id"] for p in base_picks]
        count_rows = self._run(centroid_line_variants=["count@1"])[0]
        want = self._centroid_rows(count_rows, centroid_line_tag("count", 1.0))
        got = self._centroid_rows(rows)
        assert set(got) == set(want)
        for t, r in got.items():
            assert r["fbeta"] == pytest.approx(want[t]["fbeta"])
        trained = [(r["t"], r.get("fbeta")) for r in rows if r.get("detector_tier") != "centroid"]
        assert trained == [(r["t"], r.get("fbeta")) for r in base_rows if r.get("detector_tier") != "centroid"]

    def test_corpus_variants_are_tagged_rows_on_the_centroid_clicks(self):
        """#4779: the same centroid's line over the whole pool, voted media in or out, scored on the withheld half."""
        specs = ["count_gap_floor~pool@1", "count_gap_floor~unvoted@1", "count_gap_floor@1"]
        rows, _ = self._run(centroid_line_variants=specs)
        clicks = self._centroid_rows(rows)
        for spec in specs:
            rule, _, beta = spec.partition("@")
            tagged = self._centroid_rows(rows, centroid_line_tag(rule, float(beta)))
            assert set(tagged) == set(clicks)
            assert all(r["n_test_pos"] == clicks[t]["n_test_pos"] for t, r in tagged.items())

    def test_refuses_an_unknown_rule(self):
        with pytest.raises(ValueError, match="centroid_line"):
            self._run(centroid_line="tail")


class TestTheAppRedrawsTheLine:
    """A balance change redraws the centroid's line on the cosines it was drawn on; acquisition stays at the midpoint."""

    @staticmethod
    def _fit(rule: str, beta: float):
        from vtscore.detectors.centroid_head import fit_centroid
        from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap

        rng = np.random.default_rng(2)
        dim, emb = 32, "test_embedder"
        centre = rng.standard_normal(dim)
        vecs = np.vstack([rng.standard_normal((3000, dim)), centre * 4 + rng.standard_normal((40, dim)) * 0.5])
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
        snap = {
            i: {"id": i, "media_type": "image", "embedder": emb, "embeddings": {emb: v}} for i, v in enumerate(vecs)
        }
        rows = scoring_rows_for_snap(snap, emb)
        goods = [vecs[-1], vecs[-2]]
        head, threshold, line = fit_centroid(goods, lambda h: score_rows_with_model(h, rows)[0], rule=rule, beta=beta)
        scores = np.asarray(score_rows_with_model(head, rows)[0])
        return head, threshold, line, scores

    @pytest.mark.parametrize("rule", ["count", "text", "guarded", "midpoint"])
    def test_the_line_at_its_own_balance_is_the_heads_threshold(self, rule):
        _head, threshold, line, scores = self._fit(rule, 1.0)
        assert line.rule == rule
        assert line.threshold_at(1.0) == pytest.approx(threshold, abs=1e-9)
        assert int((scores >= line.threshold_at(1.0)).sum()) == int((scores >= threshold).sum())

    @pytest.mark.parametrize("beta", [0.25, 4.0])
    def test_another_balance_keeps_what_that_line_keeps(self, beta):
        _head, _threshold, line, scores = self._fit("count", 1.0)
        want = int((line.cosines >= centroid_cut(line.cosines, rule="count", beta=beta)).sum())
        assert int((scores >= line.threshold_at(beta)).sum()) == want

    def test_acquisition_keeps_what_the_midpoint_keeps(self):
        _head, _threshold, line, scores = self._fit("count", 1.0)
        want = int((line.cosines >= calculate_gmm_threshold(line.cosines.tolist())).sum())
        assert int((scores >= line.acquisition_threshold()).sum()) == want

    def test_the_context_recuts_and_samples_through_the_line(self):
        from vtscore.state.core import DetectorContext, detector_acquisition_threshold, recut_detector_threshold

        head, threshold, line, _scores = self._fit("count", 1.0)
        ctx = DetectorContext("", name="centroid", media_type="image")
        ctx.model, ctx.threshold, ctx.centroid_line = head, threshold, line
        assert recut_detector_threshold(ctx, beta=4.0) == pytest.approx(line.threshold_at(4.0))
        assert detector_acquisition_threshold(ctx, beta=1.0) == pytest.approx(line.acquisition_threshold())
        # A trained head that replaced the centroid never reads a stale line.
        from vtscore.training.mlp import LINEAR_SVM_HEAD, build_model

        ctx.model = build_model(32, hidden_dim=LINEAR_SVM_HEAD)
        assert recut_detector_threshold(ctx, beta=4.0) != pytest.approx(line.threshold_at(4.0))


class TestTheLineCountsTheUnvoted:
    """#4779: the Goods a centroid is built from sit at the top of their own ranking; its line leaves them out."""

    @staticmethod
    def _rows():
        from vtscore.detectors.training import scoring_rows_for_snap

        rng = np.random.default_rng(5)
        dim, emb = 32, "test_embedder"
        centre = rng.standard_normal(dim)
        vecs = np.vstack([rng.standard_normal((3000, dim)), centre * 4 + rng.standard_normal((4, dim)) * 0.5])
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
        snap = {
            i: {"id": i, "media_type": "image", "embedder": emb, "embeddings": {emb: v}} for i, v in enumerate(vecs)
        }
        return vecs, scoring_rows_for_snap(snap, emb)

    def test_the_mask_leaves_the_flagged_media_out_of_the_line(self):
        from vtscore.detectors.centroid_head import fit_centroid
        from vtscore.detectors.training import score_rows_with_model

        vecs, rows = self._rows()
        good = len(vecs) - 1  # the example itself: cosine 1 to its own centroid
        score = lambda h: score_rows_with_model(h, rows)[0]  # noqa: E731
        _h, _t, all_line = fit_centroid([vecs[good]], score, beta=1.0)
        mask = [rid != good for rid in rows.ids]
        _h, _t, unvoted = fit_centroid([vecs[good]], score, beta=1.0, line_mask=mask)
        assert unvoted.cosines.size == all_line.cosines.size - 1
        assert all_line.cosines.max() > unvoted.cosines.max()

    def test_a_mask_of_the_wrong_length_is_refused(self):
        from vtscore.detectors.centroid_head import fit_centroid
        from vtscore.detectors.training import score_rows_with_model

        vecs, rows = self._rows()
        with pytest.raises(ValueError, match="line_mask"):
            fit_centroid([vecs[-1]], lambda h: score_rows_with_model(h, rows)[0], beta=1.0, line_mask=[True])

    def test_the_unvoted_mask_flags_all_but_the_labelled(self):
        from vtscore.detectors.labelset_training import _unvoted_mask

        _vecs, rows = self._rows()
        mask = _unvoted_mask(rows, {0, 3003})
        assert mask is not None
        assert len(mask) == len(rows.ids) and sum(not m for m in mask) == 2
        assert _unvoted_mask(None, {1}) is None

    def test_the_corpus_variants_parse_and_tag(self):
        from vtscore.eval.voting_iterations import CENTROID_LINE_CORPORA

        assert "unvoted" in CENTROID_LINE_CORPORA
        got = parse_centroid_line_variants("count_gap_floor~pool@1,count_gap_floor~unvoted@0.25,count@4")
        assert got == [("count_gap_floor~pool", 1.0), ("count_gap_floor~unvoted", 0.25), ("count", 4.0)]
        assert centroid_line_tag("count_gap_floor~unvoted", 1.0) == "centroid_line:count_gap_floor~unvoted@1"
        with pytest.raises(ValueError):
            parse_centroid_line_variants("count~everything@1")


def _heavy_tailed_person(n_neg: int = 10000, n_pos: int = 4, n_alike: int = 7, seed: int = 0) -> np.ndarray:
    """A face centroid's cosines on imagekit's ArcFace (#5325): a bulk with a heavier upper tail, matches far past it.

    As on FHIBE's buffalo_s cells: a few look-alikes sit 4 to 5 bulk spreads up, where a Gaussian
    bulk puts 0.3, and the person's photos about 8 spreads up.
    """
    rng = np.random.default_rng(seed)
    sd = 0.07
    bulk = rng.normal(0.02, sd, n_neg)
    alike = 0.02 + sd * rng.uniform(4.2, 4.9, n_alike)
    return np.concatenate([bulk, alike, 0.02 + sd * rng.uniform(7.5, 9.0, n_pos)])


class TestTheHigherBars:
    """#5325: the app's rule with the bulk's excess read above a higher bar."""

    def test_the_bar_rules_gate_as_the_apps_rule_does(self):
        from vtscore.detectors.centroid_head import CENTROID_BAR_ZS, CENTROID_GATED

        for z in CENTROID_BAR_ZS:
            gated = f"count_gap_floor_skip25_b1_z{z:g}"
            assert gated in CENTROID_LINE_RULES
            below, above = CENTROID_GATED[gated]
            for seed in range(4):
                cos = _heavy_tailed_person(seed=seed)
                for beta in (0.25, 0.5):
                    assert centroid_cut(cos, rule=gated, beta=beta) == centroid_cut(cos, rule=below, beta=beta)
                for beta in (1.0, 4.0):
                    assert centroid_cut(cos, rule=gated, beta=beta) == centroid_cut(cos, rule=above, beta=beta)

    def test_a_heavy_tail_is_counted_as_the_person_at_4_spreads_and_not_above_5(self):
        for seed in range(4):
            cos = _heavy_tailed_person(seed=seed)
            app = int((cos >= centroid_cut(cos, beta=1.0)).sum())
            bar = int((cos >= centroid_cut(cos, rule="count_gap_floor_skip25_b1_z5.5", beta=1.0)).sum())
            assert app > 4
            assert bar == 4

    def test_a_higher_bar_never_keeps_more_and_never_drops_below_the_drop(self):
        from vtscore.detectors.centroid_head import _top_gap_count

        for seed in range(4):
            for cos in (_rare_target(seed=seed), _heavy_tailed_person(seed=seed), _standout_person(seed=seed)):
                for beta in (0.25, 1.0, 4.0):
                    app = int((cos >= centroid_cut(cos, beta=beta)).sum())
                    for z in ("5", "5.5", "6"):
                        bar = int((cos >= centroid_cut(cos, rule=f"count_gap_floor_skip25_b1_z{z}", beta=beta)).sum())
                        assert bar <= app
                        assert bar >= _top_gap_count(cos, drop="skip25" if beta >= 1.0 else "widest")


class TestTheFaceRule:
    """#5325: face crops draw the centroid's line at the z5.5 bar; every other media type keeps the app's rule."""

    def test_each_media_type_gets_its_rule(self):
        from vtscore.detectors.centroid_head import centroid_line_rule

        assert centroid_line_rule("face") == "count_gap_floor_skip25_b1_z5.5"
        assert centroid_line_rule("face") in CENTROID_LINE_RULES
        for other in ("image", "video", "", None):
            assert centroid_line_rule(other) == CENTROID_LINE_RULE

    def test_the_harness_default_is_the_apps_rule_for_its_medias(self):
        import inspect

        from vtscore.eval.voting_iterations import production_centroid_line, simulate_voting_iterations

        assert inspect.signature(simulate_voting_iterations).parameters["centroid_line"].default is None
        assert production_centroid_line({1: {"media_type": "face"}}) == "count_gap_floor_skip25_b1_z5.5"
        assert production_centroid_line({1: {"media_type": "image"}}) == CENTROID_LINE_RULE
        assert production_centroid_line({1: {}}) == CENTROID_LINE_RULE
        assert production_centroid_line({}) == CENTROID_LINE_RULE
