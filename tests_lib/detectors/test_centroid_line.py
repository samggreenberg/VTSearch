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

from vtscore.detectors.centroid_head import CENTROID_LINE_RULE, CENTROID_LINE_RULES, centroid_cut
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
    def test_the_app_draws_the_midpoint(self):
        assert CENTROID_LINE_RULE == "midpoint"
        cos = _rare_target()
        assert centroid_cut(cos) == pytest.approx(calculate_gmm_threshold(cos.tolist()))

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
        from tests_lib.detectors.test_example_opening import _medias
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

    def test_the_midpoint_variant_is_the_apps_row(self):
        rows, _ = self._run(centroid_line_variants=["midpoint@1"])
        base = self._centroid_rows(rows)
        tagged = self._centroid_rows(rows, centroid_line_tag("midpoint", 1.0))
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

    def test_refuses_an_unknown_rule(self):
        with pytest.raises(ValueError, match="centroid_line"):
            self._run(centroid_line="tail")
