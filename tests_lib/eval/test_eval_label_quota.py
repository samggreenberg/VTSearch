"""The eval harness's default arm scores the withheld half as Test does under the label quota (#4643).

Test gives the Goods' centroid from the first Good until the votes hold 3 Goods
and 4 Bads, and the trained head from there.  "The Eval Default Arm IS the App"
(CLAUDE.md): so a default run writes a row at every click from the first Good,
its test metrics are that detector's, and ``detector_tier`` names it.  The
Train side - what Autopilot picks - is the trained head's wherever there is one,
so the quota cannot move the trajectory.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.detectors.label_quota import TIER_CENTROID, TIER_TRAINED, label_quota
from vtscore.eval.voting_iterations import simulate_voting_iterations


def _clips(n_per_cat: int = 24, dim: int = 12, seed: int = 0) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    centres = {"alpha": rng.standard_normal(dim), "beta": rng.standard_normal(dim)}
    out: dict[int, dict] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat else "beta"
        vec = centres[cat] + 0.9 * rng.standard_normal(dim)
        out[i + 1] = {
            "id": i + 1,
            "embeddings": {"emb": (vec / np.linalg.norm(vec)).astype(np.float32)},
            "category": cat,
        }
    return out


def _run(**kwargs):
    picks: list[dict] = []
    rows = simulate_voting_iterations(
        _clips(), "alpha", seed=0, max_steps=16, calibrate_count=2, spot_check="off", pick_sink=picks, **kwargs
    )
    return rows, picks


@pytest.fixture(scope="module")
def default_run():
    return _run()


@pytest.fixture(scope="module")
def pre_quota_run():
    return _run(label_quota=False)


class TestTheDefaultArm:
    def test_rows_start_at_the_first_good(self, default_run):
        rows, picks = default_run
        first_good = min(p["t"] for p in picks if p["picked_label"] == 1)
        assert [r["t"] for r in rows] == list(range(first_good, 17))

    def test_each_row_is_the_detector_the_quota_gives(self, default_run):
        rows, _picks = default_run
        assert all(r["detector_tier"] == label_quota(r["n_good"], r["n_bad"]).tier for r in rows)
        assert {r["detector_tier"] for r in rows} == {TIER_CENTROID, TIER_TRAINED}, "the run crosses the quota"

    def test_a_centroid_row_before_the_first_bad_has_metrics_and_no_train_line(self, default_run):
        rows, _picks = default_run
        no_bad = [r for r in rows if r["n_bad"] == 0]
        assert no_bad, "the opening seeds Goods first"
        for r in no_bad:
            assert r["detector_tier"] == TIER_CENTROID
            assert math.isfinite(r["auroc"]) and math.isfinite(r["recall"])
            # No head yet: nothing for the Train side to record.
            assert math.isnan(r["acq_threshold"]) and r["floor_status"] == ""

    def test_the_trained_rows_are_the_pre_quota_arms(self, default_run, pre_quota_run):
        """From the quota on nothing changed: the rows are the head's, as they always were."""
        rows, _ = default_run
        old, _ = pre_quota_run
        cols = ("cost", "precision", "recall", "auroc", "average_precision", "acq_threshold")
        trained = {r["t"]: r for r in rows if r["detector_tier"] == TIER_TRAINED}
        old_by_t = {r["t"]: r for r in old}
        assert trained
        for t, r in trained.items():
            for col in cols:
                assert r[col] == old_by_t[t][col] or (math.isnan(r[col]) and math.isnan(old_by_t[t][col])), (t, col)

    def test_the_quota_does_not_move_the_trajectory(self, default_run, pre_quota_run):
        """The Train side runs on the trained head either way, so Autopilot picks the same items."""
        _, picks = default_run
        _, old_picks = pre_quota_run
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in old_picks]


class TestThePreQuotaArm:
    def test_rows_start_at_the_first_good_and_bad_and_are_all_the_head(self, pre_quota_run):
        rows, _picks = pre_quota_run
        assert rows and all(r["n_good"] >= 1 and r["n_bad"] >= 1 for r in rows)
        assert {r["detector_tier"] for r in rows} == {TIER_TRAINED}


class TestTheCalibrationPath:
    def test_a_centroid_row_says_so_in_its_provenance(self):
        rows = simulate_voting_iterations(
            _clips(),
            "alpha",
            seed=0,
            max_steps=10,
            calibrate_count=2,
            spot_check="off",
            style="whole_image",
            emit_calibration_metrics=True,
        )
        centroid = [r for r in rows if r["detector_tier"] == TIER_CENTROID]
        assert centroid
        assert {r["threshold_provenance"] for r in centroid} == {"centroid"}
        assert all(r["threshold"] == 0.5 for r in centroid)


class TestTheCentroidTestIsThePlantedAnswer:
    def test_it_cuts_the_withheld_half_at_the_midpoint_of_its_cosines(self):
        """Cosine to the unit mean of the Goods, cut at the GMM midpoint on the test half: the metrics that gives."""
        from vtscore.eval.labels import media_is_positive
        from vtscore.eval.voting_iterations import _centroid_test
        from vtscore.training.thresholds import calculate_gmm_threshold

        clips = _clips()
        goods = {1: None, 2: None}
        test_ids = list(range(5, 49, 2))
        _step, rows, calibration = _centroid_test(
            goods,
            clips,
            test_ids,
            "alpha",
            0,
            region_voting=False,
            region_aware=False,
            style_obj=None,
            beta=1.0,
            calibration_rows=False,
        )
        assert calibration is None and len(rows) == 1

        vecs = np.stack([clips[c]["embeddings"]["emb"] for c in goods]).astype(np.float64)
        c = vecs.mean(axis=0)
        c /= np.linalg.norm(c)
        cos = np.array([float(clips[t]["embeddings"]["emb"] @ c) for t in test_ids])
        kept = cos >= calculate_gmm_threshold(cos.tolist())
        truth = np.array([media_is_positive(clips[t], "alpha") for t in test_ids])
        tp = int((kept & truth).sum())
        assert rows[0]["recall"] == pytest.approx(tp / truth.sum(), abs=1e-6)
        assert rows[0]["precision"] == pytest.approx(tp / kept.sum(), abs=1e-6)
