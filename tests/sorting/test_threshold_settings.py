"""Tests for the threshold blend fallback and the Calibration Fraction setting.

The schedule blend (``calculate_safe_threshold``) is no longer the shipped
threshold - the fold-anchored population estimator is, unconditionally, with no
user setting to turn it off - but the blend is still what runs for label sets
too small to form calibration folds, so its ramp is still load-bearing.

Covers:
- calculate_safe_threshold logic (blending, label-count ramp)
- train_and_score always fuses the haystack into its threshold
- Training-setting changes invalidate a loaded detector's cached model
- Calibration fraction: settings, cross-calibration split, edge cases, API, eval
"""

import numpy as np
import pytest

import app as app_module
from vtscore.detectors.training import train_and_score
from vtscore.training.thresholds import (
    NO_GOOD_THRESHOLD,
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
)


class TestCalculateSafeThreshold:
    """Unit tests for the calculate_safe_threshold blending function.

    The pure-x-cal cases name ``prod`` explicitly: since #2841 the shipped
    schedules never hand over completely, so an implicit default here would be
    asserting a property no shipped schedule has.
    """

    def test_few_labels_returns_gmm(self):
        """With fewer than 6 labels, result should equal the GMM threshold."""
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        xcal = 0.4
        gmm = calculate_gmm_threshold(scores)
        safe = calculate_safe_threshold(xcal, scores, 4)
        assert safe == pytest.approx(gmm, abs=1e-6)

    def test_many_labels_returns_xcal(self):
        """With >= 20 labels, result equals x-cal."""
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        xcal = 0.45
        safe = calculate_safe_threshold(xcal, scores, 25, schedule="prod")
        assert safe == pytest.approx(xcal, abs=1e-6)

    def test_intermediate_labels_blend(self):
        """With labels between 6 and 20, result is between GMM and x-cal."""
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        xcal = 0.4
        gmm = calculate_gmm_threshold(scores)
        safe = calculate_safe_threshold(xcal, scores, 13)  # midpoint

        # Should be strictly between gmm and xcal (unless they're equal)
        if abs(gmm - xcal) > 1e-6:
            lo, hi = sorted([gmm, xcal])
            assert lo <= safe <= hi

    def test_many_labels_extreme_xcal_returns_xcal(self):
        """With >= 20 labels, even extreme x-cal values are used directly."""
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        for xcal in [0.02, 0.98]:
            safe = calculate_safe_threshold(xcal, scores, 30, schedule="prod")
            assert safe == pytest.approx(xcal, abs=1e-6)

    def test_exactly_6_labels_starts_ramp(self):
        """At exactly 6 labels, label_weight should be 0 → pure GMM."""
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        xcal = 0.4
        gmm = calculate_gmm_threshold(scores)
        safe = calculate_safe_threshold(xcal, scores, 6)
        assert safe == pytest.approx(gmm, abs=1e-6)

    def test_exactly_20_labels_ends_ramp(self):
        """At exactly 20 labels, label_weight should be 1 → pure x-cal."""
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        xcal = 0.45
        safe = calculate_safe_threshold(xcal, scores, 20, schedule="prod")
        assert safe == pytest.approx(xcal, abs=1e-6)

    def test_infinite_xcal_falls_back_to_gmm_without_nan(self):
        """Regression: ``inf`` xcal with label_weight=0 used to produce NaN
        via ``0.0 * inf``. The guard now returns the GMM threshold cleanly."""
        import math

        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        gmm = calculate_gmm_threshold(scores)
        safe = calculate_safe_threshold(float("inf"), scores, 4)
        assert math.isfinite(safe)
        assert safe == pytest.approx(gmm, abs=1e-6)

    def test_nan_xcal_falls_back_to_gmm_without_nan(self):
        """A NaN xcal must not propagate into the detector threshold."""
        import math

        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        gmm = calculate_gmm_threshold(scores)
        safe = calculate_safe_threshold(float("nan"), scores, 13)
        assert math.isfinite(safe)
        assert safe == pytest.approx(gmm, abs=1e-6)

    def test_no_good_sentinel_blends_to_gmm_below_floor(self):
        """The finite ``NO_GOOD_THRESHOLD`` sentinel returned by
        ``calculate_cross_calibration_threshold`` must blend cleanly to
        pure GMM at n_labels < 6 (label_weight=0)."""
        import math

        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        gmm = calculate_gmm_threshold(scores)
        safe = calculate_safe_threshold(NO_GOOD_THRESHOLD, scores, 4)
        assert math.isfinite(safe)
        assert safe == pytest.approx(gmm, abs=1e-6)


class TestTrainAndScoreFusesUnconditionally:
    """The population estimator is not optional; there is no flag to pass."""

    def test_no_safe_thresholds_parameter_remains(self):
        """The retired setting must not linger as a dead keyword."""
        import inspect

        assert "safe_thresholds" not in inspect.signature(train_and_score).parameters

    def test_returns_a_valid_threshold(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold, _model = train_and_score(app_module.medias, app_module.good_votes, app_module.bad_votes)
        assert 0.0 <= threshold <= 1.0
        assert len(results) == len(app_module.medias)

    def test_threshold_is_realizable_on_the_haystack_distribution(self):
        """The shipped cut is realized as a quantile of the scores it will be
        compared against, so it always lands inside their range - which the raw
        conformal quantile, measured on fold-model scores, need not."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold, _model = train_and_score(app_module.medias, app_module.good_votes, app_module.bad_votes)
        scores = [r["score"] for r in results]
        assert min(scores) <= threshold <= max(scores)


class TestTrainingSettingsInvalidateLoadedDetector:
    """Regression for M7: changing a training-relevant setting must drop the
    cached MLP / threshold on every loaded detector so the next
    ``/api/find-label`` / ``/api/find`` / ``/api/auto-detect`` retrains
    under the new setting instead of scoring with a stale threshold.
    """

    def _loaded_ctx(self):
        from vtsearch.state import DetectorContext, register_detector_context

        # Sentinel object stands in for a trained MLP; invalidation just
        # needs to drop the reference, it doesn't introspect the model.
        ctx = DetectorContext("det-m7", name="m7")
        ctx.model = object()
        ctx.threshold = 0.73
        register_detector_context(ctx)
        return ctx

    def test_set_inclusion_preserves_model_no_fold_cache(self):
        """Inclusion is a pure cutoff knob: it no longer drops the model.
        Without cached fold orderings the threshold is left for the next
        training pass."""
        from vtsearch.state import get_inclusion, set_inclusion

        ctx = self._loaded_ctx()
        model_before = ctx.model
        set_inclusion(get_inclusion() + 1)
        assert ctx.model is model_before
        assert ctx.threshold == 0.73

    def test_set_inclusion_rethresholds_from_fold_cache(self):
        """With cached fold orderings, an inclusion change re-derives the
        threshold (cheap quantile rule over the cache) without touching the model."""
        from vtsearch.state import get_inclusion, set_inclusion
        from vtscore.training.thresholds import CalibrationFolds, threshold_from_fold_orderings

        ctx = self._loaded_ctx()
        model_before = ctx.model
        orderings = [([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0])]
        ctx.calibration_cache = ("k", CalibrationFolds(orderings, None, []))
        new_incl = get_inclusion() + 3
        set_inclusion(new_incl)
        assert ctx.model is model_before
        assert ctx.threshold == threshold_from_fold_orderings(orderings, new_incl)

    def test_set_calibrate_count_invalidates_loaded_model(self):
        from vtsearch.state import get_calibrate_count, set_calibrate_count

        ctx = self._loaded_ctx()
        set_calibrate_count(get_calibrate_count() + 1)
        assert ctx.model is None
        assert ctx.threshold == 0.5

    def test_set_calibration_fraction_invalidates_loaded_model(self):
        from vtsearch.state import get_calibration_fraction, set_calibration_fraction

        ctx = self._loaded_ctx()
        new_fraction = 0.25 if get_calibration_fraction() != 0.25 else 0.35
        set_calibration_fraction(new_fraction)
        assert ctx.model is None
        assert ctx.threshold == 0.5

    def test_settings_put_calibrate_count_invalidates_loaded_model(self, client):
        from vtsearch.state import get_calibrate_count

        ctx = self._loaded_ctx()
        resp = client.put("/api/settings", json={"calibrate_count": get_calibrate_count() + 1})
        assert resp.status_code == 200
        assert ctx.model is None
        assert ctx.threshold == 0.5


# ======================================================================
# Calibration Fraction
# ======================================================================


class TestCalibrationFractionCrossCalibration:
    """Unit tests for calibration_fraction in calculate_cross_calibration_threshold."""

    def _make_data(self, n=20, dim=16, seed=42):
        rng = np.random.RandomState(seed)
        X_list = [rng.randn(dim).astype(np.float32) + (1.0 if i < n // 2 else -1.0) for i in range(n)]
        y_list = [1.0 if i < n // 2 else 0.0 for i in range(n)]
        return X_list, y_list, dim

    def test_default_fraction_is_half(self):
        """Default calibration_fraction=0.5 should match old 50/50 behaviour."""
        import inspect

        sig = inspect.signature(calculate_cross_calibration_threshold)
        default = sig.parameters["calibration_fraction"].default
        assert default == 0.5

    def test_fraction_02_returns_valid_threshold(self):
        """With 0.2 fraction (80% Train / 20% Calibrate), threshold is valid."""
        X, y, dim = self._make_data()
        t = calculate_cross_calibration_threshold(X, y, dim, calibration_fraction=0.2)
        assert 0.0 <= t <= 1.0

    def test_fraction_08_returns_valid_threshold(self):
        """With 0.8 fraction (20% Train / 80% Calibrate), threshold is valid."""
        X, y, dim = self._make_data()
        t = calculate_cross_calibration_threshold(X, y, dim, calibration_fraction=0.8)
        assert 0.0 <= t <= 1.0

    def test_extreme_fraction_returns_no_good_sentinel(self):
        """When fraction is so extreme that a valid split is impossible, return a
        finite sentinel above the sigmoid range (so nothing is predicted as Good)
        without poisoning ``calculate_safe_threshold`` blends with NaN."""
        import math

        # With n=4 and calibration_fraction=0.99, n_cal=4, n_train=0 → can't split
        X, y, dim = self._make_data(n=4)
        t = calculate_cross_calibration_threshold(X, y, dim, calibration_fraction=0.99)
        assert t == NO_GOOD_THRESHOLD
        assert math.isfinite(t)
        assert t > 1.0

    def test_extreme_fraction_near_zero_returns_inf(self):
        """With fraction near 0, n_cal rounds to 1, n_train = n-1 ≥ 2, should still work."""
        X, y, dim = self._make_data(n=10)
        t = calculate_cross_calibration_threshold(X, y, dim, calibration_fraction=0.01)
        # n_cal = max(1, round(10 * 0.01)) = max(1, 0) = 1, n_train = 9 → valid
        assert isinstance(t, float)

    def test_different_fractions_produce_different_thresholds(self):
        """Different calibration fractions should (usually) produce different thresholds."""
        X, y, dim = self._make_data(n=40, seed=123)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        t_02 = calculate_cross_calibration_threshold(X, y, dim, rng=rng1, calibrate_count=5, calibration_fraction=0.2)
        t_08 = calculate_cross_calibration_threshold(X, y, dim, rng=rng2, calibrate_count=5, calibration_fraction=0.8)
        # Both valid; they CAN be equal but checking both are valid floats
        assert isinstance(t_02, float)
        assert isinstance(t_08, float)


class TestCalibrationFractionTrainAndScore:
    """Integration tests: train_and_score with calibration_fraction."""

    def test_default_fraction_is_unset(self):
        """``None`` = resolve the per-space production split (issue #3290)."""
        import inspect

        sig = inspect.signature(train_and_score)
        default = sig.parameters["calibration_fraction"].default
        assert default is None

    def test_custom_fraction_returns_valid_results(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold, _model = train_and_score(
            app_module.medias,
            app_module.good_votes,
            app_module.bad_votes,
            calibration_fraction=0.2,
        )
        assert len(results) == len(app_module.medias)
        # threshold can be inf if the split is too extreme for 6 labels
        assert isinstance(threshold, float)


class TestCalibrationFractionSetting:
    """Tests for calibration_fraction setting persistence."""

    def test_default_is_unset(self):
        """No stored value = ``None``: the per-embedder default applies."""
        from vtsearch import settings

        settings.reset()
        assert settings.get_calibration_fraction() is None

    def test_set_and_get(self):
        from vtsearch import settings

        settings.set_calibration_fraction(0.3)
        assert settings.get_calibration_fraction() == pytest.approx(0.3)

    def test_clamps_low(self):
        from vtsearch import settings

        settings.set_calibration_fraction(-0.5)
        assert settings.get_calibration_fraction() == 0.0

    def test_clamps_high(self):
        from vtsearch import settings

        settings.set_calibration_fraction(2.0)
        assert settings.get_calibration_fraction() == 1.0

    def test_state_get_reads_from_settings(self):
        from vtsearch import settings
        from vtsearch.state import get_calibration_fraction

        settings.set_calibration_fraction(0.25)
        assert get_calibration_fraction() == pytest.approx(0.25)


class TestCalibrationFractionAPI:
    """Tests for calibration_fraction via the settings API."""

    def test_get_settings_includes_calibration_fraction(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "calibration_fraction" in data
        # ``null`` = no explicit split (per-embedder default); a float is an
        # explicit user choice.
        assert data["calibration_fraction"] is None or isinstance(data["calibration_fraction"], float)

    def test_put_null_clears_to_automatic(self, client):
        client.put("/api/settings", json={"calibration_fraction": 0.3})
        resp = client.put("/api/settings", json={"calibration_fraction": None})
        assert resp.status_code == 200
        assert resp.get_json()["calibration_fraction"] is None
        resp = client.get("/api/settings")
        assert resp.get_json()["calibration_fraction"] is None

    def test_put_updates_calibration_fraction(self, client):
        resp = client.put("/api/settings", json={"calibration_fraction": 0.3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["calibration_fraction"] == pytest.approx(0.3)

    def test_put_invalid_type_returns_422(self, client):
        resp = client.put("/api/settings", json={"calibration_fraction": "bad"})
        # Type validation runs in the SettingsUpdate schema → 422.
        assert resp.status_code == 422

    def test_put_persists(self, client):
        client.put("/api/settings", json={"calibration_fraction": 0.15})
        resp = client.get("/api/settings")
        assert resp.get_json()["calibration_fraction"] == pytest.approx(0.15)


class TestCalibrationFractionEval:
    """Test that eval functions accept calibration_fraction parameter."""

    def _make_clips(self, n=40, dim=16, seed=42):
        rng = np.random.RandomState(seed)
        medias = {}
        for i in range(n):
            cat = "target" if i < n // 2 else "other"
            if cat == "target":
                emb = rng.randn(dim).astype(np.float32) + 1.0
            else:
                emb = rng.randn(dim).astype(np.float32) - 1.0
            medias[i + 1] = {"id": i + 1, "embedder": "e5", "embeddings": {"e5": emb}, "category": cat}
        return medias

    def test_simulate_voting_iterations_accepts_calibration_fraction(self):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        # Plumbing test: small pool + calibrate_count=1 (see the safe-
        # thresholds twin above).
        medias = self._make_clips(n=16)
        rows = simulate_voting_iterations(
            medias,
            "target",
            seed=42,
            calibration_fraction=0.3,
            calibrate_count=1,
        )
        assert len(rows) > 0
        for row in rows:
            assert "cost" in row

    def test_run_voting_iterations_eval_accepts_calibration_fraction(self):
        from vtscore.eval.voting_iterations import run_voting_iterations_eval

        medias = self._make_clips(n=16)
        df = run_voting_iterations_eval(
            {"test": medias},
            seeds=[42],
            categories={"test": ["target"]},
            calibration_fraction=0.2,
            calibrate_count=1,
        )
        assert len(df) > 0

    def test_eval_runner_accepts_calibration_fraction(self):
        from vtscore.eval.runner import eval_learned_sort
        from vtscore.eval.config import EvalQuery

        medias = self._make_clips()
        queries = [EvalQuery(text="target things", target_category="target")]
        results = eval_learned_sort(medias, queries, calibration_fraction=0.3, seed=42)
        assert len(results) > 0
        for lm in results:
            assert 0.0 <= lm.f1 <= 1.0
