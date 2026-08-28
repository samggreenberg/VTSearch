"""The per-embedder ``calibration_fraction`` default (issues #3287 / #3290).

The shipped Train/Calibrate split is keyed on the space the detector learns
in: 0.3 for single-vector embedders, 0.5 for patch-grid embedders, 0.5 when
the space is unknown.  An explicit value always wins.  The app resolves the
embedder's ``supports_patch_regions`` capability; the eval harness resolves
``patch_grid`` presence - both through the same
:func:`vtscore.training.thresholds.production_split_for` table, so the values
cannot drift (the ``training.split_fraction_default`` mirror in
``scripts/check-eval-app-sync.py`` pins the predicates).
"""

import numpy as np
import pytest

from vtscore.detectors.training import resolve_calibration_fraction
from vtscore.training.thresholds import (
    PRODUCTION_SPLIT,
    PRODUCTION_SPLIT_BY_SPACE,
    production_split_for,
)


class TestProductionSplitFor:
    def test_single_vector_space(self):
        assert production_split_for(patch_space=False) == PRODUCTION_SPLIT_BY_SPACE["single_vector"] == 0.3

    def test_patch_space(self):
        assert production_split_for(patch_space=True) == PRODUCTION_SPLIT_BY_SPACE["patch"] == 0.5

    def test_unknown_space_takes_fallback(self):
        """The three-state contract: ``None`` is "unknown", not a guess."""
        assert production_split_for(patch_space=None) == PRODUCTION_SPLIT == 0.5


class TestResolveCalibrationFraction:
    def test_explicit_setting_wins_on_any_embedder(self):
        assert resolve_calibration_fraction(0.42, "siglip") == pytest.approx(0.42)
        assert resolve_calibration_fraction(0.42, "dinov3_patch") == pytest.approx(0.42)
        assert resolve_calibration_fraction(0.42, None) == pytest.approx(0.42)

    def test_explicit_half_is_still_explicit(self):
        """0.5 typed by the user is an explicit choice, not "unset"."""
        assert resolve_calibration_fraction(0.5, "siglip") == 0.5

    def test_single_vector_embedder_resolves_to_030(self):
        assert resolve_calibration_fraction(None, "siglip") == 0.3

    def test_patch_embedder_resolves_to_050_regardless_of_style(self):
        """The predicate reads the model's *capability*: ``dinov3_patch``
        wants 0.5 even in the boxless whole-image fallback, where it emits no
        patches at all (#3287 measured 0.3 as +0.015 worse there)."""
        assert resolve_calibration_fraction(None, "dinov3_patch") == 0.5

    def test_no_embedder_takes_unknown_fallback(self):
        assert resolve_calibration_fraction(None, None) == PRODUCTION_SPLIT
        assert resolve_calibration_fraction(None, "") == PRODUCTION_SPLIT

    def test_unregistered_embedder_takes_unknown_fallback(self):
        """A name the registry doesn't know is "unknown", not "single-vector"."""
        assert resolve_calibration_fraction(None, "no_such_embedder_xyz") == PRODUCTION_SPLIT


class TestEvalDefaultResolution:
    """``simulate_voting_iterations(calibration_fraction=None)`` resolves the
    app's per-space split, keyed on ``patch_grid`` presence."""

    def _clips(self, n=16, dim=16, seed=42, patch_grid=False):
        rng = np.random.default_rng(seed)
        medias = {}
        for i in range(n):
            cat = "target" if i < n // 2 else "other"
            emb = rng.standard_normal(dim).astype(np.float32) + (1.0 if cat == "target" else -1.0)
            media = {"id": i + 1, "embedder": "e5", "embeddings": {"e5": emb}, "category": cat}
            if patch_grid:
                media["patch_grid"] = rng.standard_normal((4, dim)).astype(np.float32)
            medias[i + 1] = media
        return medias

    def test_default_resolves_single_vector_split(self, monkeypatch):
        from vtscore.eval.voting_iterations import simulate_voting_iterations
        from vtscore.training import thresholds

        seen: list[bool | None] = []
        real = thresholds.production_split_for

        def spy(*, patch_space):
            seen.append(patch_space)
            return real(patch_space=patch_space)

        monkeypatch.setattr(thresholds, "production_split_for", spy)
        rows = simulate_voting_iterations(
            self._clips(),
            "target",
            seed=42,
            calibrate_count=1,
        )
        assert len(rows) > 0
        assert seen == [False]

    def test_explicit_fraction_skips_resolution(self, monkeypatch):
        from vtscore.eval.voting_iterations import simulate_voting_iterations
        from vtscore.training import thresholds

        seen: list[bool | None] = []
        real = thresholds.production_split_for

        def spy(*, patch_space):
            seen.append(patch_space)
            return real(patch_space=patch_space)

        monkeypatch.setattr(thresholds, "production_split_for", spy)
        rows = simulate_voting_iterations(
            self._clips(),
            "target",
            seed=42,
            calibration_fraction=0.5,
            calibrate_count=1,
        )
        assert len(rows) > 0
        assert seen == []
