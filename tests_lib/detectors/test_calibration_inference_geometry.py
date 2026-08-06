"""Production threshold calibration runs in inference geometry.

Issue #2731: on a patch dataset ``_build_vote_xy`` gives a Good vote one row
and a Bad vote its ~13 flooded leaves, and the calibrator collapses each bag
with ``max``.  ``max`` is an upward-biased order statistic, so 13 draws beat 1
with no signal at all - calibration understated what a positive scores relative
to a negative, while :func:`_score_all_media` scores *every* image as a max over
all ~24 region nodes.

The fix hands each bag the row stack the scorer will pool
(:func:`~vtscore.detectors.training.inference_score_rows`), so both classes are
max-over-24 exactly as at inference.  These tests pin the wiring, the
all-or-nothing coverage rule, the legacy no-op, and the cache-key invalidation.
"""

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vtscore.detectors.training import (
    _build_vote_xy,
    _calibration_score_rows,
    _flood_context,
    _score_all_media,
    inference_score_rows,
)
from vtscore.media.patch_embed import RegionVector


DIM = 8


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _patch_media(cid: int, rng) -> dict[str, Any]:
    """A media with a 4-node region tree: CLS + 2 leaves + 1 internal."""
    cls = _unit(rng.standard_normal(DIM))
    leaf1 = _unit(rng.standard_normal(DIM))
    leaf2 = _unit(rng.standard_normal(DIM))
    return {
        "id": cid,
        "md5": f"md5-{cid:04x}",
        "media_type": "image",
        "embedder": "dinov3_patch",
        "embeddings": {"dinov3_patch": cls},
        "patch_regions": [
            RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=cls, children=None),
            RegionVector(box=(0.0, 0.0, 0.5, 1.0), vec=leaf1, children=None),
            RegionVector(box=(0.5, 0.0, 1.0, 1.0), vec=leaf2, children=None),
            RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=_unit(leaf1 + leaf2), children=(1, 2)),
        ],
    }


def _legacy_media(cid: int, rng) -> dict[str, Any]:
    vec = _unit(rng.standard_normal(DIM))
    return {
        "id": cid,
        "md5": f"md5-{cid:04x}",
        "media_type": "image",
        "embedder": "siglip",
        "embeddings": {"siglip": vec},
    }


def _linear_scorer(direction):
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(np.asarray(direction, dtype=np.float32)[None, :] * 10.0))
        linear.bias.zero_()
    model = nn.Sequential(linear)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# inference_score_rows
# ---------------------------------------------------------------------------


class TestInferenceScoreRows:
    def test_patch_media_yields_every_region_node(self):
        media = _patch_media(1, np.random.default_rng(0))
        rows = inference_score_rows(media, "dinov3_patch")
        assert rows is not None
        # All 4 nodes - internals included, unlike ``bad_negative_vecs``.
        assert rows.shape == (4, DIM)
        for i, node in enumerate(media["patch_regions"]):
            np.testing.assert_allclose(rows[i], node.vec)

    def test_matches_the_rows_score_all_media_pools(self):
        """The stack must be exactly what the scorer max-pools, or calibration
        is measuring a different geometry than inference."""
        rng = np.random.default_rng(1)
        clips = {1: _patch_media(1, rng), 2: _patch_media(2, rng)}
        model = _linear_scorer(_unit(rng.standard_normal(DIM)))

        _ids, scores, _best = _score_all_media(model, clips, "dinov3_patch")
        for cid, scored in zip(sorted(clips), scores, strict=True):
            rows = inference_score_rows(clips[cid], "dinov3_patch")
            with torch.no_grad():
                pooled = float(torch.sigmoid(model(torch.tensor(rows))).max())
            assert abs(scored - pooled) < 1e-6

    def test_region_less_media_yields_its_single_image_vector(self):
        media = _legacy_media(1, np.random.default_rng(2))
        rows = inference_score_rows(media, "siglip")
        assert rows is not None
        assert rows.shape == (1, DIM)
        np.testing.assert_allclose(rows[0], media["embeddings"]["siglip"])

    def test_none_when_the_media_has_no_vector_at_all(self):
        assert inference_score_rows({"id": 1, "embeddings": {}}, "siglip") is None


# ---------------------------------------------------------------------------
# _build_vote_xy wiring
# ---------------------------------------------------------------------------


class TestBuildVoteXyScoreRows:
    def test_both_classes_get_the_same_width_stack(self):
        rng = np.random.default_rng(3)
        clips = {1: _patch_media(1, rng), 2: _patch_media(2, rng)}
        X, y, groups, score_rows = _build_vote_xy(clips, {1: None}, {2: None}, {}, "dinov3_patch")

        # Training is still asymmetric - 1 Good row vs 3 flooded Bad rows...
        assert y == [1.0, 0.0, 0.0, 0.0]
        assert len(X) == 4
        # ...but every bag is *scored* over its full 4-node tree.
        assert set(score_rows) == {("g", 1), ("b", 2)}
        assert score_rows[("g", 1)].shape == score_rows[("b", 2)].shape == (4, DIM)

    def test_legacy_dataset_stacks_are_single_rows(self):
        rng = np.random.default_rng(4)
        clips = {1: _legacy_media(1, rng), 2: _legacy_media(2, rng)}
        _X, _y, groups, score_rows = _build_vote_xy(clips, {1: None}, {2: None}, {}, "siglip")
        assert groups == [("g", 1), ("b", 2)]
        assert all(rows.shape == (1, DIM) for rows in score_rows.values())


# ---------------------------------------------------------------------------
# _calibration_score_rows - the gate
# ---------------------------------------------------------------------------


class TestCalibrationScoreRowsGate:
    @staticmethod
    def _flooded():
        rng = np.random.default_rng(5)
        clips = {1: _patch_media(1, rng), 2: _patch_media(2, rng)}
        return _build_vote_xy(clips, {1: None}, {2: None}, {}, "dinov3_patch")

    def test_covered_flooded_labels_produce_a_stack_per_bag(self):
        X, y, groups, score_rows = self._flooded()
        _n, cal_groups, _w = _flood_context(X, y, groups)
        out = _calibration_score_rows(groups, cal_groups, score_rows)
        assert out is not None
        assert set(out) == {("g", 1), ("b", 2)}
        assert all(isinstance(v, np.ndarray) and v.shape == (4, DIM) for v in out.values())

    def test_unflooded_labels_are_a_no_op(self):
        """Legacy datasets must reach the calibrator byte-identically."""
        rng = np.random.default_rng(6)
        clips = {1: _legacy_media(1, rng), 2: _legacy_media(2, rng)}
        X, y, groups, score_rows = _build_vote_xy(clips, {1: None}, {2: None}, {}, "siglip")
        _n, cal_groups, _w = _flood_context(X, y, groups)
        assert cal_groups is None
        assert _calibration_score_rows(groups, cal_groups, score_rows) is None

    def test_partial_coverage_declines_rather_than_skewing(self):
        """A missing Good stack would leave that bag at max-over-1 while the Bad
        bags widened to max-over-4 - deeper than the bias being corrected."""
        X, y, groups, score_rows = self._flooded()
        _n, cal_groups, _w = _flood_context(X, y, groups)
        score_rows.pop(("g", 1))
        assert _calibration_score_rows(groups, cal_groups, score_rows) is None

    def test_empty_score_rows_declines(self):
        X, y, groups, _score_rows = self._flooded()
        _n, cal_groups, _w = _flood_context(X, y, groups)
        assert _calibration_score_rows(groups, cal_groups, {}) is None
        assert _calibration_score_rows(groups, cal_groups, None) is None


# ---------------------------------------------------------------------------
# The bias this corrects
# ---------------------------------------------------------------------------


class TestGoodBagNoLongerUnderstated:
    """With one model held fixed, widening a Good bag from its 1 training row to
    its full scoring stack can only raise that bag's calibration score - which
    is exactly the understatement the old geometry baked in."""

    @staticmethod
    def _bags(rng, n_good=3, n_bad=3, bad_rows=4):
        X, y, groups, score_rows = [], [], [], {}
        for g in range(n_good):
            rows = np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(4)])
            X.append(rows[0])  # the snapped node the Good vote trains on
            y.append(1.0)
            groups.append(("g", g))
            score_rows[("g", g)] = rows
        for b in range(n_bad):
            rows = np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(bad_rows)])
            for vec in rows[:bad_rows]:
                X.append(vec)
                y.append(0.0)
                groups.append(("b", b))
            score_rows[("b", b)] = rows
        return X, y, groups, score_rows

    def test_positive_calibration_scores_rise_to_inference_width(self, monkeypatch):
        import vtscore.training.mlp as mlp_mod
        from vtscore.training.thresholds import compute_fold_orderings

        rng = np.random.default_rng(7)
        model = _linear_scorer(_unit(rng.normal(0, 1, DIM)))
        monkeypatch.setattr(mlp_mod, "train_model", lambda *a, **k: model)
        X, y, groups, score_rows = self._bags(rng)

        # A fresh RandomState per call - it is stateful, so sharing one instance
        # would give the two calls different fold splits and prove nothing.
        def _kwargs() -> dict[str, Any]:
            return dict(rng=np.random.RandomState(42), calibrate_count=1, hidden_dim=8, groups=groups)

        before, _ = compute_fold_orderings(X, y, DIM, **_kwargs())
        after, _ = compute_fold_orderings(X, y, DIM, score_rows_by_group=score_rows, **_kwargs())

        before_pos = [s for s, lbl in zip(*before[0], strict=True) if lbl == 1.0]
        after_pos = [s for s, lbl in zip(*after[0], strict=True) if lbl == 1.0]
        assert before_pos and after_pos
        # Max over a superset can only be >=, and here it strictly rises.
        assert all(a >= b - 1e-9 for a, b in zip(after_pos, before_pos, strict=True))
        assert any(a > b + 1e-6 for a, b in zip(after_pos, before_pos, strict=True))

    def test_bad_bags_gain_the_internal_nodes_they_never_flooded(self, monkeypatch):
        """A Bad vote trains its leaves down but is *scored* over the internals
        too; calibration now sees those rows."""
        import vtscore.training.mlp as mlp_mod
        from vtscore.training.thresholds import _pooled_group_scores

        rng = np.random.default_rng(8)
        model = _linear_scorer(_unit(rng.normal(0, 1, DIM)))
        monkeypatch.setattr(mlp_mod, "train_model", lambda *a, **k: model)
        X, y, groups, score_rows = self._bags(rng, n_good=1, n_bad=1, bad_rows=2)
        # The Bad bag trained on 2 rows but is scored over 4.
        rows_by_group = {("b", 0): [1, 2], ("g", 0): [0]}
        X_np = np.stack(X)
        trained = _pooled_group_scores(model, [("b", 0)], rows_by_group, X_np, None)
        scored = _pooled_group_scores(model, [("b", 0)], rows_by_group, X_np, score_rows)
        assert scored[0] >= trained[0] - 1e-9


# ---------------------------------------------------------------------------
# Calibration cache
# ---------------------------------------------------------------------------


class TestCalibrationCacheKey:
    @staticmethod
    def _key(score_rows):
        from vtscore.training.thresholds import _calibration_cache_key

        rng = np.random.default_rng(9)
        X = [_unit(rng.normal(0, 1, DIM)) for _ in range(4)]
        y = [1.0, 1.0, 0.0, 0.0]
        groups = [("g", 0), ("g", 1), ("b", 2), ("b", 3)]
        return _calibration_cache_key(X, y, 2, 0.5, 8, groups, score_rows)

    def test_score_rows_enter_the_key(self):
        rng = np.random.default_rng(10)
        rows_a = {("g", 0): np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(3)])}
        rows_b = {("g", 0): np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(3)])}
        assert self._key(None) != self._key(rows_a)
        assert self._key(rows_a) != self._key(rows_b)
        assert self._key(rows_a) == self._key(dict(rows_a))

    def test_key_is_insertion_order_independent(self):
        rng = np.random.default_rng(11)
        a = np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(3)])
        b = np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(3)])
        assert self._key({("g", 0): a, ("b", 1): b}) == self._key({("b", 1): b, ("g", 0): a})

    def test_cached_orderings_are_not_served_across_a_geometry_change(self):
        from vtscore.state.core import DetectorContext
        from vtscore.training.thresholds import calibration_folds_cached

        rng = np.random.default_rng(12)
        X = [_unit(rng.normal(0, 1, DIM)) for _ in range(4)]
        X += [_unit(rng.normal(0, 1, DIM)) for _ in range(2)]
        y = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        groups = [("g", 0), ("g", 1), ("b", 2), ("b", 2), ("b", 3), ("b", 3)]
        score_rows = {g: np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(4)]) for g in set(groups)}

        det = DetectorContext("d1")

        def _call(rows):
            return calibration_folds_cached(
                X,
                y,
                DIM,
                calibrate_count=1,
                calibration_fraction=0.5,
                hidden_dim=8,
                det_ctx=det,
                groups=groups,
                score_rows_by_group=rows,
            )

        _call(None)
        cached = det.calibration_cache
        assert cached is not None
        key_without = cached[0]
        _call(score_rows)
        cached = det.calibration_cache
        assert cached is not None
        assert cached[0] != key_without


@pytest.mark.parametrize("embedder", ["dinov3_patch", None])
def test_score_rows_survive_a_missing_embedder_name(embedder):
    """``_build_vote_xy`` resolves the embedder itself when handed ``None``; the
    region stack is read off the tree either way."""
    rng = np.random.default_rng(13)
    clips = {1: _patch_media(1, rng), 2: _patch_media(2, rng)}
    _X, _y, _groups, score_rows = _build_vote_xy(clips, {1: None}, {2: None}, {}, embedder)
    assert set(score_rows) == {("g", 1), ("b", 2)}
    assert all(rows.shape == (4, DIM) for rows in score_rows.values())
