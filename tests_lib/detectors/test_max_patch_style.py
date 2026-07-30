"""Tests for the Max-Patch experiment detection styles.

Covers the pieces added for ``docs/plans/max-patch-experiment.md``: the
``nearest_patch_to_box`` helper, the three detection styles in
:mod:`vtscore.eval.patch_styles` (whole_image / max_hac / max_patch), and the
``style`` wiring through the voting-iterations harness.

Everything runs on small synthetic patch grids - no model downloads.
"""

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vtscore.eval.patch_styles import (
    MaxHacStyle,
    MaxPatchHacStyle,
    MaxPatchPcaHacStyle,
    MaxPatchStyle,
    WholeImageStyle,
    build_patch_hac_tree,
    resolve_style,
)
from vtscore.eval.voting_iterations import _VOTING_COLUMNS, run_voting_iterations_eval, simulate_voting_iterations
from vtscore.media.patch_embed import (
    PatchEmbedOutput,
    build_region_tree,
    nearest_patch_to_box,
    snap_box_to_region,
    to_fp16,
)

_TIMING_COLS = {"elapsed_seconds", "train_seconds", "xcal_seconds", "pool_score_seconds", "test_score_seconds"}

DIM = 32
GRID = 4  # 4x4 patch grid
K = 4  # HAC leaves


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _make_grid(rng, plant_vec=None, plant_cell=None):
    """Random unit-vector (GRID, GRID, DIM) grid, optionally planting one cell."""
    grid = rng.normal(0, 1.0, (GRID, GRID, DIM)).astype(np.float32)
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)
    if plant_vec is not None:
        grid[plant_cell] = _unit(plant_vec)
    return grid


def _cell_box(row, col):
    """The exact normalised box of grid cell (row, col)."""
    return (col / GRID, row / GRID, (col + 1) / GRID, (row + 1) / GRID)


def _patch_media(mid, category, rng, plant_vec=None, plant_cell=None, with_region_label=None):
    """A synthetic patch-dataset media: CLS vector, fp16 grid, HAC region tree."""
    grid = _make_grid(rng, plant_vec, plant_cell)
    saliency = np.full((GRID, GRID), 1.0 / (GRID * GRID), dtype=np.float32)
    cls_vec = _unit(grid.reshape(-1, DIM).mean(axis=0))
    output = PatchEmbedOutput(cls_vec=cls_vec, patch_grid=grid, patch_saliency=saliency)
    media = {
        "id": mid,
        "category": category,
        "embeddings": {"emb": cls_vec},
        "patch_grid": grid.astype(np.float16),
        "patch_regions": to_fp16(build_region_tree(output, k=K, alpha=0.5)),
    }
    if with_region_label is not None and plant_cell is not None:
        media["regions"] = [{"box": list(_cell_box(*plant_cell)), "label": with_region_label}]
    return media


def _planted_dataset(n_per_cat=30, seed=0):
    """Two-category patch dataset where cat0 images carry a planted target patch.

    Every cat0 image plants the (noised) target vector in one grid cell and
    annotates that cell as its ground-truth region; cat1 images are pure noise.
    """
    rng = np.random.default_rng(seed)
    target = _unit(np.eye(DIM, dtype=np.float32)[0] * 4.0)
    medias = {}
    mid = 1
    for _ in range(n_per_cat):
        cell = (int(rng.integers(0, GRID)), int(rng.integers(0, GRID)))
        vec = _unit(target + rng.normal(0, 0.1, DIM).astype(np.float32))
        medias[mid] = _patch_media(mid, "cat0", rng, plant_vec=vec, plant_cell=cell, with_region_label="cat0")
        mid += 1
    for _ in range(n_per_cat):
        medias[mid] = _patch_media(mid, "cat1", rng)
        mid += 1
    return medias, target


def _linear_scorer(direction):
    """A hand-built ``nn.Sequential`` whose score is monotone in ``x @ direction``."""
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(np.asarray(direction, dtype=np.float32)[None, :] * 10.0))
        linear.bias.zero_()
    model = nn.Sequential(linear)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# nearest_patch_to_box
# ---------------------------------------------------------------------------


class TestNearestPatchToBox:
    def test_box_over_cell_returns_that_cell(self):
        rng = np.random.default_rng(1)
        grid = _make_grid(rng)
        for row, col in [(0, 0), (2, 1), (3, 3)]:
            got = nearest_patch_to_box(grid, _cell_box(row, col))
            np.testing.assert_allclose(got, _unit(grid[row, col]), rtol=1e-5)

    def test_multi_cell_box_prefers_center_cell(self):
        rng = np.random.default_rng(2)
        grid = _make_grid(rng)
        # Box spanning a 3x3 block of cells centred on (1, 1).
        box = (0.0, 0.0, 0.75, 0.75)
        got = nearest_patch_to_box(grid, box)
        np.testing.assert_allclose(got, _unit(grid[1, 1]), rtol=1e-5)

    def test_thin_box_snaps_to_nearest_center(self):
        rng = np.random.default_rng(3)
        grid = _make_grid(rng)
        # Zero-area box at the middle of cell (0, 3).
        box = (0.875, 0.125, 0.875, 0.125)
        got = nearest_patch_to_box(grid, box)
        np.testing.assert_allclose(got, _unit(grid[0, 3]), rtol=1e-5)

    def test_swapped_and_out_of_range_corners_tolerated(self):
        rng = np.random.default_rng(4)
        grid = _make_grid(rng)
        straight = nearest_patch_to_box(grid, (0.25, 0.5, 0.5, 0.75))
        swapped = nearest_patch_to_box(grid, (0.5, 0.75, 0.25, 0.5))
        np.testing.assert_allclose(straight, swapped)
        clamped = nearest_patch_to_box(grid, (-3.0, -3.0, 4.0, 4.0))
        assert np.isfinite(clamped).all()

    def test_result_is_unit_norm_even_from_fp16(self):
        rng = np.random.default_rng(5)
        grid = _make_grid(rng).astype(np.float16)
        got = nearest_patch_to_box(np.asarray(grid), (0.0, 0.0, 0.25, 0.25))
        assert got.dtype == np.float32
        assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-3

    def test_bad_shapes_raise(self):
        from typing import Any

        with pytest.raises(ValueError):
            nearest_patch_to_box(np.zeros((4, 4)), (0, 0, 1, 1))
        three_tuple: Any = (0, 0, 1)
        with pytest.raises(ValueError):
            nearest_patch_to_box(np.zeros((4, 4, 8)), three_tuple)


# ---------------------------------------------------------------------------
# MaxPatchStyle
# ---------------------------------------------------------------------------


class TestMaxPatchStyle:
    def test_good_vec_with_box_is_nearest_patch(self):
        rng = np.random.default_rng(10)
        media = _patch_media(1, "cat0", rng)
        style = MaxPatchStyle()
        box = _cell_box(2, 3)
        got = style.good_vec(media, box)
        expected = nearest_patch_to_box(np.asarray(media["patch_grid"]), box)
        np.testing.assert_allclose(got, expected)

    def test_good_vec_without_box_is_whole_image_vector(self):
        rng = np.random.default_rng(11)
        media = _patch_media(1, "cat0", rng)
        style = MaxPatchStyle()
        np.testing.assert_allclose(style.good_vec(media, None), media["embeddings"]["emb"])

    def test_bad_vecs_flood_whole_image_vector_and_every_patch(self):
        rng = np.random.default_rng(12)
        media = _patch_media(1, "cat1", rng)
        vecs = MaxPatchStyle().bad_vecs(media)
        # The full-image row leads, then every raw patch: a Bad vote must
        # suppress the *entire* scoring pool, exactly as ``max_hac`` floods its
        # CLS node alongside the HAC leaves.
        assert len(vecs) == GRID * GRID + 1
        np.testing.assert_allclose(vecs[0], media["embeddings"]["emb"], rtol=1e-3)
        flat = np.asarray(media["patch_grid"], dtype=np.float32).reshape(-1, DIM)
        np.testing.assert_allclose(np.stack(vecs[1:]), flat, rtol=1e-3)

    def test_gridless_media_falls_back_to_whole_image(self):
        media = {"id": 1, "category": "c", "embeddings": {"emb": _unit(np.ones(DIM))}}
        style = MaxPatchStyle()
        assert len(style.bad_vecs(media)) == 1
        np.testing.assert_allclose(style.good_vec(media, (0, 0, 1, 1)), media["embeddings"]["emb"])
        scores = style.score_media(_linear_scorer(np.ones(DIM)), {1: media})
        assert set(scores) == {1}

    def test_score_media_is_max_over_patches(self):
        rng = np.random.default_rng(13)
        target = _unit(np.eye(DIM, dtype=np.float32)[0])
        planted = _patch_media(1, "cat0", rng, plant_vec=target, plant_cell=(1, 2))
        noise = _patch_media(2, "cat1", rng)
        style = MaxPatchStyle()
        scores = style.score_media(_linear_scorer(target), {1: planted, 2: noise})
        assert scores[1] > scores[2]
        # The planted image's score equals the sigmoid of its best scoring row
        # (the full-image row plus every patch).
        rows = style.score_rows(planted)
        expected = float(1.0 / (1.0 + np.exp(-(rows @ (target * 10.0)).max())))
        assert abs(scores[1] - expected) < 1e-3

    def test_exemplar_sims_max_over_patches(self):
        rng = np.random.default_rng(14)
        target = _unit(np.eye(DIM, dtype=np.float32)[1])
        planted = _patch_media(1, "cat0", rng, plant_vec=target, plant_cell=(0, 0))
        noise = _patch_media(2, "cat1", rng)
        sims = MaxPatchStyle().exemplar_sims({1: planted, 2: noise}, target)
        assert sims[1] > sims[2]
        assert sims[1] == pytest.approx(1.0, abs=2e-3)


# ---------------------------------------------------------------------------
# Train/score geometry parity - the invariant that broke MaxPatch on Caltech
# ---------------------------------------------------------------------------


class TestTrainScoreGeometryParity:
    """Every vector a style can train a vote on must be a row it also scores.

    When it is not, the classifier learns to separate the *training* geometry
    from the *scoring* geometry (each Bad vote floods scoring-geometry rows as
    negatives), calibration measures positives in a geometry inference never
    evaluates, and the threshold lands outside the production score range -
    perfect ranking, FPR 0, catastrophic FNR.  ``max_hac`` always satisfied
    this because ``patch_regions[0]`` is the CLS full-image node; ``max_patch``
    did not until the full-image row was added to its pool.
    """

    @pytest.mark.parametrize(
        "style_cls", [MaxPatchStyle, MaxHacStyle, WholeImageStyle, MaxPatchHacStyle, MaxPatchPcaHacStyle]
    )
    def test_boxless_good_vote_trains_on_a_scored_row(self, style_cls):
        rng = np.random.default_rng(40)
        media = _patch_media(1, "cat0", rng)
        style = style_cls()
        # A boxless Good vote (Caltech-101: no ground-truth regions at all).
        vote_vec = np.asarray(style.good_vec(media, None), dtype=np.float32)
        rows = style.score_rows(media)
        assert any(np.allclose(r, vote_vec, atol=2e-3) for r in rows), (
            f"{style_cls.__name__}: boxless Good vote trains on a vector that is "
            f"not among the {len(rows)} rows this style max-pools at inference"
        )

    @pytest.mark.parametrize(
        "style_cls", [MaxPatchStyle, MaxHacStyle, WholeImageStyle, MaxPatchHacStyle, MaxPatchPcaHacStyle]
    )
    def test_boxed_good_vote_trains_on_a_scored_row(self, style_cls):
        rng = np.random.default_rng(41)
        media = _patch_media(1, "cat0", rng)
        style = style_cls()
        vote_vec = np.asarray(style.good_vec(media, _cell_box(2, 1)), dtype=np.float32)
        rows = style.score_rows(media)
        assert any(np.allclose(r, vote_vec, atol=2e-3) for r in rows)

    @pytest.mark.parametrize("style_cls", [MaxPatchStyle, WholeImageStyle, MaxPatchHacStyle, MaxPatchPcaHacStyle])
    def test_bad_vote_suppresses_every_scored_row(self, style_cls):
        """A Bad vote asserts *no* row of the image should score high, so its
        flood must cover the whole scoring pool - otherwise an un-suppressed
        row survives to max-pool the image back up at inference."""
        rng = np.random.default_rng(42)
        media = _patch_media(1, "cat1", rng)
        style = style_cls()
        flooded = [np.asarray(v, dtype=np.float32) for v in style.bad_vecs(media)]
        for row in style.score_rows(media):
            assert any(np.allclose(row, f, atol=2e-3) for f in flooded), (
                f"{style_cls.__name__}: a scored row is never trained down by a Bad vote"
            )

    def test_max_hac_floods_every_scored_row_except_hac_internals(self):
        """``max_hac``'s flood covers the CLS node and the leaves, but not the
        HAC **internal** nodes - which it nonetheless max-pools at inference.

        That gap is deliberate (``bad_negative_vecs`` drops internals as
        saliency-weighted pools of leaves it already suppresses, so they would
        add correlated duplicates without adding coverage), but it means
        ``max_hac`` scores rows no Bad vote trains down directly.  Pinned here
        so the exception stays explicit rather than silent.
        """
        rng = np.random.default_rng(46)
        media = _patch_media(1, "cat1", rng)
        style = MaxHacStyle()
        flooded = [np.asarray(v, dtype=np.float32) for v in style.bad_vecs(media)]
        regions = media["patch_regions"]
        rows = style.score_rows(media)
        assert len(rows) == len(regions)
        uncovered = [i for i, row in enumerate(rows) if not any(np.allclose(row, f, atol=2e-3) for f in flooded)]
        assert uncovered, "expected the HAC internals to be uncovered"
        # Every uncovered row is an internal node; no leaf and not the CLS node.
        assert all(regions[i].children is not None for i in uncovered)
        assert 0 not in uncovered

    @pytest.mark.parametrize(
        "style_cls", [MaxPatchStyle, MaxHacStyle, WholeImageStyle, MaxPatchHacStyle, MaxPatchPcaHacStyle]
    )
    def test_score_media_is_max_pool_over_score_rows(self, style_cls):
        rng = np.random.default_rng(43)
        media = _patch_media(1, "c", rng)
        direction = _unit(rng.normal(0, 1, DIM))
        style = style_cls()
        got = style.score_media(_linear_scorer(direction), {1: media})[1]
        rows = style.score_rows(media)
        expected = float(1.0 / (1.0 + np.exp(-(rows @ (direction * 10.0)).max())))
        assert got == pytest.approx(expected, abs=2e-3)

    def test_max_patch_score_rows_lead_with_whole_image_vector(self):
        rng = np.random.default_rng(44)
        media = _patch_media(1, "c", rng)
        rows = MaxPatchStyle().score_rows(media)
        assert rows.shape == (GRID * GRID + 1, DIM)
        np.testing.assert_allclose(rows[0], media["embeddings"]["emb"], rtol=1e-3)

    def test_max_hac_score_rows_include_the_cls_node(self):
        """The structural reason ``max_hac`` never showed the Caltech failure."""
        rng = np.random.default_rng(45)
        media = _patch_media(1, "c", rng)
        regions = media["patch_regions"]
        assert regions[0].children is None
        assert regions[0].box == (0.0, 0.0, 1.0, 1.0)
        rows = MaxHacStyle().score_rows(media)
        np.testing.assert_allclose(rows[0], np.asarray(regions[0].vec, dtype=np.float32), rtol=1e-3)


class TestMaxPatchHacStyle:
    """The raw-patch-leaf HAC hybrid: multi-scale tree, snap, all-node flood."""

    def test_tree_node_count_and_scales(self):
        rng = np.random.default_rng(20)
        media = _patch_media(1, "cat0", rng)
        tree = build_patch_hac_tree(np.asarray(media["patch_grid"], dtype=np.float32), media["embeddings"]["emb"])
        n = GRID * GRID
        assert len(tree) == 2 * n  # CLS whole-image node + n raw-patch leaves + (n-1) merges
        assert tree[0].box == (0.0, 0.0, 1.0, 1.0)
        leaves = [t for t in tree if t.children is None]
        internals = [t for t in tree if t.children is not None]
        assert len(leaves) == n + 1
        assert len(internals) == n - 1
        areas = [(t.box[2] - t.box[0]) * (t.box[3] - t.box[1]) for t in internals]
        assert max(areas) == pytest.approx(1.0, abs=1e-6)
        assert min(areas) < 0.5

    def test_score_rows_lead_with_whole_image_and_flood_covers_all(self):
        rng = np.random.default_rng(21)
        media = _patch_media(1, "c", rng)
        style = MaxPatchHacStyle()
        rows = style.score_rows(media)
        assert rows.shape == (2 * GRID * GRID, DIM)  # every tree node is scored
        np.testing.assert_allclose(rows[0], media["embeddings"]["emb"], atol=3e-3)
        flooded = [np.asarray(v, dtype=np.float32) for v in style.bad_vecs(media)]
        assert len(flooded) == rows.shape[0]  # all-node flood covers every scored row

    def test_good_vec_snaps_multiscale(self):
        rng = np.random.default_rng(22)
        media = _patch_media(1, "cat0", rng)
        style = MaxPatchHacStyle()
        got_full = style.good_vec(media, (0.0, 0.0, 1.0, 1.0))
        np.testing.assert_allclose(got_full, _unit(media["embeddings"]["emb"]), atol=3e-3)
        got_small = style.good_vec(media, _cell_box(1, 1))
        assert abs(float(np.linalg.norm(got_small)) - 1.0) < 1e-3

    def test_good_vec_without_box_is_whole_image(self):
        rng = np.random.default_rng(23)
        media = _patch_media(1, "cat0", rng)
        np.testing.assert_allclose(MaxPatchHacStyle().good_vec(media, None), media["embeddings"]["emb"])

    def test_gridless_media_falls_back(self):
        media = {"id": 1, "category": "c", "embeddings": {"emb": _unit(np.ones(DIM))}}
        style = MaxPatchHacStyle()
        assert len(style.bad_vecs(media)) == 1
        np.testing.assert_allclose(style.good_vec(media, (0, 0, 1, 1)), media["embeddings"]["emb"])
        assert set(style.score_media(_linear_scorer(np.ones(DIM)), {1: media})) == {1}

    def test_learns_planted_signal(self):
        medias, target = _planted_dataset(n_per_cat=25, seed=25)
        seed_scores = resolve_style("max_patch_hac").exemplar_sims(medias, target)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=1,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=12,
            style="max_patch_hac",
            seed_scores=seed_scores,
        )
        assert rows
        assert rows[-1]["average_precision"] > 0.7  # all-node flood learns the planted signal


class TestMaxPatchPcaHacStyle:
    """MaxPatchHAC with a PCA-denoised merge order: same scoring, different tree."""

    def test_pca_changes_tree_topology_not_node_count(self):
        rng = np.random.default_rng(30)
        media = _patch_media(1, "cat0", rng)
        grid = np.asarray(media["patch_grid"], dtype=np.float32)
        cls = media["embeddings"]["emb"]
        raw = build_patch_hac_tree(grid, cls)
        pca = build_patch_hac_tree(grid, cls, pca_dims=8)
        assert len(raw) == len(pca)  # same node count (CLS + n leaves + n-1 merges)
        # the merge order (internal children) differs under PCA
        assert [n.children for n in raw] != [n.children for n in pca]

    def test_subclass_reuses_maxpatchhac_scoring_geometry(self):
        rng = np.random.default_rng(31)
        media = _patch_media(1, "c", rng)
        style = MaxPatchPcaHacStyle()
        rows = style.score_rows(media)
        assert rows.shape == (2 * GRID * GRID, DIM)
        np.testing.assert_allclose(rows[0], media["embeddings"]["emb"], atol=3e-3)
        assert len(style.bad_vecs(media)) == rows.shape[0]  # all-node flood covers every row

    def test_learns_planted_signal(self):
        medias, target = _planted_dataset(n_per_cat=25, seed=32)
        seed_scores = resolve_style("max_patch_pca_hac").exemplar_sims(medias, target)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=1,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=12,
            style="max_patch_pca_hac",
            seed_scores=seed_scores,
        )
        assert rows
        assert rows[-1]["average_precision"] > 0.7


# ---------------------------------------------------------------------------
# Calibration in inference geometry
# ---------------------------------------------------------------------------


class TestCalibrationInInferenceGeometry:
    """``compute_fold_orderings(score_rows_by_group=...)`` collapses each bag
    over the rows the *scorer* pools, not the rows the fold model trained on.

    Without it a Good bag is a max over its 1 training row while a Bad bag is a
    max over the ~197 it flooded; ``max`` is an upward-biased order statistic,
    so the min-cost cut lands systematically high and over-rejects positives.
    """

    @staticmethod
    def _fixed_model_patch(monkeypatch, direction):
        """Pin every fold fit to one known linear model, so fold scores are exact."""
        import vtscore.training.mlp as mlp_mod

        model = _linear_scorer(direction)
        monkeypatch.setattr(mlp_mod, "train_model", lambda *a, **k: model)
        return model

    @staticmethod
    def _bags(rng, n_good=3, n_bad=3, bad_rows=5):
        X, y, groups = [], [], []
        for g in range(n_good):
            X.append(_unit(rng.normal(0, 1, DIM)))
            y.append(1.0)
            groups.append(("g", g))
        for b in range(n_bad):
            for _ in range(bad_rows):
                X.append(_unit(rng.normal(0, 1, DIM)))
                y.append(0.0)
                groups.append(("b", b))
        return X, y, groups

    def test_group_score_is_max_over_supplied_rows(self, monkeypatch):
        from vtscore.training.thresholds import compute_fold_orderings

        rng = np.random.default_rng(50)
        direction = _unit(rng.normal(0, 1, DIM))
        self._fixed_model_patch(monkeypatch, direction)
        X, y, groups = self._bags(rng)

        # Give every bag - Good and Bad alike - a 4-row inference stack.
        score_rows = {g: np.stack([_unit(rng.normal(0, 1, DIM)) for _ in range(4)]) for g in set(groups)}
        orderings, fallback = compute_fold_orderings(
            X,
            y,
            DIM,
            rng=np.random.RandomState(42),
            calibrate_count=1,
            hidden_dim=8,
            groups=groups,
            score_rows_by_group=score_rows,
        )
        assert fallback is None
        scores, labels = orderings[0]
        assert len(scores) == len(labels)
        # Every returned score must be the max-pooled sigmoid over that bag's
        # supplied rows - and must match one of the bags exactly.
        pooled = {float(1.0 / (1.0 + np.exp(-(rows @ (direction * 10.0)).max()))) for rows in score_rows.values()}
        for s in scores:
            assert any(abs(s - p) < 1e-4 for p in pooled)

    def test_override_changes_the_ordering_vs_training_geometry(self, monkeypatch):
        """The override is not a no-op: pooling over inference rows gives a
        different calibration ordering than pooling over training rows."""
        from vtscore.training.thresholds import compute_fold_orderings

        rng = np.random.default_rng(51)
        direction = _unit(rng.normal(0, 1, DIM))
        self._fixed_model_patch(monkeypatch, direction)
        X, y, groups = self._bags(rng)

        # A fresh RandomState per call: it is stateful, so sharing one instance
        # would give the two calls different fold splits and prove nothing.
        def _kwargs() -> dict[str, Any]:
            return dict(rng=np.random.RandomState(42), calibrate_count=1, hidden_dim=8, groups=groups)

        base, _ = compute_fold_orderings(X, y, DIM, **_kwargs())
        # Give each Good bag the wide stack it would really be scored over:
        # its single training row plus extra rows that can only raise the max.
        score_rows = {}
        for g in set(groups):
            rows = [X[i] for i, gg in enumerate(groups) if gg == g]
            if g[0] == "g":
                rows = rows + [_unit(direction + 0.01 * rng.normal(0, 1, DIM))]
            score_rows[g] = np.stack(rows)
        widened, _ = compute_fold_orderings(X, y, DIM, score_rows_by_group=score_rows, **_kwargs())

        base_pos = [s for s, lbl in zip(*base[0], strict=True) if lbl == 1.0]
        wide_pos = [s for s, lbl in zip(*widened[0], strict=True) if lbl == 1.0]
        assert base_pos and wide_pos
        # Max over a superset can only be >=, and here it strictly rises: the
        # single-row Good bag was understating what production will score.
        assert all(w >= b - 1e-9 for w, b in zip(wide_pos, base_pos, strict=True))
        assert any(w > b + 1e-6 for w, b in zip(wide_pos, base_pos, strict=True))

    def test_none_override_leaves_production_path_byte_identical(self):
        """Every live caller passes ``None``; that path must not shift."""
        from vtscore.training.thresholds import compute_fold_orderings

        rng = np.random.default_rng(52)
        X, y, groups = self._bags(rng)

        def _kwargs() -> dict[str, Any]:
            return dict(rng=np.random.RandomState(42), calibrate_count=2, hidden_dim=8, groups=groups)

        torch.manual_seed(0)
        a, fa = compute_fold_orderings(X, y, DIM, **_kwargs())
        torch.manual_seed(0)
        b, fb = compute_fold_orderings(X, y, DIM, score_rows_by_group=None, **_kwargs())
        assert fa == fb
        assert a == b


# ---------------------------------------------------------------------------
# MaxHacStyle - production parity
# ---------------------------------------------------------------------------


class TestMaxHacStyle:
    def test_good_vec_snaps_to_region_node(self):
        rng = np.random.default_rng(20)
        media = _patch_media(1, "cat0", rng)
        box = _cell_box(1, 1)
        got = MaxHacStyle().good_vec(media, box)
        expected = snap_box_to_region(media["patch_regions"], box)
        assert expected is not None
        np.testing.assert_allclose(got, expected)

    def test_bad_vecs_are_cls_plus_leaves(self):
        rng = np.random.default_rng(21)
        media = _patch_media(1, "cat1", rng)
        vecs = MaxHacStyle().bad_vecs(media)
        # childless nodes = 1 CLS + K leaves; internals are excluded.
        assert len(vecs) == K + 1

    def test_score_media_matches_production_scorer(self):
        from vtscore.detectors.training import score_media_with_model

        rng = np.random.default_rng(22)
        clips = {mid: _patch_media(mid, "c", rng) for mid in (1, 2, 3)}
        direction = _unit(rng.normal(0, 1, DIM))
        model = _linear_scorer(direction)
        style_scores = MaxHacStyle().score_media(model, clips)
        prod_scores = {r["id"]: r["score"] for r in score_media_with_model(model, clips)}
        for mid in clips:
            # Production rounds to 4 decimals; the style path keeps raw floats.
            assert style_scores[mid] == pytest.approx(prod_scores[mid], abs=1e-3)

    def test_exemplar_sims_max_over_region_nodes(self):
        rng = np.random.default_rng(23)
        media = _patch_media(1, "c", rng)
        query = _unit(rng.normal(0, 1, DIM))
        sims = MaxHacStyle().exemplar_sims({1: media}, query)
        expected = max(float(np.asarray(r.vec, dtype=np.float32) @ query) for r in media["patch_regions"])
        assert sims[1] == pytest.approx(expected, abs=2e-3)


# ---------------------------------------------------------------------------
# WholeImageStyle
# ---------------------------------------------------------------------------


class TestWholeImageStyle:
    def test_votes_and_scores_use_image_vector(self):
        rng = np.random.default_rng(30)
        media = _patch_media(1, "c", rng)
        style = WholeImageStyle()
        np.testing.assert_allclose(style.good_vec(media, (0.0, 0.0, 0.5, 0.5)), media["embeddings"]["emb"])
        assert len(style.bad_vecs(media)) == 1
        direction = _unit(rng.normal(0, 1, DIM))
        scores = style.score_media(_linear_scorer(direction), {1: media})
        cls_vec = np.asarray(media["embeddings"]["emb"], dtype=np.float32)
        expected = float(1.0 / (1.0 + np.exp(-(cls_vec @ (direction * 10.0)))))
        assert scores[1] == pytest.approx(expected, abs=1e-4)

    def test_resolve_style_registry(self):
        assert isinstance(resolve_style("whole_image"), WholeImageStyle)
        assert isinstance(resolve_style("max_hac"), MaxHacStyle)
        assert isinstance(resolve_style("max_patch"), MaxPatchStyle)
        # Fresh instance per call: the matrix memo must not leak across runs.
        assert resolve_style("max_patch") is not resolve_style("max_patch")
        with pytest.raises(KeyError):
            resolve_style("nope")


# ---------------------------------------------------------------------------
# Harness wiring
# ---------------------------------------------------------------------------


def _drop_timing(rows):
    return [{k: v for k, v in r.items() if k not in _TIMING_COLS} for r in rows]


class TestStyleVotingSimulation:
    @pytest.mark.parametrize("style", ["whole_image", "max_hac", "max_patch", "max_patch_hac", "max_patch_pca_hac"])
    def test_style_run_produces_learnable_rows(self, style):
        medias, _target = _planted_dataset(n_per_cat=25, seed=7)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=0,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=10,
            style=style,
        )
        assert rows, f"style {style} produced no rows"
        for r in rows:
            assert r["style"] == style
            assert set(_VOTING_COLUMNS) == set(r.keys())
            assert np.isfinite(r["cost"])
            assert 0.0 <= r["average_precision"] <= 1.0

    def test_style_runs_are_deterministic(self):
        from typing import Any

        medias, _ = _planted_dataset(n_per_cat=20, seed=8)
        kwargs: dict[str, Any] = dict(
            target_category="cat0",
            seed=3,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=8,
            style="max_patch",
        )
        a = simulate_voting_iterations(dict(medias), **kwargs)
        b = simulate_voting_iterations(dict(medias), **kwargs)
        assert _drop_timing(a) == _drop_timing(b)

    def test_max_patch_learns_planted_signal(self):
        medias, target = _planted_dataset(n_per_cat=25, seed=9)
        seed_scores = resolve_style("max_patch").exemplar_sims(medias, target)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=1,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=24,
            seed_scores=seed_scores,
            style="max_patch",
        )
        # The planted patch is a separable signal: the ranking must sit far
        # above chance (prevalence 0.5) throughout the back half of the run.
        # The exact values are deterministic (~0.79-0.80) but left slack for
        # cross-platform torch numerics.  No monotone-improvement assertion:
        # exemplar seeding makes even the first trainable step strong, and AP
        # wobbles a few points between retrains.
        assert all(r["average_precision"] > 0.7 for r in rows[len(rows) // 2 :])

    def test_style_rejects_svm_trainer(self):
        medias, _ = _planted_dataset(n_per_cat=10, seed=10)
        with pytest.raises(ValueError, match="MLP trainer"):
            simulate_voting_iterations(
                medias,
                target_category="cat0",
                seed=0,
                trainer="svm_linear",
                style="max_patch",
            )

    def test_default_run_records_empty_style(self):
        medias, _ = _planted_dataset(n_per_cat=12, seed=11)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=0,
            dataset_name="synthetic",
            max_steps=6,
        )
        assert rows
        assert all(r["style"] == "" for r in rows)

    def test_eval_wrapper_runs_style_grid(self):
        medias, _ = _planted_dataset(n_per_cat=12, seed=12)
        df = run_voting_iterations_eval(
            {"synthetic": medias},
            seeds=[0],
            categories={"synthetic": ["cat0"]},
            region_voting=True,
            max_steps=5,
            styles=["max_hac", "max_patch"],
        )
        assert set(df["style"].unique()) == {"max_hac", "max_patch"}
        assert list(df.columns) == list(_VOTING_COLUMNS)

    def test_safe_thresholds_with_style(self):
        medias, _ = _planted_dataset(n_per_cat=15, seed=13)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=2,
            dataset_name="synthetic",
            region_voting=True,
            safe_thresholds=True,
            max_steps=8,
            style="max_patch",
        )
        assert rows
        assert all(np.isfinite(r["cost"]) for r in rows)

    def test_exemplar_seed_scores_drive_seed_phase(self):
        medias, target = _planted_dataset(n_per_cat=15, seed=14)
        style = resolve_style("max_patch")
        seed_scores = style.exemplar_sims(medias, target)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=4,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=8,
            seed_scores=seed_scores,
            style="max_patch",
        )
        assert rows
        # Seeding follows the exemplar ranking, whose top items are the planted
        # positives - so the first trainable step already has 1+ good votes.
        assert rows[0]["n_good"] >= 1
