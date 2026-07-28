"""Tests for the Max-Patch experiment detection styles.

Covers the pieces added for ``docs/plans/max-patch-experiment.md``: the
``nearest_patch_to_box`` helper, the three detection styles in
:mod:`vtscore.eval.patch_styles` (whole_image / max_hac / max_patch), and the
``style`` wiring through the voting-iterations harness.

Everything runs on small synthetic patch grids - no model downloads.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from vtscore.eval.patch_styles import MaxHacStyle, MaxPatchStyle, WholeImageStyle, resolve_style
from vtscore.eval.voting_iterations import _VOTING_COLUMNS, run_voting_iterations_eval, simulate_voting_iterations
from vtscore.media.patch_embed import (
    PatchEmbedOutput,
    build_region_tree,
    nearest_patch_to_box,
    snap_box_to_region,
    to_fp16,
)

_TIMING_COLS = {"elapsed_seconds", "train_seconds", "xcal_seconds", "pool_score_seconds", "test_score_seconds"}

DIM = 16
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
    model = nn.Sequential(nn.Linear(DIM, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(np.asarray(direction, dtype=np.float32)[None, :] * 10.0))
        model[0].bias.zero_()
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
        with pytest.raises(ValueError):
            nearest_patch_to_box(np.zeros((4, 4)), (0, 0, 1, 1))
        with pytest.raises(ValueError):
            nearest_patch_to_box(np.zeros((4, 4, 8)), (0, 0, 1))


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

    def test_bad_vecs_flood_every_patch(self):
        rng = np.random.default_rng(12)
        media = _patch_media(1, "cat1", rng)
        vecs = MaxPatchStyle().bad_vecs(media)
        assert len(vecs) == GRID * GRID
        flat = np.asarray(media["patch_grid"], dtype=np.float32).reshape(-1, DIM)
        np.testing.assert_allclose(np.stack(vecs), flat, rtol=1e-3)

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
        # The planted image's score equals the sigmoid of its best patch row.
        flat = np.asarray(planted["patch_grid"], dtype=np.float32).reshape(-1, DIM)
        expected = float(1.0 / (1.0 + np.exp(-(flat @ (target * 10.0)).max())))
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
# MaxHacStyle - production parity
# ---------------------------------------------------------------------------


class TestMaxHacStyle:
    def test_good_vec_snaps_to_region_node(self):
        rng = np.random.default_rng(20)
        media = _patch_media(1, "cat0", rng)
        box = _cell_box(1, 1)
        got = MaxHacStyle().good_vec(media, box)
        expected = snap_box_to_region(media["patch_regions"], box)
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
    @pytest.mark.parametrize("style", ["whole_image", "max_hac", "max_patch"])
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
        medias, _ = _planted_dataset(n_per_cat=20, seed=8)
        kwargs = dict(
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
        medias, _ = _planted_dataset(n_per_cat=25, seed=9)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=1,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=16,
            style="max_patch",
        )
        # The planted patch is a perfectly separable signal; by the last step
        # the ranking metrics should reflect a learned detector.
        assert rows[-1]["average_precision"] > 0.8

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
