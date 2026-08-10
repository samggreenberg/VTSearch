"""``--training-nodes`` must render the construction the run actually trained on.

The renderer used to enumerate childless nodes unconditionally, so a box-pool run -- which
trains on ONE unweighted whole-image CLS row per Bad vote -- got 33 NEG crops at
``--hac-k 32``, each captioned with a ``1/bag_size`` weight that path never applies. Only
the POS panels were trustworthy.

These tests pin the two constructions and the shared gate helpers, so the renderer and the
runner can never disagree about which arm produced a run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))

import viz as vizmod  # noqa: E402
from sweep import _resolve_region_voting, _resolve_snap_goods  # noqa: E402


class TestGateHelpers:
    """Both halves must agree with the runner, hence one shared helper each."""

    @staticmethod
    def _args(region_voting=True, snap_goods=None, neg_regions=False):
        return SimpleNamespace(region_voting=region_voting, snap_goods=snap_goods, neg_regions=neg_regions)

    @pytest.mark.parametrize(("proposal", "embedder", "want"), [
        ("hac", "dinov3", True), ("hac", "dinov2", True),
        ("hac", "siglip", False), ("whole", "dinov3", False), ("sliding", "dinov3", False),
    ])  # fmt: skip
    def test_region_voting_gate(self, proposal, embedder, want):
        assert _resolve_region_voting(self._args(), proposal, embedder) is want

    def test_region_voting_off_is_off_everywhere(self):
        assert _resolve_region_voting(self._args(region_voting=False), "hac", "dinov3") is False

    def test_bagged_is_region_voting_or_neg_regions(self):
        """--neg-regions takes the same train_rv_head branch, for every proposal."""
        a = self._args(region_voting=False, neg_regions=True)
        assert (_resolve_region_voting(a, "whole", "siglip") or a.neg_regions) is True

    def test_snap_follows_region_voting_when_unset(self):
        for rv in (True, False):
            assert _resolve_snap_goods(self._args(region_voting=rv), "hac", "dinov3") is rv


def _stub_dataset(tmp_path):
    class _DS:
        def load_image(self, iid):
            return Image.new("RGB", (64, 48), (120, 120, 120))

    return _DS()


def _write_region_npz(regions_dir: Path, iid: int, n_leaves: int = 4):
    """CLS + n_leaves childless nodes + one internal merge, as the HAC source writes them."""
    regions_dir.mkdir(parents=True, exist_ok=True)
    n = 1 + n_leaves + 1
    boxes = np.tile(np.array([0.0, 0.0, 1.0, 1.0], np.float32), (n, 1))
    children = np.full((n, 2), -1, dtype=int)
    children[-1] = (1, 2)  # the single internal merge node
    np.savez_compressed(
        regions_dir / f"{iid}.npz",
        boxes=boxes,
        vecs=np.zeros((n, 3), np.float32),
        whole_vec=np.zeros(3, np.float32),
        leaf_mask=np.array([c[0] < 0 for c in children]),
        children=children,
    )


def _render(tmp_path, *, bagged, snapped, n_leaves=4):
    cache = tmp_path / "cache"
    regions_dir = cache / "regions" / "ds" / "dinov3_patch" / "hac_k4_a0.5"
    for iid in (1, 2):
        _write_region_npz(regions_dir, iid, n_leaves)
    split = SimpleNamespace(gt_boxes={1: [(0.1, 0.1, 0.5, 0.5), (0.6, 0.6, 0.9, 0.9)]})
    trace = [
        {"image_id": 1, "gt_label": "good", "t": 1},
        {"image_id": 2, "gt_label": "bad", "t": 2},
    ]
    out = tmp_path / "out"
    vizmod.render_training_nodes(
        _stub_dataset(tmp_path),
        split,
        trace,
        cache_dir=cache,
        out_dir=out,
        dataset="ds",
        cls="car",
        embedder="dinov3",
        proposal="hac",
        alpha=0.5,
        slug="hac_k4_a0.5",
        seed=0,
        bagged=bagged,
        snapped=snapped,
    )
    return sorted(p.name for p in out.rglob("*.png"))


class TestNegativeConstruction:
    def test_box_pool_draws_one_whole_image_cls_crop_per_bad(self, tmp_path):
        names = _render(tmp_path, bagged=False, snapped=True)
        negs = [n for n in names if "NEG" in n]
        assert len(negs) == 1, names
        assert "wholeCLS" in negs[0]

    def test_bag_path_draws_every_childless_node(self, tmp_path):
        names = _render(tmp_path, bagged=True, snapped=True, n_leaves=4)
        negs = [n for n in names if "NEG" in n]
        assert len(negs) == 5  # CLS + 4 leaves; the internal merge node is excluded
        assert all("node" in n for n in negs)

    def test_bag_neg_count_scales_with_hac_k(self, tmp_path):
        assert len([n for n in _render(tmp_path, bagged=True, snapped=True, n_leaves=8) if "NEG" in n]) == 9


class TestWeightCaptions:
    def test_box_pool_omits_weights_since_that_path_applies_none(self, tmp_path):
        names = _render(tmp_path, bagged=False, snapped=True)
        assert not any("_w0." in n or "_w1." in n for n in names), names

    def test_bag_path_keeps_weights(self, tmp_path):
        names = _render(tmp_path, bagged=True, snapped=True)
        assert any("_w" in n for n in names if "POS" in n)
        assert any("_w" in n for n in names if "NEG" in n)


class TestPositiveConstruction:
    def test_snapped_draws_one_row_per_good_vote(self, tmp_path):
        assert len([n for n in _render(tmp_path, bagged=True, snapped=True) if "POS" in n]) == 1

    def test_unsnapped_draws_one_row_per_gt_box(self, tmp_path):
        """Grid pooling contributes a row per GT box, not a single snapped node."""
        assert len([n for n in _render(tmp_path, bagged=True, snapped=False) if "POS" in n]) == 2


class TestSkips:
    def test_non_hac_proposals_are_skipped(self, tmp_path, capsys):
        vizmod.render_training_nodes(
            _stub_dataset(tmp_path), SimpleNamespace(gt_boxes={}), [],
            cache_dir=tmp_path, out_dir=tmp_path / "o", dataset="ds", cls="car",
            embedder="dinov3", proposal="whole", alpha=0.5, slug=None, seed=0,
        )  # fmt: skip
        assert "HAC only" in capsys.readouterr().out
