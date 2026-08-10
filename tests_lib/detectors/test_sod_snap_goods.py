"""``--snap-goods`` separates good-vote geometry from negative construction.

``--region-voting`` used to drive two independent halves of the label construction at
once: the *positive* side (a Good vote's box snaps to its best-IoU HAC node, set on the
region source) and the *negative* side (a Bad vote floods CLS+leaves into a bag, read off
``RegionCurveInputs`` by ``_train_pool_head``). That made the middle arm -- snapped goods
with one whole-image CLS row per Bad -- unreachable.

These tests pin the split: the default still follows ``--region-voting`` byte-for-byte, the
cache slug keys on snapping (which is what actually changes the cached exemplars), and the
HAC/patch-embedder gate applies to both halves.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))

from sweep import _proposal_slug, _resolve_snap_goods  # noqa: E402


def _args(*, region_voting=True, snap_goods=None, **kw):
    base = dict(
        region_voting=region_voting,
        snap_goods=snap_goods,
        scales=(1.0,),
        overlap=0.5,
        min_window=48,
        hac_alpha_default=0.5,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestResolveSnapGoods:
    @pytest.mark.parametrize("rv", [True, False])
    def test_unset_follows_region_voting(self, rv):
        """Backwards compatibility: every pre-split command must behave identically."""
        assert _resolve_snap_goods(_args(region_voting=rv), "hac", "dinov3") is rv

    def test_explicit_true_overrides_region_voting_off(self):
        """The new middle arm: snapped goods, whole-image CLS bads."""
        args = _args(region_voting=False, snap_goods=True)
        assert _resolve_snap_goods(args, "hac", "dinov3") is True

    def test_explicit_false_overrides_region_voting_on(self):
        args = _args(region_voting=True, snap_goods=False)
        assert _resolve_snap_goods(args, "hac", "dinov3") is False

    @pytest.mark.parametrize("proposal", ["whole", "sliding", "dino"])
    def test_no_op_off_the_hac_proposal(self, proposal):
        assert _resolve_snap_goods(_args(snap_goods=True), proposal, "dinov3") is False

    @pytest.mark.parametrize("embedder", ["siglip", "siglip2", "clip"])
    def test_no_op_without_a_patch_embedder(self, embedder):
        assert _resolve_snap_goods(_args(snap_goods=True), "hac", embedder) is False

    @pytest.mark.parametrize("embedder", ["dinov2", "dinov3"])
    def test_both_patch_embedders_are_eligible(self, embedder):
        assert _resolve_snap_goods(_args(snap_goods=True), "hac", embedder) is True


class TestSlugKeysOnSnapping:
    def test_snapped_and_grid_pooled_exemplars_get_different_slugs(self):
        """They are different cached bytes; sharing a dir would serve the wrong exemplars."""
        snapped = _proposal_slug("hac", _args(), 0.5, snap_goods=True)
        grid = _proposal_slug("hac", _args(), 0.5, snap_goods=False)
        assert snapped != grid
        assert "_rv" in snapped
        assert "_rv" not in grid

    def test_pre_split_slugs_are_unchanged(self):
        """Snapping defaults to region voting, so historical caches keep their exact key."""
        assert _proposal_slug("hac", _args(), 0.5, snap_goods=True) == "hac_rv_k12_a0.5"
        assert _proposal_slug("hac", _args(), 0.5, snap_goods=False) == "hac_k12_a0.5"

    def test_middle_arm_shares_the_snapped_cache(self):
        """snap+flood and snap+CLS differ only in training, not in cached bytes, so they
        must reuse one exemplar/region cache rather than re-embedding."""
        full_rv = _proposal_slug("hac", _args(region_voting=True), 0.5, snap_goods=True)
        middle = _proposal_slug("hac", _args(region_voting=False, snap_goods=True), 0.5, snap_goods=True)
        assert full_rv == middle

    @pytest.mark.parametrize("proposal", ["whole", "sliding", "dino"])
    def test_snapping_does_not_leak_into_other_proposal_slugs(self, proposal):
        on = _proposal_slug(proposal, _args(), 0.5, snap_goods=True)
        off = _proposal_slug(proposal, _args(), 0.5, snap_goods=False)
        assert on == off
