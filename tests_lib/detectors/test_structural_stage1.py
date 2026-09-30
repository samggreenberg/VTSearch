"""The tiled Stage 1 and the verification cache (#3928, build step 3).

Synthetic features and a stub matcher, no SIFT: what is pinned is which pages
reach Stage 2 on a tiled dataset, how large the shortlist is, that the cache
verifies only what it has not seen, and that a dataset without tiles is
untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vtscore.media.structural import SIFT_DESCRIPTOR_DIM, MatchStats, StructuralFeatures
from vtscore.media.structural_tiles import fit_projection, raw_tiles, tile_vectors
from vtscore.training import structural_stage1 as s1
from vtscore.training.structural_similarity import (
    VerificationScorer,
    maybe_structural_rerank,
    maybe_structural_rerank_example,
    structural_rerank,
)

DIM = 16


def _features(seed: int, n: int = 600) -> StructuralFeatures:
    rng = np.random.default_rng(seed)
    kp = np.zeros((n, 4), dtype=np.float32)
    kp[:, :2] = rng.random((n, 2))
    desc = (rng.random((n, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
    return StructuralFeatures(keypoints=kp, descriptors=desc).compact()


@pytest.fixture(scope="module")
def projection():
    rows = np.concatenate([raw_tiles(_features(s))[0] for s in range(4)])
    return fit_projection(rows, DIM)


@pytest.fixture
def tiled(monkeypatch, projection):
    """Every page tiled with the small projection, which the queries also use."""
    import vtscore.media.structural_tiles as st

    monkeypatch.setattr(st, "load_tile_projection", lambda *_a, **_k: projection)
    monkeypatch.setattr(s1, "_cuda", lambda: False)
    s1._MATRIX_CACHE.clear()

    def make(n_pages: int) -> dict[int, dict[str, Any]]:
        snap = {}
        for mid in range(n_pages):
            feats = _features(100 + mid)
            snap[mid] = {
                "embedder": "sift_vlad_doc",
                "local_features": feats,
                "tile_vectors": tile_vectors(feats, projection),
            }
        return snap

    return make


class _CountingMatcher:
    """Scores a page by a fixed inlier count and counts every (template, page) it verifies."""

    def __init__(self, inliers: dict[int, int]):
        self.inliers = inliers  # id(page features) -> inliers
        self.calls = 0

    def detect_and_describe(self, image_gray: np.ndarray, *, max_features: int = 0) -> StructuralFeatures:
        raise NotImplementedError

    def verify(self, template: StructuralFeatures, candidate: StructuralFeatures) -> MatchStats:
        del template
        self.calls += 1
        n = self.inliers.get(id(candidate), 0)
        return MatchStats(inlier_count=n, model_ok=n > 0)


class TestTiledStage1:
    def test_pages_rank_by_their_best_tile_and_untiled_pages_come_last(self, tiled):
        snap = tiled(5)
        snap[5] = {"embedder": "sift_vlad_doc", "local_features": _features(99)}  # no tiles
        target = snap[3]["tile_vectors"].vectors[:1].astype(np.float32)
        out = s1.tiled_stage1(snap, target)
        assert out[0]["id"] == 3 and out[0]["score"] == pytest.approx(1.0, abs=1e-2)
        assert out[-1] == {"id": 5, "score": 0.0}
        assert [e["score"] for e in out[:5]] == sorted((e["score"] for e in out[:5]), reverse=True)

    def test_shortlist_size_follows_the_hardware(self, monkeypatch):
        monkeypatch.setattr(s1, "_cuda", lambda: True)
        assert s1.tiled_top_k(50_000) == s1.TILED_TOP_K == 1000
        assert s1.tiled_top_k(300) == 300
        monkeypatch.setattr(s1, "_cuda", lambda: False)
        assert s1.tiled_top_k(50_000) == s1.TILED_TOP_K_CPU == 500

    def test_the_matrix_is_stacked_once_per_loaded_page_set(self, tiled):
        snap = tiled(4)
        first = s1._tile_matrix(snap)
        assert s1._tile_matrix(dict(snap)) is first  # a new snapshot dict over the same pages
        snap[0] = dict(snap[0], tile_vectors=snap[1]["tile_vectors"])  # a reload swaps objects
        assert s1._tile_matrix(snap) is not first


class TestVerificationCache:
    def test_a_new_vote_verifies_only_its_own_template(self):
        pages = [(mid, _features(mid, n=30)) for mid in range(10)]
        matcher = _CountingMatcher({id(f): mid for mid, f in pages})
        cache = s1.VerificationCache()
        t1, t2 = ("good-1", _features(50, 30)), ("good-2", _features(51, 30))
        first = cache.best_many([t1], pages, matcher)
        assert matcher.calls == 10
        second = cache.best_many([t1, t2], pages, matcher)
        assert matcher.calls == 20  # only good-2 x 10 pages
        assert [s.inlier_count for s in second] == [s.inlier_count for s in first]

    def test_a_reloaded_page_is_verified_again(self):
        page = _features(1, 30)
        matcher = _CountingMatcher({id(page): 9})
        cache = s1.VerificationCache()
        cache.best_many([("t", page)], [(0, page)], matcher)
        reloaded = _features(1, 30)  # same content, new object: a dataset reload
        matcher.inliers[id(reloaded)] = 9
        cache.best_many([("t", page)], [(0, reloaded)], matcher)
        assert matcher.calls == 2

    def test_cached_and_uncached_reranks_agree(self):
        feats = {mid: _features(mid, 30) for mid in range(6)}
        matcher = _CountingMatcher({id(f): (mid * 7) % 20 for mid, f in feats.items()})
        snap = {mid: {"local_features": f} for mid, f in feats.items()}
        results = [{"id": mid, "score": 1.0 - mid / 10} for mid in feats]
        templates = [_features(40, 30), _features(41, 30)]
        plain = structural_rerank(results, snap, templates, VerificationScorer(), matcher, top_k=4)
        cached = structural_rerank(
            results,
            snap,
            templates,
            VerificationScorer(),
            matcher,
            top_k=4,
            template_keys=["a", "b"],
            cache=s1.VerificationCache(),
        )
        assert plain == cached


class TestChokepoint:
    def test_a_tiled_dataset_replaces_stage1_and_widens_the_shortlist(self, tiled, monkeypatch):
        snap = tiled(8)
        # The Good (page 0) boxes nothing, so its own tiles are the queries.
        matcher = _CountingMatcher({id(snap[mid]["local_features"]): 30 for mid in snap})
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)

        class Ctx:
            structural_verification_cache = None
            anchored_cut_cache = calibration_cache = line_ranking = None

        # The caller's Stage 1 ranks page 0 last; the tiled Stage 1 must put it first.
        results = [{"id": mid, "score": 1.0 - mid / 10} for mid in range(1, 8)] + [{"id": 0, "score": 0.0}]
        ctx = Ctx()
        out, thresh = maybe_structural_rerank(results, 0.77, snap, {0: None}, {}, ctx)
        assert thresh == 0.5
        assert len(out) == 8 and matcher.calls == 8  # all 8 verified: min(500, 8) is the shortlist
        assert isinstance(ctx.structural_verification_cache, s1.VerificationCache)
        # A retrain with the same Good verifies nothing new.
        maybe_structural_rerank(results, 0.77, snap, {0: None}, {}, ctx)
        assert matcher.calls == 8

    def test_a_dataset_without_tiles_keeps_the_callers_stage1_and_shortlist(self, monkeypatch):
        feats = {mid: _features(mid, 30) for mid in range(60)}
        snap = {mid: {"embedder": "sift_vlad", "local_features": f} for mid, f in feats.items()}
        matcher = _CountingMatcher({})
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)
        results = [{"id": mid, "score": 1.0} for mid in range(60)]
        maybe_structural_rerank(results, 0.3, snap, {0: None}, {})
        assert matcher.calls == 50  # DEFAULT_RERANK_TOP_K, one template

    def test_example_sort_on_a_tiled_dataset_ranks_by_the_crops_tiles(self, tiled, monkeypatch):
        snap = tiled(6)
        matcher = _CountingMatcher({})
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)
        crop = snap[4]["local_features"]  # the crop *is* page 4, so page 4's tiles match it best
        results = [{"id": mid, "score": 0.0} for mid in snap]
        out, _ = maybe_structural_rerank_example(results, 0.3, snap, crop)
        assert matcher.calls == 6
        assert {e["id"] for e in out} == set(snap)


class TestShortlistGrowth:
    """#4391's pre-registered K policies; ``"fixed"`` is what ships until the verdict."""

    def test_only_the_adaptive_policy_extends_and_only_while_the_tail_passes(self, monkeypatch):
        ids = list(range(3000))
        monkeypatch.setattr(s1, "EXTEND_WINDOW", 100)
        passing = {mid: 1.0 for mid in range(900, 1000)}  # the shortlist's last 100 all verify
        monkeypatch.setattr(s1, "K_POLICY", "fixed")
        assert not s1.should_extend(ids, passing, 1000)
        monkeypatch.setattr(s1, "K_POLICY", "adaptive")
        assert s1.should_extend(ids, passing, 1000)
        assert not s1.should_extend(ids, {mid: 1.0 for mid in range(900, 905)}, 1000)  # 5% < 10%
        monkeypatch.setattr(s1, "TILED_K_CAP", 1000)
        assert not s1.should_extend(ids, passing, 1000)  # at the cap

    def test_the_cap_policy_verifies_the_ceiling_from_the_start(self, monkeypatch):
        monkeypatch.setattr(s1, "K_POLICY", "cap")
        assert s1.tiled_top_k(50_000) == s1.TILED_K_CAP
        assert s1.tiled_top_k(2_000) == 2_000

    def test_adaptive_growth_verifies_more_blocks_on_a_tiled_dataset(self, tiled, monkeypatch):
        snap = tiled(12)
        # Every page verifies strongly, so every tail passes and the shortlist grows to the cap.
        matcher = _CountingMatcher({id(snap[mid]["local_features"]): 30 for mid in snap})
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)
        monkeypatch.setattr(s1, "_cuda", lambda: False)
        monkeypatch.setattr(s1, "TILED_TOP_K_CPU", 4)
        monkeypatch.setattr(s1, "EXTEND_STEP", 4)
        monkeypatch.setattr(s1, "EXTEND_WINDOW", 2)
        monkeypatch.setattr(s1, "TILED_K_CAP", 10)
        results = [{"id": mid, "score": 0.0} for mid in snap]
        monkeypatch.setattr(s1, "K_POLICY", "fixed")
        maybe_structural_rerank(results, 0.5, snap, {0: None}, {})
        assert s1.LAST_TOP_K == 4 and matcher.calls == 4
        monkeypatch.setattr(s1, "K_POLICY", "adaptive")
        matcher.calls = 0
        maybe_structural_rerank(results, 0.5, snap, {0: None}, {})
        assert s1.LAST_TOP_K == 10 and matcher.calls == 10  # 4 -> 8 -> 10 (cap); no page verified twice
