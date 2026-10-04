"""The tiled Stage 1 and the verification cache (#3928, build step 3).

Synthetic features and a stub matcher, no SIFT: what is pinned is which pages
reach Stage 2 on a tiled dataset, how large the shortlist is, that the cache
verifies only what it has not seen, and that a dataset without tiles is
untouched.
"""

from __future__ import annotations

import dataclasses
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
        # A tight fit, so #4440's geometry cuts (applied before the first Bad) do not demote it.
        return MatchStats(inlier_count=n, model_ok=n > 0, inlier_ratio=0.9, median_reproj_error=0.001)


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
        assert s1.tiled_top_k(50_000) == s1.TILED_TOP_K == 2000
        assert s1.tiled_top_k(300) == 300
        monkeypatch.setattr(s1, "_cuda", lambda: False)
        assert s1.tiled_top_k(50_000) == s1.TILED_TOP_K_CPU == 1000

    def test_gpu_scoring_frees_the_cache_before_measuring_and_says_when_it_falls_back(self, monkeypatch, caplog):
        # #4170: memory PyTorch had cached did not count as free, so a 32 GB card fell back
        # to the CPU from the second matrix on, without a word.
        import sys
        from types import SimpleNamespace

        calls: list[str] = []
        cuda = SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty"),
            mem_get_info=lambda: (calls.append("measure"), (1024, 2048))[1],
        )
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
        s1._GPU_CACHE.clear()
        matrix = np.zeros((1000, DIM), dtype=np.float16)
        with caplog.at_level("WARNING", logger=s1._log.name):
            out = s1._gpu_page_scores(matrix, np.array([0, 500]), np.zeros((1, DIM), dtype=np.float32))
        assert out is None
        assert calls == ["empty", "measure"]
        assert "scoring on the CPU" in caplog.text

    def test_a_seeded_crop_queries_whole_and_a_page_by_its_tiles(self, tiled):
        # #4170: a crop's tiles are fragments of the mark, so a seeded crop is one whole-VLAD query.
        snap = tiled(2)
        page = s1.vote_queries({0: None}, snap, {})
        assert page is not None and page.shape[0] == snap[0]["tile_vectors"].count > 1
        snap[0] = dict(snap[0], seeded_example=True)
        crop = s1.vote_queries({0: None}, snap, {})
        assert crop is not None and crop.shape == (1, DIM)

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

    def test_a_pruned_template_is_verified_only_where_its_parent_passes(self):
        # #4432: pages 0-9 fit the unpruned template with 0-9 inliers, so only 8 and 9 clear the gate.
        pages = [(mid, _features(mid, n=30)) for mid in range(10)]
        matcher = _CountingMatcher({id(f): mid for mid, f in pages})
        cache = s1.VerificationCache()
        parent = ("good", _features(50, 30))
        cache.best_many([parent], pages, matcher)
        assert matcher.calls == 10
        pruned = (("good", "pruned"), _features(51, 20))
        out = cache.best_many([pruned], pages, matcher, parents={pruned[0]: parent})
        assert matcher.calls == 12  # the parent's fits were cached; the pruned one ran on pages 8 and 9
        assert [s.inlier_count for s in out] == [0] * 8 + [8, 9]
        assert not any(s.model_ok for s in out[:8])

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


class TestBadCeiling:
    """#4367 (pre-registered R1): on a tiled dataset the line sits above the best fit any Bad reached."""

    def _setup(self, tiled, monkeypatch, inliers_by_page):
        snap = tiled(6)
        matcher = _CountingMatcher({id(snap[mid]["local_features"]): n for mid, n in inliers_by_page.items()})
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)

        class Ctx:
            structural_verification_cache = None
            anchored_cut_cache = calibration_cache = line_ranking = None

        return snap, Ctx()

    def test_the_line_rises_above_the_best_fitting_bad(self, tiled, monkeypatch):
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 60, 2: 40, 3: 30, 4: 12, 5: 3})
        results = [{"id": mid, "score": 0.0} for mid in snap]
        # Page 3 is a Bad that fits with 30 inliers: only pages with >= 31 may be returned.
        out, thresh = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={3: None})
        assert thresh == pytest.approx(31 / (31 + 8))
        assert {e["id"] for e in out if e["score"] >= thresh} == {0, 1, 2}
        # Without the Bad it is the shipped 8-inlier gate.
        _out, gate = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx)
        assert gate == 0.5

    def test_a_page_at_exactly_the_ceiling_plus_one_is_returned(self, tiled, monkeypatch):
        # #4464: 11 / 19 = 0.578947..., which a 4-decimal score rounded down below the unrounded line.
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 11, 2: 10, 3: 10, 4: 9, 5: 3})
        results = [{"id": mid, "score": 0.0} for mid in snap]
        out, thresh = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={3: None})
        assert {e["id"] for e in out if e["score"] >= thresh} == {0, 1}

    def test_a_bad_that_does_not_fit_leaves_the_gate(self, tiled, monkeypatch):
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 60, 2: 40, 3: 30, 4: 12, 5: 3})
        results = [{"id": mid, "score": 0.0} for mid in snap]
        _out, thresh = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={5: None})
        assert thresh == 0.5  # 3 inliers is below the gate: max(8, 3 + 1) = 8

    def test_an_untiled_dataset_keeps_the_gate_whatever_the_bads(self, monkeypatch):
        feats = {mid: _features(mid, 30) for mid in range(6)}
        snap = {mid: {"embedder": "sift_vlad", "local_features": f} for mid, f in feats.items()}
        matcher = _CountingMatcher({id(f): 40 for f in feats.values()})
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)
        results = [{"id": mid, "score": 0.0} for mid in snap]
        _out, thresh = maybe_structural_rerank(results, 0.3, snap, {0: None}, {}, bad_votes={3: None})
        assert thresh == 0.5


class TestRecallEnd:
    """#4458: at beta 4 the returned set is the beta-1 set plus every verified page above the recall floor."""

    def _setup(self, tiled, monkeypatch, inliers_by_page, loose=()):
        snap = tiled(6)
        by_id = {id(snap[mid]["local_features"]): n for mid, n in inliers_by_page.items()}
        loose_ids = {id(snap[mid]["local_features"]) for mid in loose}

        class Matcher(_CountingMatcher):
            def verify(self, template, candidate):
                stats = super().verify(template, candidate)
                if id(candidate) in loose_ids:
                    return dataclasses.replace(stats, inlier_ratio=0.3)
                return stats

        matcher = Matcher(by_id)
        monkeypatch.setattr("vtscore.training.structural_similarity._resolve_matcher", lambda _snap: matcher)

        class Ctx:
            structural_verification_cache = None
            anchored_cut_cache = calibration_cache = line_ranking = None

        return snap, Ctx()

    @staticmethod
    def _returned(out, thresh):
        return {e["id"] for e in out if e["score"] >= thresh}

    def test_the_line_drops_to_the_floor_below_the_bad_ceiling(self, tiled, monkeypatch):
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 60, 2: 40, 3: 30, 4: 12, 5: 3})
        results = [{"id": mid, "score": 0.0} for mid in snap]
        out1, t1 = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={3: None}, beta=1.0)
        out4, t4 = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={3: None}, beta=4.0)
        # One Good: no leave-one-out median, so the floor is 10; page 4 (12 inliers) joins at beta 4.
        assert t1 == pytest.approx(31 / 39, abs=1e-6) and t4 == pytest.approx(10 / 18, abs=1e-6)
        assert self._returned(out1, t1) == {0, 1, 2}
        assert self._returned(out1, t1) <= self._returned(out4, t4) and 4 in self._returned(out4, t4)

    def test_the_floor_follows_the_goods_own_fits(self, tiled, monkeypatch):
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 60, 2: 40, 3: 30, 4: 12, 5: 3})
        results = [{"id": mid, "score": 0.0} for mid in snap]
        # Goods 0 and 1 fit each other with 90 and 60 inliers: median 75, floor ceil(18.75) = 19.
        out, t = maybe_structural_rerank(results, 0.5, snap, {0: None, 1: None}, {}, ctx, bad_votes={3: None}, beta=4.0)
        assert t == pytest.approx(19 / 27, abs=1e-6)
        assert 4 not in self._returned(out, t) and 2 in self._returned(out, t)

    def test_before_a_bad_a_loose_fit_above_the_floor_is_kept(self, tiled, monkeypatch):
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 3, 2: 3, 3: 3, 4: 12, 5: 9}, loose=(4, 5))
        results = [{"id": mid, "score": 0.0} for mid in snap]
        out1, t1 = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, beta=1.0)
        out4, t4 = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, beta=4.0)
        assert t1 == t4 == 0.5
        assert self._returned(out1, t1) == {0}  # H1: loose fits fall below the line
        assert self._returned(out4, t4) == {0, 4}  # 12 >= the floor of 10; page 5 (9, loose) stays out

    def test_no_balance_is_the_shipped_line(self, tiled, monkeypatch):
        snap, ctx = self._setup(tiled, monkeypatch, {0: 90, 1: 60, 2: 40, 3: 30, 4: 12, 5: 3})
        results = [{"id": mid, "score": 0.0} for mid in snap]
        _o, t_none = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={3: None})
        _o, t_half = maybe_structural_rerank(results, 0.5, snap, {0: None}, {}, ctx, bad_votes={3: None}, beta=0.25)
        assert t_none == t_half == pytest.approx(31 / 39, abs=1e-6)


class TestStoplist:
    """#4170 / #4180 (pre-registered arms): Good templates drop what a Bad page also matches."""

    def _shared(self):
        """Template 0 = 20 'mark' descriptors + 20 'glyph' descriptors; the Bad page holds only the glyphs."""
        rng = np.random.default_rng(7)
        mark = (rng.random((20, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        glyph = (rng.random((20, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        kp = np.zeros((40, 4), dtype=np.float32)
        kp[:, :2] = rng.random((40, 2))
        tpl = StructuralFeatures(keypoints=kp, descriptors=np.vstack([mark, glyph]))
        noise = (rng.random((200, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        bad_kp = np.zeros((220, 4), dtype=np.float32)
        bad = StructuralFeatures(keypoints=bad_kp, descriptors=np.vstack([glyph + 1.0, noise]))
        return tpl, bad, mark

    def test_descriptors_a_bad_matches_are_dropped(self, monkeypatch):
        import vtscore.training.structural_similarity as ss

        tpl, bad, _ = self._shared()
        monkeypatch.setattr(ss, "STOPLIST_POLICY", "all")
        out, tags = ss._stoplist(
            [("g", tpl)], {"b": None}, {"b": {"local_features": bad}}, _CountingMatcher({}), None, {}
        )
        assert out[0][1].count == 20  # the 20 glyphs are gone, the mark stays
        assert "g" in tags

    def test_a_descriptor_another_good_also_has_is_kept(self, monkeypatch):
        import vtscore.training.structural_similarity as ss

        tpl, bad, mark = self._shared()
        rng = np.random.default_rng(9)
        # Another Good page carries the glyphs too (the mark inside a lockup, #4180): they stay.
        other = StructuralFeatures(
            keypoints=np.zeros((40, 4), dtype=np.float32),
            descriptors=np.vstack(
                [tpl.descriptors_f32()[20:] - 1.0, (rng.random((20, SIFT_DESCRIPTOR_DIM)) * 255)]
            ).astype(np.float32),
        )
        monkeypatch.setattr(ss, "STOPLIST_POLICY", "all")
        snap = {"b": {"local_features": bad}, "g": {"local_features": tpl}, "h": {"local_features": other}}
        out, _tags = ss._stoplist([("g", tpl), ("h", other)], {"b": None}, snap, _CountingMatcher({}), None, {})
        assert out[0][1].count == 40

    def test_shipped_policy_is_off(self):
        import vtscore.training.structural_similarity as ss

        assert ss.STOPLIST_POLICY == "off"


class TestEarlyGeometryLine:
    """#4440 (pre-registered H1): before the first Bad vote, the returned set needs a tight fit."""

    def test_a_loose_fit_scores_half_and_keeps_its_order(self):
        from vtscore.training.structural_similarity import VerificationScorer

        scorer = VerificationScorer(ratio_min=0.75, reproj_max=0.005)
        tight = MatchStats(inlier_count=40, model_ok=True, inlier_ratio=0.9, median_reproj_error=0.001)
        loose = MatchStats(inlier_count=40, model_ok=True, inlier_ratio=0.4, median_reproj_error=0.001)
        looser = MatchStats(inlier_count=60, model_ok=True, inlier_ratio=0.9, median_reproj_error=0.009)
        assert scorer.score(tight) == pytest.approx(40 / 48)
        assert scorer.score(loose) == pytest.approx(40 / 48 / 2)
        assert scorer.score(loose) < 0.5  # below the line
        assert scorer.score(loose) < scorer.score(looser) < 0.5  # failing fits keep their inlier order

    def test_the_cuts_apply_only_until_a_bad_vote_exists(self, tiled):
        from vtscore.training.structural_similarity import _line_scorer

        snap = tiled(3)
        early = _line_scorer(True, {}, snap)
        assert early.ratio_min is not None and early.reproj_max is not None
        later = _line_scorer(True, {1: None}, snap)  # page 1 is a Bad with features
        assert later.ratio_min is None and later.reproj_max is None
        assert _line_scorer(False, {}, snap).ratio_min is None  # photos: never
