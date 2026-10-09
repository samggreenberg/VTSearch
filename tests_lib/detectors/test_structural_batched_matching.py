"""Tests for batched structural matching and the detection-resolution cap.

Both land the same issue (#3900, "sift-vlad CPU vs GPU"): the only part of the
structural pipeline a GPU can reach is Stage-2 descriptor matching, so that is
re-expressed as one batched ``torch`` distance computation
(:func:`~vtscore.media.structural.ratio_test_matches`), while the dominant
Stage-1 cost - SIFT detection, which has no GPU path at all with the shipped
OpenCV wheel - is cut by bounding the resolution it runs at
(:func:`~vtscore.media.structural.cap_detect_resolution`).

The contract both must hold is *equivalence*: the batched path returns what the
per-pair ``cv2.BFMatcher`` loop returned, and a matcher without ``verify_many``
still works unchanged.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from vtscore.media import structural
from vtscore.media.structural import (
    DEFAULT_MAX_FEATURES,
    SIFT_DESCRIPTOR_DIM,
    MatchStats,
    SiftMatcher,
    StructuralFeatures,
    cap_detect_resolution,
    ratio_test_matches,
)
from vtscore.training.structural_similarity import best_match_stats, best_match_stats_many

_LOWE = 0.75


def _textured_image(seed: int = 0, size: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((size, size)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _structured_image(seed: int = 0, size: int = 600) -> np.ndarray:
    """An image whose content survives a downscale, unlike pure noise.

    :func:`_textured_image` is band-limited noise: its entire signal lives in the
    highest frequencies, so an area-averaged downsample destroys it and features
    detected at two different resolutions have literally nothing in common.  Real
    imagery is not like that, so the cross-resolution tests need drawn geometry -
    edges and corners that are still there at a quarter of the size.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40, dtype=np.uint8)
    for _ in range(14):
        p = rng.integers(20, size - 20, size=4)
        cv2.rectangle(
            img,
            (int(p[0]), int(p[1])),
            (int(p[2]), int(p[3])),
            int(rng.integers(90, 255)),
            int(rng.integers(2, 6)),
        )
    for _ in range(10):
        c = rng.integers(40, size - 40, size=2)
        cv2.circle(img, (int(c[0]), int(c[1])), int(rng.integers(20, 70)), int(rng.integers(90, 255)), 3)
    for _ in range(12):
        p = rng.integers(0, size, size=4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), int(rng.integers(90, 255)), 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _descs(n: int, seed: int, dim: int = SIFT_DESCRIPTOR_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, dim)) * 255).astype(np.float32)


def _reference_matches(t_desc: np.ndarray, c_desc: np.ndarray) -> set[tuple[int, int]]:
    """The correspondences ``cv2.BFMatcher`` + the Lowe ratio loop produce."""
    if t_desc.shape[0] < 2 or c_desc.shape[0] < 2:
        return set()
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(t_desc, c_desc, k=2)
    return {
        (m.queryIdx, m.trainIdx) for pair in knn if len(pair) >= 2 for m, n in [pair] if m.distance < _LOWE * n.distance
    }


def _as_set(pair: tuple[np.ndarray, np.ndarray]) -> set[tuple[int, int]]:
    return set(zip(pair[0].tolist(), pair[1].tolist()))


# --------------------------------------------------------------------------
# ratio_test_matches
# --------------------------------------------------------------------------


class TestRatioTestMatches:
    def test_agrees_with_bfmatcher_on_real_sift_descriptors(self) -> None:
        """The batched path reproduces the cv2 loop it replaces."""
        matcher = SiftMatcher()
        feats = [matcher.detect_and_describe(_textured_image(s), max_features=200) for s in range(4)]
        template = feats[0].descriptors_f32()
        batched = ratio_test_matches(template, [f.descriptors_f32() for f in feats], ratio=_LOWE)
        for got, feat in zip(batched, feats):
            assert _as_set(got) == _reference_matches(template, feat.descriptors_f32())

    def test_ragged_candidate_sizes_are_padded_without_contaminating_matches(self) -> None:
        """Short candidates are padded into the batch; the padding must not win a match."""
        template = _descs(30, seed=1)
        candidates = [_descs(n, seed=10 + n) for n in (2, 5, 40, 3, 25)]
        batched = ratio_test_matches(template, candidates, ratio=_LOWE)
        assert len(batched) == len(candidates)
        for got, cand in zip(batched, candidates):
            assert _as_set(got) == _reference_matches(template, cand)
            # Every returned candidate index must address a real descriptor.
            assert got[1].size == 0 or int(got[1].max()) < cand.shape[0]

    @pytest.mark.parametrize("n_template", [0, 1])
    def test_degenerate_template_yields_no_matches(self, n_template: int) -> None:
        """A template without a second-nearest neighbour cannot ratio-test."""
        got = ratio_test_matches(_descs(n_template, seed=2), [_descs(10, seed=3)], ratio=_LOWE)
        assert len(got) == 1
        assert got[0][0].size == 0

    def test_candidate_with_one_descriptor_yields_no_matches(self) -> None:
        got = ratio_test_matches(_descs(10, seed=4), [_descs(1, seed=5), _descs(10, seed=6)], ratio=_LOWE)
        assert got[0][0].size == 0
        assert _as_set(got[1]) == _reference_matches(_descs(10, seed=4), _descs(10, seed=6))

    def test_empty_candidate_list(self) -> None:
        assert ratio_test_matches(_descs(10, seed=7), [], ratio=_LOWE) == []

    def test_chunking_does_not_change_the_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shortlist split across chunks yields the same matches as one batch."""
        import vtscore.media.structural as structural

        template = _descs(40, seed=8)
        candidates = [_descs(20 + i, seed=100 + i) for i in range(7)]
        whole = ratio_test_matches(template, candidates, ratio=_LOWE)
        monkeypatch.setattr(structural, "_MATCH_CHUNK_ELEMENTS", 1)  # force one candidate per chunk
        chunked = ratio_test_matches(template, candidates, ratio=_LOWE)
        assert [_as_set(p) for p in chunked] == [_as_set(p) for p in whole]


class TestDeviceDescriptorCache:
    """#4469: stored candidate descriptors stay on the device across templates, with the same matches."""

    @pytest.fixture(autouse=True)
    def _fresh(self):
        from vtscore.media import structural

        structural.release_device_descriptors()
        yield
        structural.release_device_descriptors()

    def test_uint8_and_cached_candidates_match_like_float32(self) -> None:
        template = _descs(40, seed=1)
        cands = [_descs(n, seed=10 + n).astype(np.uint8) for n in (30, 55, 2, 80)]
        plain = ratio_test_matches(template, [c.astype(np.float32) for c in cands], ratio=_LOWE)
        cached = ratio_test_matches(template, cands, ratio=_LOWE, cache=True)
        again = ratio_test_matches(template, cands, ratio=_LOWE, cache=True)  # served from the cache
        assert [_as_set(p) for p in cached] == [_as_set(p) for p in plain] == [_as_set(p) for p in again]

    def test_a_reused_id_never_returns_another_arrays_descriptors(self) -> None:
        import weakref

        import torch

        from vtscore.media import structural

        fresh = np.full((3, 4), 7, dtype=np.uint8)
        dead = np.zeros((3, 4), dtype=np.uint8)
        stale = torch.zeros((3, 4), dtype=torch.uint8)
        # A cache entry left by a freed array whose id the new array now has.
        structural._DEVICE_DESCRIPTORS[id(fresh)] = (weakref.ref(dead), stale)
        structural._DEVICE_DESCRIPTOR_BYTES[0] = stale.nbytes
        out = structural._device_descriptors(fresh, "cpu", cache=True)
        assert out.dtype == torch.float32
        assert torch.equal(out, torch.full((3, 4), 7.0))

    def test_the_least_recently_used_pages_go_past_the_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from vtscore.media import structural

        monkeypatch.setattr(structural, "DEVICE_DESCRIPTOR_LIMIT_BYTES", 2 * 100 * 128)
        pages = [np.zeros((100, 128), dtype=np.uint8) for _ in range(3)]
        for p in pages:
            structural._device_descriptors(p, "cpu", cache=True)
        kept = [k for k in structural._DEVICE_DESCRIPTORS]
        assert kept == [id(pages[1]), id(pages[2])]
        assert structural._DEVICE_DESCRIPTOR_BYTES[0] == 2 * 100 * 128

    def test_verify_many_on_compact_features_matches_float_features(self) -> None:
        matcher = SiftMatcher()
        feats = [matcher.detect_and_describe(_textured_image(s), max_features=200) for s in range(5)]
        compact = [f.compact() for f in feats]
        template = compact[0]
        a = matcher.verify_many(template, compact)
        b = [matcher.verify(template, c) for c in compact]
        assert [(x.inlier_count, x.tentative_count, x.model_ok) for x in a] == [
            (y.inlier_count, y.tentative_count, y.model_ok) for y in b
        ]


# --------------------------------------------------------------------------
# SiftMatcher.verify_many
# --------------------------------------------------------------------------


class TestVerifyMany:
    def test_matches_per_pair_verify(self) -> None:
        matcher = SiftMatcher()
        feats = [matcher.detect_and_describe(_textured_image(s), max_features=200) for s in range(5)]
        template = feats[0]
        batched = matcher.verify_many(template, feats)
        single = [matcher.verify(template, c) for c in feats]
        assert len(batched) == len(single)
        for b, s in zip(batched, single):
            assert (b.inlier_count, b.tentative_count, b.model_ok) == (s.inlier_count, s.tentative_count, s.model_ok)
            assert b.inlier_ratio == pytest.approx(s.inlier_ratio)

    def test_empty_candidates(self) -> None:
        matcher = SiftMatcher()
        feats = matcher.detect_and_describe(_textured_image(0), max_features=100)
        assert matcher.verify_many(feats, []) == []

    def test_degenerate_template_returns_one_empty_stat_per_candidate(self) -> None:
        matcher = SiftMatcher()
        empty = StructuralFeatures(
            keypoints=np.zeros((0, 4), dtype=np.float32),
            descriptors=np.zeros((0, SIFT_DESCRIPTOR_DIM), dtype=np.float32),
        )
        cands = [matcher.detect_and_describe(_textured_image(s), max_features=100) for s in range(3)]
        got = matcher.verify_many(empty, cands)
        assert len(got) == 3
        assert all(s.inlier_count == 0 and not s.model_ok for s in got)


# --------------------------------------------------------------------------
# SiftMatcher._fit_similarity_many: the RANSAC fits on a thread pool (#4516)
# --------------------------------------------------------------------------


def _correspondence_jobs(n_pairs: int, seed: int = 0) -> tuple[np.ndarray, list]:
    """A template's keypoints and *n_pairs* ``(c_kp, t_idx, c_idx)`` fit jobs against it.

    Each pair plants a similarity-mapped inlier set among random outliers, at
    0-60 correspondences like a Stage-2 shortlist's, so a batch holds fits that
    pass the gate, fits that fail it, and pairs too small to fit at all.
    """
    rng = np.random.default_rng(seed)
    t_kp = rng.random((200, 4)).astype(np.float32)
    jobs = []
    for _ in range(n_pairs):
        n = int(rng.integers(0, 60))
        t_idx = rng.choice(200, size=n, replace=False).astype(np.int64)
        inliers = int(rng.integers(0, n + 1))
        theta, s = rng.uniform(-np.pi, np.pi), rng.uniform(0.05, 2.0)
        rot = s * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = rng.random((n, 2))
        xy[:inliers] = t_kp[t_idx[:inliers], :2] @ rot.T + rng.random(2) + rng.normal(0, 0.002, (inliers, 2))
        c_kp = np.zeros((n, 4), dtype=np.float16)
        c_kp[:, :2] = xy
        jobs.append((c_kp, t_idx, np.arange(n, dtype=np.int64)))
    return t_kp, jobs


def _fit_before_4516(t_kp: np.ndarray, c_kp: np.ndarray, t_idx: np.ndarray, c_idx: np.ndarray) -> MatchStats:
    """``SiftMatcher._fit_similarity`` as it was before #4516, numpy calls and all."""
    tentative = int(t_idx.shape[0])
    if tentative < 2:
        return MatchStats(tentative_count=tentative)
    src = np.ascontiguousarray(t_kp[t_idx, :2], dtype=np.float32)
    dst = np.ascontiguousarray(c_kp[c_idx, :2], dtype=np.float32)
    model, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.02, maxIters=2000, confidence=0.99, refineIters=10
    )
    if model is None or inlier_mask is None or not np.isfinite(model).all():
        return MatchStats(tentative_count=tentative)
    mask = inlier_mask.ravel().astype(bool)
    inlier_count = int(mask.sum())
    scale = float(np.hypot(float(model[0, 0]), float(model[1, 0])))
    reflection = bool(np.linalg.det(np.asarray(model[:, :2], dtype=np.float64)) < 0)
    model_ok = inlier_count >= 4 and 0.03 <= scale <= 10.0 and not reflection
    mean_err = median_err = spread = 0.0
    box = None
    if inlier_count:
        src_in, dst_in = src[mask], dst[mask]
        proj = (src_in @ model[:, :2].T) + model[:, 2]
        errs = np.linalg.norm(proj - dst_in, axis=1)
        mean_err = float(errs.mean())
        median_err = float(np.median(errs))
        centroid = dst_in.mean(axis=0)
        spread = float(np.sqrt(((dst_in - centroid) ** 2).sum(axis=1).mean()))
        x0, y0 = dst_in.min(axis=0)
        x1, y1 = dst_in.max(axis=0)
        box = (float(x0), float(y0), float(x1), float(y1))
    return MatchStats(
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / tentative,
        tentative_count=tentative,
        mean_reproj_error=mean_err,
        median_reproj_error=median_err,
        scale=scale,
        reflection=reflection,
        inlier_spread=spread,
        model_ok=model_ok,
        inlier_box=box if model_ok else None,
    )


def _spy_threads(monkeypatch: pytest.MonkeyPatch, matcher: SiftMatcher) -> set[str]:
    """Record the name of every thread *matcher* runs a RANSAC call on."""
    names: set[str] = set()
    ransac = matcher._ransac

    def spy(*args):
        names.add(threading.current_thread().name)
        return ransac(*args)

    monkeypatch.setattr(matcher, "_ransac", spy)
    return names


class TestFitSimilarityMany:
    def test_stats_are_bit_identical_to_the_numpy_calls_they_replaced(self) -> None:
        """The cheaper median/mean/norm/det are the same float operations: every field is equal, not approximately."""
        t_kp, jobs = _correspondence_jobs(300)
        got = [SiftMatcher()._fit_similarity(t_kp, *job) for job in jobs]
        want = [_fit_before_4516(t_kp, *job) for job in jobs]
        assert got == want
        # The batch exercises every branch: no fit, a failed fit, a passing fit, odd and even inlier counts.
        assert {s.tentative_count < 2 for s in want} == {True, False}
        assert {s.model_ok for s in want} == {True, False}
        assert {s.inlier_count % 2 for s in want if s.inlier_count} == {0, 1}

    def test_pool_returns_the_loop_s_fits_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(structural, "_ransac_workers", lambda: 4)
        matcher = SiftMatcher()
        t_kp, jobs = _correspondence_jobs(3 * structural._RANSAC_POOL_MIN_PAIRS + 7, seed=1)
        loop = [matcher._fit_similarity(t_kp, *job) for job in jobs]
        threads = _spy_threads(monkeypatch, matcher)
        assert matcher._fit_similarity_many(t_kp, jobs) == loop
        assert threads and all(name.startswith("ransac") for name in threads)

    def test_small_batch_fits_inline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(structural, "_ransac_workers", lambda: 4)
        matcher = SiftMatcher()
        threads = _spy_threads(monkeypatch, matcher)
        t_kp, jobs = _correspondence_jobs(structural._RANSAC_POOL_MIN_PAIRS - 1, seed=2)
        assert len(matcher._fit_similarity_many(t_kp, jobs)) == len(jobs)
        assert threads == {threading.current_thread().name}

    @pytest.mark.parametrize(("cpus", "workers"), [(64, 4), (4, 4), (3, 3), (1, 1)])
    def test_workers_follow_the_affinity_mask_up_to_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, cpus: int, workers: int
    ) -> None:
        monkeypatch.setattr(structural.os, "sched_getaffinity", lambda _pid: set(range(cpus)), raising=False)
        assert structural._ransac_workers() == workers

    def test_workers_without_an_affinity_mask_use_the_cpu_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(structural.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(structural.os, "cpu_count", lambda: 2)
        assert structural._ransac_workers() == 2


# --------------------------------------------------------------------------
# best_match_stats_many
# --------------------------------------------------------------------------


class _PairOnlyMatcher:
    """A ``StructuralMatcher`` with no ``verify_many`` - the third-party shape."""

    def __init__(self) -> None:
        self._inner = SiftMatcher()

    def detect_and_describe(self, image_gray, *, max_features: int = DEFAULT_MAX_FEATURES):
        return self._inner.detect_and_describe(image_gray, max_features=max_features)

    def verify(self, template, candidate):
        return self._inner.verify(template, candidate)


class TestBestMatchStatsMany:
    def _feats(self, n: int = 5) -> list[StructuralFeatures]:
        m = SiftMatcher()
        return [m.detect_and_describe(_textured_image(s), max_features=200) for s in range(n)]

    def test_equals_per_candidate_best_match_stats(self) -> None:
        matcher = SiftMatcher()
        feats = self._feats()
        templates = [(i, f) for i, f in enumerate(feats[:2])]
        got = best_match_stats_many(templates, feats, matcher)
        want = [best_match_stats([t for _, t in templates], c, matcher) for c in feats]
        assert [(g.inlier_count, g.model_ok) for g in got] == [(w.inlier_count, w.model_ok) for w in want]

    def test_matcher_without_verify_many_falls_back_to_the_pair_loop(self) -> None:
        """The protocol never required ``verify_many``; a matcher lacking it still works."""
        feats = self._feats()
        templates = [(i, f) for i, f in enumerate(feats[:2])]
        pair_only = _PairOnlyMatcher()
        assert not hasattr(pair_only, "verify_many")
        got = best_match_stats_many(templates, feats, pair_only)
        want = best_match_stats_many(templates, feats, SiftMatcher())
        assert [(g.inlier_count, g.model_ok) for g in got] == [(w.inlier_count, w.model_ok) for w in want]

    def test_skip_holds_out_a_candidates_own_template(self) -> None:
        """Leave-one-out: an item verified against its own template must not self-match."""
        matcher = SiftMatcher()
        feats = self._feats(3)
        templates = [(i, f) for i, f in enumerate(feats)]
        ids = list(range(len(feats)))
        with_self = best_match_stats_many(templates, feats, matcher)
        without_self = best_match_stats_many(templates, feats, matcher, skip=lambda key, i: key == ids[i])
        # Every item trivially self-matches when its own template is in play, and
        # loses that free win when it is held out.
        assert all(s.inlier_count > 0 for s in with_self)
        for held, free in zip(without_self, with_self):
            assert held.inlier_count < free.inlier_count

    def test_empty_inputs(self) -> None:
        matcher = SiftMatcher()
        feats = self._feats(2)
        assert best_match_stats_many([], feats, matcher) == [MatchStats(), MatchStats()]
        assert best_match_stats_many([(0, feats[0])], [], matcher) == []


# --------------------------------------------------------------------------
# cap_detect_resolution
# --------------------------------------------------------------------------


class TestCapDetectResolution:
    def test_downsamples_above_the_budget_and_preserves_aspect(self) -> None:
        img = np.zeros((1000, 2000), dtype=np.uint8)
        out = cap_detect_resolution(img, 500_000)
        assert out.shape[0] * out.shape[1] <= 500_000
        assert out.shape[1] / out.shape[0] == pytest.approx(2.0, rel=0.02)

    def test_image_under_the_budget_is_returned_untouched(self) -> None:
        img = _textured_image(0, size=100)
        out = cap_detect_resolution(img, 500_000)
        assert out is img

    @pytest.mark.parametrize("budget", [0, -1])
    def test_non_positive_budget_disables_the_cap(self, budget: int) -> None:
        img = np.zeros((1000, 2000), dtype=np.uint8)
        assert cap_detect_resolution(img, budget) is img

    def test_tiny_images_are_never_degenerate(self) -> None:
        for shape in [(1, 1), (1, 50), (3, 3)]:
            out = cap_detect_resolution(np.zeros(shape, dtype=np.uint8), 1)
            assert out.ndim == 2 and out.shape[0] >= 1 and out.shape[1] >= 1


class TestDetectionCapInMatcher:
    def test_cap_is_applied_and_coordinates_stay_normalised(self) -> None:
        """A capped detection still reports keypoints in [0, 1] of the original frame."""
        img = _structured_image(0, size=900)  # 810k px
        capped = SiftMatcher(max_detect_pixels=100_000)
        feats = capped.detect_and_describe(img, max_features=DEFAULT_MAX_FEATURES)
        assert feats.count > 0
        kp = feats.keypoints_f32()
        assert kp[:, 0].min() >= 0.0 and kp[:, 0].max() <= 1.0
        assert kp[:, 1].min() >= 0.0 and kp[:, 1].max() <= 1.0
        assert feats.descriptors_f32().shape[1] == SIFT_DESCRIPTOR_DIM

    def test_uncapped_matcher_detects_at_native_resolution(self) -> None:
        """``max_detect_pixels=0`` opts out, and finds more keypoints for it."""
        img = _structured_image(1, size=900)
        capped = SiftMatcher(max_detect_pixels=40_000).detect_and_describe(img, max_features=5000)
        native = SiftMatcher(max_detect_pixels=0).detect_and_describe(img, max_features=5000)
        assert native.count > capped.count

    def test_features_from_different_caps_still_match(self) -> None:
        """SIFT is scale-invariant, so a capped template verifies against an uncapped
        candidate - which is what keeps pickles written before the cap usable."""
        img = _structured_image(2, size=600)
        capped = SiftMatcher(max_detect_pixels=90_000).detect_and_describe(img, max_features=400)
        native = SiftMatcher(max_detect_pixels=0).detect_and_describe(img, max_features=400)
        stats = SiftMatcher().verify(capped, native)
        assert stats.model_ok and stats.inlier_count >= 8
