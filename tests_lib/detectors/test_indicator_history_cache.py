"""Tests for the read-only indicator-history path and its per-step cache costs.

The progress-plot modal used to load its series through the
``calculate_*_over_time`` helpers, which call ``_ensure_cache`` and therefore
retrain one MLP per uncached label step - plus, on a cold cache, a full
hierarchical-k-means coverage-atlas build - on the calling thread while holding
``_progress_lock``.  ``/api/labeling-status`` deliberately defers exactly that
work to a background worker (issue #2397), so the cache is behind for most of a
labeling session and the modal absorbed the whole deferred build.

These tests pin the three pieces that fixed it:

* :func:`cached_indicator_history` reads the cache and **never advances it**,
  reporting ``complete=False`` instead, and never blocks on ``_progress_lock``.
* The stability pass's monitored pool is materialised once per dataset rather
  than once per label step.
* The fallback coverage-atlas build applies the same depth cap as every other
  build site.
"""

from __future__ import annotations

import threading

import numpy as np

import vtscore.detectors.labeling_progress as lp
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY


def _active_cache() -> lp._ProgressCache:
    """The cache for the pair these tests run under, as an entry point would resolve it."""
    with lp._progress_lock:
        return lp._active_cache()


def _pool_tensor():
    """The single memoised stability pool tensor, or ``None`` when none is built."""
    tensors = [pool.X for pool in lp._monitored_pools.values()]
    return tensors[0] if tensors else None


def _linear_model(dim: int = 8):
    """A tiny fixed linear model producing ``(n, 1)`` logits."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(dim, 1))


def _clips(n: int, dim: int = 8, seed: int = 0) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    return {
        cid: {EMBEDDINGS_KEY: {"test": rng.standard_normal(dim).astype(np.float32)}, "embedder": "test"}
        for cid in range(n)
    }


def _history(n_votes: int) -> list[tuple[int, str, float]]:
    return [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(n_votes)]


def _votes(n_votes: int) -> tuple[dict[int, None], dict[int, None]]:
    good = {k: None for k in range(n_votes) if k % 2 == 0}
    bad = {k: None for k in range(n_votes) if k % 2 == 1}
    return good, bad


class TestCachedIndicatorHistory:
    def test_cold_cache_reports_incomplete_without_advancing(self):
        """The whole point: a cold read must not trigger any per-step training."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        data, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 0)

        assert complete is False
        assert data == []
        # No steps were built: the cache is exactly as cold as we left it.
        assert _active_cache().steps == []

    def test_warm_cache_returns_series_for_every_metric(self):
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        # Advance the cache the way the background worker does.
        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        for metric in ("smart", "stable", "diverse"):
            data, complete = lp.cached_indicator_history(metric, clips, history, good, bad, 0)
            assert complete is True, metric
            assert len(data) > 0, metric

    def test_partially_advanced_cache_reports_incomplete(self):
        """A cache covering only a prefix must not yield a truncated plot."""
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)
        # More votes have landed since the last background refresh.
        data, complete = lp.cached_indicator_history("smart", clips, _history(8), good, bad, 0)

        assert complete is False
        assert data == []

    def test_inclusion_change_reports_incomplete(self):
        """A cache built for another inclusion value would be rebuilt on read."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)
        _, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 5)

        assert complete is False
        # The read must not have rebuilt the cache under the new inclusion.
        assert _active_cache().inclusion == 0

    def test_does_not_block_on_an_in_flight_cache_build(self, monkeypatch):
        """A click landing mid-refresh falls through instead of hanging.

        The background worker holds ``_progress_lock`` for the entire build, so
        a blocking read would reintroduce exactly the hang this path exists to
        avoid.
        """
        monkeypatch.setattr(lp, "_CACHE_READ_LOCK_TIMEOUT", 0.05)
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        holding = threading.Event()
        release = threading.Event()

        def _hold_lock():
            with lp._progress_lock:
                holding.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        try:
            assert holding.wait(timeout=5)
            # The cache is complete, but the lock is held: report unavailable
            # rather than waiting for the holder.
            data, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 0)
            assert complete is False
            assert data == []
        finally:
            release.set()
            holder.join(timeout=5)

        # Once the lock frees up the same read succeeds.
        _, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 0)
        assert complete is True


class TestMonitoredPoolReuse:
    def test_pool_tensor_is_built_once_per_cache_lifetime(self):
        """Advancing further steps must reuse the pool, not re-materialise it.

        Rebuilding it per step was an O(N x D) numpy materialisation on every
        label-history step and dominated the cost of advancing the cache.
        """
        clips = _clips(200)
        good, bad = _votes(6)
        lp.clear_progress_cache()

        lp.calculate_prediction_stability_over_time(clips, _history(6), 0)
        first = _pool_tensor()
        assert first is not None

        # Advance the cache with more steps: same pool object, no rebuild.
        lp.calculate_prediction_stability_over_time(clips, _history(12), 0)
        assert _pool_tensor() is first

    def test_pool_is_shared_by_every_detector_over_one_dataset(self):
        """The pool is a pure function of ``clips_dict``, so it is keyed by dataset.

        Caching several ``(dataset, detector)`` pairs at once must not multiply
        the biggest thing this module holds - the pool tensor is up to
        ``_STABILITY_MAX_SAMPLES`` x embedding-dim floats.
        """
        clips = _clips(60)
        lp.clear_progress_cache()

        lp.calculate_prediction_stability_over_time(clips, _history(6), 0)
        first = _pool_tensor()
        assert first is not None
        dataset_id = _active_cache().key[0]

        # A second detector over the same dataset must reuse that tensor rather
        # than materialise its own.
        other = lp._ProgressCache(key=(dataset_id, "det_other"), good_ids={0, 2}, bad_ids={1, 3})
        with lp._progress_lock:
            lp._compute_step_stability(
                other,
                _linear_model(),
                threshold=0.5,
                clips_dict=clips,
                all_media_ids=sorted(clips),
                t=1,
                num_labels=4,
            )

        assert lp._monitored_pools[dataset_id].X is first
        assert len(lp._monitored_pools) == 1

    def test_pool_is_dropped_on_cache_clear(self):
        """Medias may have changed, so the derived pool must not survive."""
        clips = _clips(60)
        lp.clear_progress_cache()
        lp.calculate_prediction_stability_over_time(clips, _history(6), 0)
        assert _pool_tensor() is not None

        lp.clear_progress_cache()

        assert lp._monitored_pools == {}

    def test_stability_counts_match_the_unsampled_pool(self):
        """Scoring the whole pool and dropping labels must not change the counts."""
        clips = _clips(60)
        lp.clear_progress_cache()

        stability = lp.calculate_prediction_stability_over_time(clips, _history(6), 0)

        assert stability
        for entry in stability:
            # 60 medias, `num_labels` of them labeled at that step.
            assert entry["num_unlabeled"] == 60 - entry["num_labels"]


class TestFallbackAtlasDepth:
    def test_fallback_build_applies_the_depth_cap(self, monkeypatch):
        """The fallback atlas must not be deeper than the context atlas it stands in for.

        Omitting ``max_depth`` left this build on ``COVERAGE_ATLAS_MAX_DEPTH``
        while every other build site passes ``auto_max_depth``, so the throwaway
        atlas cost many more k-means fits than the real one.
        """
        import vtscore.state.coverage_atlas as ca

        monkeypatch.setattr(lp, "_active_context_atlas", lambda: None)
        monkeypatch.setattr(ca, "auto_max_depth", lambda n, k=3, **kw: 2)

        atlas = lp._build_coverage_atlas(_clips(120))

        assert atlas is not None
        assert atlas.max_depth == 2

    def test_reuses_the_context_atlas_structure_when_ids_match(self, monkeypatch):
        """The clone path skips the hierarchical k-means entirely."""
        from vtscore.state.coverage_atlas import CoverageAtlas

        clips = _clips(120)
        vectors = {cid: media[EMBEDDINGS_KEY]["test"] for cid, media in clips.items()}
        ctx_atlas = CoverageAtlas(vectors, k=3)
        monkeypatch.setattr(lp, "_active_context_atlas", lambda: ctx_atlas)

        atlas = lp._build_coverage_atlas(clips)

        assert atlas is not ctx_atlas
        # Structure shared by reference; only the label overlay is fresh.
        assert atlas.nodes is ctx_atlas.nodes
