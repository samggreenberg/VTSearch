"""Tests for the subsampling fast-path in ``_compute_step_stability``.

``/api/labeling-status`` is polled every 2 s during labeling and advances the
per-step cache on the request thread; the stability step runs a forward pass
over *all* unlabeled media - O(dataset) on the first poll after each new vote.
Above ``_STABILITY_MAX_SAMPLES`` the pass scores a deterministic seeded sample
of the eligible pool instead.  These tests pin that (a) the monitored set is
bounded by the cap, (b) it is stable/deterministic across steps so the
step-to-step flip comparison stays meaningful, and (c) at or below the cap the
full unlabeled set is used unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import vtscore.detectors.labeling_progress as lp
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY


def _clips(n: int, dim: int = 8, seed: int = 0) -> dict[int, dict]:
    """Build ``n`` media dicts each carrying a single seeded embedding."""
    rng = np.random.default_rng(seed)
    return {
        cid: {EMBEDDINGS_KEY: {"test": rng.standard_normal(dim).astype(np.float32)}, "embedder": "test"}
        for cid in range(n)
    }


def _model(dim: int = 8) -> nn.Sequential:
    """A tiny fixed linear model producing ``(n, 1)`` logits."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(dim, 1))


def _cache(good: set[int], bad: set[int]) -> lp._ProgressCache:
    """A standalone cache seeded with a labelset, as ``_ensure_cache`` would leave it."""
    return lp._ProgressCache(key=("ds_subsample", "det_subsample"), good_ids=set(good), bad_ids=set(bad))


def _run_step(cache: lp._ProgressCache, clips: dict[int, dict], model: nn.Sequential) -> dict | None:
    all_media_ids = sorted(clips.keys())
    return lp._compute_step_stability(
        cache,
        model,
        threshold=0.5,
        clips_dict=clips,
        all_media_ids=all_media_ids,
        t=1,
        num_labels=len(cache.good_ids) + len(cache.bad_ids),
    )


class TestStabilitySubsample:
    def test_monitored_set_bounded_by_cap(self, monkeypatch):
        """Above the cap the forward pass scores at most ``_STABILITY_MAX_SAMPLES`` items."""
        monkeypatch.setattr(lp, "_STABILITY_MAX_SAMPLES", 50)
        clips = _clips(300)
        cache = _cache({0, 1}, {2, 3})

        _run_step(cache, clips, _model())  # seeds prev predictions (stability is None first step)
        assert cache.prev_predictions is not None
        assert len(cache.prev_predictions) <= 50

        stability = _run_step(cache, clips, _model())
        assert stability is not None
        assert stability["num_unlabeled"] <= 50
        assert stability["num_unlabeled"] == len(cache.prev_predictions)

    def test_monitored_set_deterministic_across_steps(self, monkeypatch):
        """The sampled ids are identical across calls, so flips compare like-for-like."""
        monkeypatch.setattr(lp, "_STABILITY_MAX_SAMPLES", 50)
        clips = _clips(300)
        cache = _cache({0, 1}, {2, 3})

        _run_step(cache, clips, _model())
        assert cache.prev_predictions is not None
        first = set(cache.prev_predictions)

        # Drop the memoised pool as well, so the second step re-runs the seeded
        # sampling from scratch rather than trivially reusing the first draw.
        lp._monitored_pools.clear()
        cache.prev_predictions = None
        _run_step(cache, clips, _model())
        assert cache.prev_predictions is not None
        second = set(cache.prev_predictions)

        assert first == second

    def test_below_cap_uses_full_unlabeled_set(self):
        """At or below the cap every unlabeled item is scored (no sampling)."""
        clips = _clips(300)  # well under the 50k default cap
        cache = _cache({0, 1}, {2, 3})

        _run_step(cache, clips, _model())
        stability = _run_step(cache, clips, _model())

        assert stability is not None
        assert stability["num_unlabeled"] == 300 - 4  # all unlabeled scored
