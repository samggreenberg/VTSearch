"""Tests for the read-only indicator-history path and its per-step cache costs.

The progress-plot modal used to load its series through the
``calculate_*_over_time`` helpers, which call ``_ensure_cache`` and therefore
retrain one MLP per uncached label step - plus, on a cold cache, a full
hierarchical-k-means coverage-atlas build - on the calling thread while holding
``_progress_lock``.  ``/api/labeling-status`` deliberately defers exactly that
work to a background worker (issue #2397), so the cache is behind for most of a
labeling session and the modal absorbed the whole deferred build.

These tests pin the pieces that fixed it:

* :func:`cached_indicator_history` reads the cache and **never advances it**,
  reporting ``complete=False`` instead, and never blocks on ``_progress_lock``.
  It hands over whatever prefix it has - the same steps the Smart/Stable lights
  were derived from - rather than withholding it until the cache is complete.
* The stability pass's monitored pool is materialised once per cache lifetime
  rather than once per label step.
* The fallback coverage-atlas build applies the same depth cap as every other
  build site.
* The model walk and the diversity replay are separate caches, so neither
  metric funds the other's dominant cost.
* An inclusion change re-derives cutoffs from the cached models instead of
  discarding them.
* The walk reports per-step progress and can publish a partial series.
* Correcting an existing label rewrites that click in place and invalidates
  from it, and the replay reproduces a cold rebuild exactly; a first label
  appends and invalidates nothing.
* The caches are stamped with the detector they belong to.
"""

from __future__ import annotations

import threading

import numpy as np

import vtscore.detectors.labeling_progress as lp
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY


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
        assert lp._cached_steps == []

    def test_warm_cache_returns_series_for_the_model_metrics(self):
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        # Advance the cache the way the background worker does.
        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        for metric in ("smart", "stable"):
            data, complete = lp.cached_indicator_history(metric, clips, history, good, bad, 0)
            assert complete is True, metric
            assert len(data) > 0, metric

    def test_model_walk_does_not_build_the_diversity_series(self):
        """The two caches are independent, so neither funds the other's cost.

        Advancing the model cache must not build a coverage atlas (a
        hierarchical k-means over every embedding, and the dominant cost of the
        old shared walk), and building the diversity series must not train a
        single model.
        """
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)
        assert lp._cache_coverage_atlas is None
        assert lp._cached_diversity == []
        data, complete = lp.cached_indicator_history("diverse", clips, history, good, bad, 0)
        assert complete is False
        assert data == []

        lp.clear_progress_cache()
        lp.calculate_diversity_level_over_time(clips, history)
        assert lp._cached_steps == []
        data, complete = lp.cached_indicator_history("diverse", clips, history, good, bad, 0)
        assert complete is True
        assert len(data) > 0

    def test_partially_advanced_cache_reports_incomplete(self):
        """A prefix is served, but flagged so the caller tops it up.

        See ``TestPrefixIsServedNotWithheld`` for why the prefix is handed over
        rather than withheld.
        """
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)
        # More votes have landed since the last background refresh.
        data, complete = lp.cached_indicator_history("smart", clips, _history(8), good, bad, 0)

        assert complete is False
        assert 0 < len(data) < 8

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
        assert lp._cache_inclusion == 0

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
        first = lp._cache_monitored_X
        assert first is not None
        assert lp._cache_monitored_ids is not None

        # Advance the cache with more steps: same pool object, no rebuild.
        lp.calculate_prediction_stability_over_time(clips, _history(12), 0)
        assert lp._cache_monitored_X is first

    def test_pool_is_dropped_on_cache_clear(self):
        """Medias may have changed, so the derived pool must not survive."""
        clips = _clips(60)
        lp.clear_progress_cache()
        lp.calculate_prediction_stability_over_time(clips, _history(6), 0)
        assert lp._cache_monitored_X is not None

        lp.clear_progress_cache()

        assert lp._cache_monitored_X is None
        assert lp._cache_monitored_ids is None
        assert lp._cache_monitored_set is None

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


class TestRethresholdOnInclusionChange:
    """Moving the inclusion slider must not retrain the per-step models.

    Inclusion never reaches ``train_model`` - it is applied afterwards as a pure
    cutoff - so every cached model is still exactly the model its step would
    train under the new value.  The slider used to call
    ``clear_progress_cache()`` anyway, which meant the next metric read retrained
    one model per label step from scratch.
    """

    def test_models_survive_and_are_reused(self):
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)
        before = [step["model"] for step in lp._cached_steps if step["model"] is not None]
        assert before

        lp.rethreshold_progress_cache(5)
        lp.calculate_error_cost_over_time(clips, history, good, bad, 5)

        after = [step["model"] for step in lp._cached_steps if step["model"] is not None]
        assert len(after) == len(before)
        # Identity, not equality: nothing was retrained.
        assert all(a is b for a, b in zip(after, before, strict=True))
        assert lp._cache_inclusion == 5

    def test_cutoffs_actually_move(self):
        """Reusing the models must still re-derive every step's threshold."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, history, good, bad, -8)
        low = [step["threshold"] for step in lp._cached_steps if step["threshold"] is not None]

        lp.rethreshold_progress_cache(8)
        lp.calculate_error_cost_over_time(clips, history, good, bad, 8)
        high = [step["threshold"] for step in lp._cached_steps if step["threshold"] is not None]

        assert low and len(low) == len(high)
        # A more inclusive setting cannot raise the bar on any step.
        assert all(h <= low_t + 1e-6 for h, low_t in zip(high, low, strict=True))
        assert any(h < low_t for h, low_t in zip(high, low, strict=True))

    def test_the_expensive_atlas_is_not_dropped(self):
        """The coverage atlas is inclusion-independent, so it must survive."""
        clips = _clips(60)
        history = _history(8)
        lp.clear_progress_cache()
        lp.calculate_diversity_level_over_time(clips, history)
        lp.calculate_error_cost_over_time(clips, history, *_votes(8), 0)
        atlas = lp._cache_coverage_atlas
        assert atlas is not None

        lp.rethreshold_progress_cache(3)

        assert lp._cache_coverage_atlas is atlas
        assert len(lp._cached_diversity) == len(history)


class TestCacheWalkProgress:
    """The walk is the whole runtime of the eval job, so it must report progress.

    It used to run between a single 0/N and a single N/N update, leaving the
    modal's bar pinned at 0 % for the entire wait.
    """

    def test_ensure_cache_reports_every_step(self):
        clips = _clips(60)
        history = _history(8)
        lp.clear_progress_cache()
        seen: list[tuple[int, int]] = []

        lp.calculate_error_cost_over_time(clips, history, *_votes(8), 0, on_step=lambda d, t: seen.append((d, t)))

        assert seen == [(k, len(history)) for k in range(1, len(history) + 1)]

    def test_diversity_replay_reports_every_step(self):
        clips = _clips(60)
        history = _history(8)
        lp.clear_progress_cache()
        seen: list[tuple[int, int]] = []

        lp.calculate_diversity_level_over_time(clips, history, on_step=lambda d, t: seen.append((d, t)))

        assert seen == [(k, len(history)) for k in range(1, len(history) + 1)]

    def test_partial_series_grows_with_the_walk(self):
        """A mid-walk snapshot is a valid prefix, so the chart can fill in."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lengths: list[int] = []

        def _snapshot(done: int, total: int) -> None:
            lengths.append(len(lp.partial_indicator_series("smart", clips, good, bad, 0)))

        final = lp.calculate_error_cost_over_time(clips, history, good, bad, 0, on_step=_snapshot)

        assert lengths == sorted(lengths)
        assert lengths[-1] == len(final)


class TestPrefixIsServedNotWithheld:
    """A behind cache still hands over what it has.

    The Smart and Stable lights are computed from these very steps, so a panel
    showing a red/yellow/green verdict is a panel whose curve is already partly
    computed.  Returning an empty list until the cache covered the *whole*
    history - which it stops doing the instant the next vote lands - made the
    user wait out a full recompute to see data the light was derived from.
    """

    def test_behind_cache_returns_its_prefix(self):
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        # Background worker got as far as 4 votes; 8 have now landed.
        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)

        for metric in ("smart", "stable"):
            data, complete = lp.cached_indicator_history(metric, clips, _history(8), good, bad, 0)
            assert complete is False, metric
            assert len(data) > 0, metric

    def test_prefix_matches_the_first_points_of_the_finished_series(self):
        """The prefix must be a true prefix, not a differently-computed stub."""
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)
        partial, _ = lp.cached_indicator_history("smart", clips, _history(8), good, bad, 0)

        full = lp.calculate_error_cost_over_time(clips, _history(8), good, bad, 0)

        assert full[: len(partial)] == partial

    def test_a_cold_cache_still_returns_nothing(self):
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        data, complete = lp.cached_indicator_history("smart", clips, _history(8), good, bad, 0)

        assert complete is False
        assert data == []
        assert lp._cached_steps == []

    def test_another_inclusion_value_withholds_the_prefix(self):
        """Cutoffs - and so every plotted value - belong to one inclusion."""
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)

        data, complete = lp.cached_indicator_history("smart", clips, _history(8), good, bad, 5)

        assert complete is False
        assert data == []


class TestCacheOwnership:
    """The caches are module-level; ``label_history`` is per-detector.

    The active detector is resolved per request from a header, so switching
    between two already-resident detectors fires no registration hook. Without
    an owner stamp the previous detector's steps stayed put, and the resume-from-
    ``len(cache)`` rule would serve them verbatim or graft the new detector's
    events onto them.
    """

    def test_a_detector_switch_discards_the_other_detector_s_steps(self, monkeypatch):
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        monkeypatch.setattr(lp, "_current_owner", lambda: ("ds", "detector-a"))
        lp.calculate_error_cost_over_time(clips, _history(8), good, bad, 0)
        assert len(lp._cached_steps) == 8

        # Same dataset, different detector, and a *shorter* history - the case
        # the old code returned early on, serving detector A's curve for B.
        monkeypatch.setattr(lp, "_current_owner", lambda: ("ds", "detector-b"))
        data, complete = lp.cached_indicator_history("smart", clips, _history(4), good, bad, 0)
        assert (data, complete) == ([], False)
        assert lp.is_status_cache_fresh(_history(4), 0) is False

        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)
        assert len(lp._cached_steps) == 4

    def test_a_shrinking_history_discards_the_cache(self, monkeypatch):
        """A failed label-sync rolls ``label_history`` back to a saved prefix."""
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        monkeypatch.setattr(lp, "_current_owner", lambda: ("ds", "det"))
        lp.calculate_error_cost_over_time(clips, _history(8), good, bad, 0)
        assert len(lp._cached_steps) == 8

        lp.calculate_error_cost_over_time(clips, _history(3), good, bad, 0)

        assert len(lp._cached_steps) == 3

    def test_the_diversity_cache_is_owned_too(self, monkeypatch):
        clips = _clips(60)
        lp.clear_progress_cache()
        monkeypatch.setattr(lp, "_current_owner", lambda: ("ds", "detector-a"))
        lp.calculate_diversity_level_over_time(clips, _history(8))
        assert len(lp._cached_diversity) == 8

        monkeypatch.setattr(lp, "_current_owner", lambda: ("ds", "detector-b"))
        lp.calculate_diversity_level_over_time(clips, _history(4))

        assert len(lp._cached_diversity) == 4


class TestCorrectionInvalidatesFromThatPoint:
    """A corrected click is replayed, and the replay matches a cold rebuild.

    ``correct_label_in_history`` rewrites the click *in place*, so every step
    from it onward has to refit against the corrected label -- that is the
    correction propagating, and it is the whole reason to invalidate.  What must
    not differ is the *result*: resuming from the surviving prefix has to produce
    exactly what rebuilding the corrected history from scratch produces.

    The old code got this wrong twice over.  It invalidated against an
    append-only history, so the replay re-read the unchanged events and
    reproduced the models it had just deleted -- a rebuild that changed nothing.
    And it nulled the prediction baseline at the truncation point, so the first
    replayed step recorded no flip count and the Stable series came out a point
    short of a cold rebuild.
    """

    def _corrected(self, n: int, media_id: int, label: str) -> list[tuple[int, str, float]]:
        """*n* clicks with *media_id*'s own click rewritten to *label* in place."""
        history = _history(n)
        idx = next(i for i, ev in enumerate(history) if ev[0] == media_id)
        history[idx] = (media_id, label, history[idx][2])
        return history

    def test_replay_matches_a_cold_rebuild_of_the_corrected_history(self):
        clips = _clips(200)
        corrected = self._corrected(10, 4, "bad")
        good = {k: None for k in range(10) if k % 2 == 0 and k != 4}
        bad = {k: None for k in range(10) if k % 2 == 1 or k == 4}

        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, _history(10), *_votes(10), 0)
        lp.invalidate_progress_cache_from(4)
        assert len(lp._cached_steps) == 4, "steps before media 4's click must survive"
        incremental = lp.calculate_error_cost_over_time(clips, corrected, good, bad, 0)
        incremental_stability = lp.calculate_prediction_stability_over_time(clips, corrected, 0)

        lp.clear_progress_cache()
        cold = lp.calculate_error_cost_over_time(clips, corrected, good, bad, 0)
        cold_stability = lp.calculate_prediction_stability_over_time(clips, corrected, 0)

        assert incremental == cold
        assert incremental_stability == cold_stability

    def test_the_stability_chain_resumes_instead_of_restarting(self):
        """Nulling the baseline dropped the flip count at the resume point."""
        clips = _clips(200)
        corrected = self._corrected(10, 4, "bad")

        lp.clear_progress_cache()
        lp.calculate_prediction_stability_over_time(clips, _history(10), 0)
        lp.invalidate_progress_cache_from(4)
        resumed = lp.calculate_prediction_stability_over_time(clips, corrected, 0)

        lp.clear_progress_cache()
        cold = lp.calculate_prediction_stability_over_time(clips, corrected, 0)

        assert len(resumed) == len(cold)
        assert [entry["time_index"] for entry in resumed] == [entry["time_index"] for entry in cold]

    def test_the_correction_actually_changes_the_later_steps(self):
        """If the replay reproduced the old models the invalidation was pointless."""
        clips = _clips(200)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, _history(10), *_votes(10), 0)
        before = [(sorted(s["good_ids"]), sorted(s["bad_ids"])) for s in lp._cached_steps]

        corrected = self._corrected(10, 4, "bad")
        lp.invalidate_progress_cache_from(4)
        good = {k: None for k in range(10) if k % 2 == 0 and k != 4}
        bad = {k: None for k in range(10) if k % 2 == 1 or k == 4}
        lp.calculate_error_cost_over_time(clips, corrected, good, bad, 0)
        after = [(sorted(s["good_ids"]), sorted(s["bad_ids"])) for s in lp._cached_steps]

        assert before[:4] == after[:4], "the surviving prefix must be identical"
        assert before[4:] != after[4:], "the corrected label must reach every later step"
        for step_good, _step_bad in after[4:]:
            assert 4 not in step_good

    def test_an_unlabeled_media_invalidates_nothing(self):
        clips = _clips(200)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, _history(10), *_votes(10), 0)

        lp.invalidate_progress_cache_from(999)

        assert len(lp._cached_steps) == 10

    def test_a_polarity_correction_leaves_the_diversity_replay_alone(self):
        """Coverage is polarity-agnostic, so only a removal rewinds it."""
        clips = _clips(200)
        lp.clear_progress_cache()
        lp.calculate_diversity_level_over_time(clips, _history(10))
        atlas = lp._cache_coverage_atlas
        series = list(lp._cached_diversity)

        lp.invalidate_progress_cache_from(4)
        assert lp._cached_diversity == series
        assert lp._cache_coverage_atlas is atlas

        # A removal does drop a labeled item, so the series is rewound - but the
        # atlas structure, the expensive part, is kept for the replay.
        lp.invalidate_progress_cache_from(4, coverage_changed=True)
        assert lp._cached_diversity == []
        assert lp._cache_coverage_atlas is atlas
