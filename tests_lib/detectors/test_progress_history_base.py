"""The Smart / Stable cache replays the label history on top of the restored labels (#5358).

A detector's saved labels come back into its votes silently: the label history
records none of them, but every learned sort trains on them, and its model is
keyed by the votes it trained on.  The cache used to build its label sets from
the history alone, so they never equalled a sort's key, no step ever found its
model, and Smart and Stable stayed on "no history yet" for the whole session,
so Autopilot never got past Refine Boundary.  The detector's
``label_history_base`` is what the cache now starts from.
"""

from __future__ import annotations

import numpy as np

import vtscore.detectors.labeling_progress as lp
from vtscore.detectors.learned_sort import model_matches_local_votes
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY
from vtscore.state.core import DetectorContext, set_thread_detector_context

#: The restored labels: a Good (the example the detector was made from) and a Bad.
BASE_GOOD = 100
BASE_BAD = 101


def _clips(n: int = 120, dim: int = 8, seed: int = 0) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    return {
        cid: {EMBEDDINGS_KEY: {"test": rng.standard_normal(dim).astype(np.float32)}, "embedder": "test"}
        for cid in range(n)
    }


def _linear_model(seed: int, dim: int = 8):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, 1))


def _history(n_votes: int) -> list[tuple[int, str, float]]:
    return [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(n_votes)]


def _activate(base_good: set[int], base_bad: set[int]) -> DetectorContext:
    """Make a detector whose votes started from *base_good* / *base_bad* the active one."""
    ctx = DetectorContext("det_5358")
    ctx.label_history_base = (frozenset(base_good), frozenset(base_bad))
    set_thread_detector_context(ctx)
    lp.clear_progress_cache()
    return ctx


def _sort_after_every_vote(history, base_good: set[int], base_bad: set[int]) -> tuple[dict, dict]:
    """Inject a model per vote, keyed by the votes a learned sort would have trained on.

    Those are the restored labels plus every vote so far: ``run_learned_sort``
    keys its model by ``dict(good_votes)`` / ``dict(bad_votes)``.
    """
    good = dict.fromkeys(base_good)
    bad = dict.fromkeys(base_bad)
    for t, (mid, label, _) in enumerate(history):
        (good if label == "good" else bad)[mid] = None
        (bad if label == "good" else good).pop(mid, None)
        lp.inject_live_model(dict(good), dict(bad), _linear_model(t), 0.5)
    return good, bad


class TestRestoredLabelsAreTheBase:
    def test_every_sorted_step_carries_its_model(self):
        clips, history = _clips(), _history(10)
        _activate({BASE_GOOD}, {BASE_BAD})
        good, bad = _sort_after_every_vote(history, {BASE_GOOD}, {BASE_BAD})

        smart = lp.calculate_fbeta_over_time(clips, history, good, bad)
        stable = lp.calculate_prediction_stability_over_time(clips, history)

        assert [p["time_index"] for p in smart] == list(range(10))
        # One entry per retraining after the first, which has nothing to compare against.
        assert len(stable) == 9

    def test_the_steps_label_sets_hold_the_restored_labels(self):
        clips, history = _clips(), _history(4)
        _activate({BASE_GOOD}, {BASE_BAD})
        _sort_after_every_vote(history, {BASE_GOOD}, {BASE_BAD})

        lp.calculate_prediction_stability_over_time(clips, history)
        with lp._progress_lock:
            steps = lp._active_cache().steps
        assert set(steps[0]["good_ids"]) == {BASE_GOOD, 0}
        assert set(steps[0]["bad_ids"]) == {BASE_BAD}
        # A point's label count is the votes the user sees, restored ones included.
        assert len(steps[-1]["good_ids"]) + len(steps[-1]["bad_ids"]) == 2 + 4

    def test_restored_labels_leave_the_stability_pool(self):
        """A restored label is labelled, so it is no flip candidate."""
        clips, history = _clips(), _history(4)
        _activate({BASE_GOOD}, {BASE_BAD})
        _sort_after_every_vote(history, {BASE_GOOD}, {BASE_BAD})

        stable = lp.calculate_prediction_stability_over_time(clips, history)

        assert {e["num_pool"] - e["num_unlabeled"] for e in stable} == {2 + 2, 2 + 3, 2 + 4}

    def test_without_the_base_no_step_finds_its_model(self):
        """The shape of the bug: the same sorts against a cache that does not know the base."""
        clips, history = _clips(), _history(10)
        _activate(set(), set())
        good, bad = _sort_after_every_vote(history, {BASE_GOOD}, {BASE_BAD})

        assert lp.calculate_fbeta_over_time(clips, history, good, bad) == []


class TestFlipOfARestoredLabel:
    def test_replays_from_the_base(self):
        """Every model trained on the restored label, so a flip of it discards every step."""
        clips, history = _clips(), _history(6)
        _activate({BASE_GOOD}, {BASE_BAD})
        _sort_after_every_vote(history, {BASE_GOOD}, {BASE_BAD})
        lp.calculate_prediction_stability_over_time(clips, history)

        lp.invalidate_progress_cache_from(BASE_GOOD)

        with lp._progress_lock:
            cache = lp._active_cache()
            assert cache.steps == []
            assert cache.good_ids == {BASE_GOOD}
            assert cache.bad_ids == {BASE_BAD}
            assert cache.live_models == {}


class TestDropProgressCache:
    def test_drops_only_the_named_pair(self):
        clips, history = _clips(), _history(4)
        ctx = _activate({BASE_GOOD}, {BASE_BAD})
        lp.calculate_prediction_stability_over_time(clips, history)
        other = ("other_dataset", "other_detector")
        with lp._progress_lock:
            key = lp._active_cache().key
            lp._caches[other] = lp._ProgressCache(key=other)

        lp.drop_progress_cache(key[0], ctx.detector_id)

        with lp._progress_lock:
            assert key not in lp._caches
            assert other in lp._caches

    def test_a_new_cache_reads_the_new_base(self):
        """A rehydrate replaces the base and drops the pair, so the next cache starts from the new one."""
        ctx = _activate({BASE_GOOD}, {BASE_BAD})
        with lp._progress_lock:
            key = lp._active_cache().key
        ctx.label_history_base = (frozenset({7}), frozenset())
        lp.drop_progress_cache(*key)

        with lp._progress_lock:
            cache = lp._active_cache()
            assert cache.good_ids == {7}
            assert cache.bad_ids == set()


class TestForeignLabelsDoNotKeepTheModelOut:
    """A label this dataset does not hold is in every model the detector serves here."""

    def test_matches_on_the_local_votes(self):
        assert model_matches_local_votes(object(), True, {1, 2}, {3}, {1: None, 2: None}, {3: None})

    def test_still_refuses_local_votes_that_differ(self):
        assert not model_matches_local_votes(object(), True, {1, 2}, {3}, {1: None}, {3: None})
