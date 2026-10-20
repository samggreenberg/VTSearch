"""What the Smart indicator reads a model at (issues #4243, #5134).

Smart re-scores its window of recent models against the current labelset.  Since
#5134 every point in that window is the model's **best F-beta** over every cut
through its scores, at the session's balance (F1 with none): cut-free, so it
cannot move when the served line switches rule, and read at one beta for the
whole window, so moving the balance re-reads every point rather than mixing two.

Before #5134 every point was ``FPR + FNR`` at
:data:`~vtscore.detectors.cost_trend.SMART_INCLUSION`, at each model's own cut for
that inclusion (:func:`~vtscore.detectors.cost_trend.smart_cut`, #4243).  That
rule is the eval harness's ``cost`` arm now, and stays pinned here as such.

Pinned three ways: the rules themselves, the app's progress cache, and the eval
harness's Smart window, which has to read the way the app does for a study of
the phase machine to mean anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vtscore.detectors.labeling_progress as lp
from vtscore.detectors.cost_trend import SMART_INCLUSION, SMART_WINDOW, best_fbeta, smart_cut
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY
from vtscore.training.thresholds import NO_GOOD_THRESHOLD, inclusion_cost_weights


class TestSmartCut:
    def test_prices_at_inclusion_zero(self):
        """Inclusion 0 is FPR + FNR: neither error kind outweighs the other."""
        assert SMART_INCLUSION == 0
        assert inclusion_cost_weights(SMART_INCLUSION) == (1.0, 1.0)

    def test_served_at_the_smart_inclusion_keeps_the_served_line_without_a_recut(self):
        """A user at the default Inclusion is scored exactly as before: no re-cut at all."""

        def recut(_k: float) -> float:
            raise AssertionError("a model served at SMART_INCLUSION must not be re-cut")

        assert smart_cut(0.42, SMART_INCLUSION, recut) == 0.42

    def test_otherwise_re_cuts_at_the_smart_inclusion(self):
        asked: list[float] = []

        def recut(k: float) -> float:
            asked.append(k)
            return 0.3

        assert smart_cut(0.8, 3, recut) == 0.3
        assert asked == [SMART_INCLUSION]

    def test_a_line_no_inclusion_drew_is_always_re_cut(self):
        """A balance draws its own line, a set rather than an inclusion (#4413): Smart re-cuts at its inclusion."""
        asked: list[float] = []

        def recut(k: float) -> float:
            asked.append(k)
            return 0.3

        assert smart_cut(0.8, None, recut) == 0.3
        assert asked == [SMART_INCLUSION]

    @pytest.mark.parametrize("unavailable", [None, math.nan, math.inf])
    def test_an_inclusion_blind_cut_keeps_the_served_line(self, unavailable):
        """No estimator to re-derive (a fallback or blend cut): the served line is the answer."""
        assert smart_cut(0.8, -2, lambda _k: unavailable) == 0.8


def _clips(n: int, dim: int = 8, seed: int = 0) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    return {
        cid: {EMBEDDINGS_KEY: {"test": rng.standard_normal(dim).astype(np.float32)}, "embedder": "test"}
        for cid in range(n)
    }


def _linear_model(dim: int = 8):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(dim, 1))


class TestProgressCacheReadsBestFbeta:
    """The app's Smart series, through the same ``_eval_cached_models`` the status reads (#5134)."""

    N_VOTES = 8

    def _series(self, threshold: float, beta: float | None = None, **kwargs):
        """The Smart series after a sort per vote, each model served at *threshold*; the models too."""
        pytest.importorskip("torch")
        clips = _clips(40)
        history = [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(self.N_VOTES)]
        good: dict[int, None] = {}
        bad: dict[int, None] = {}
        models = []
        lp.clear_progress_cache()
        for mid, label, _ in history:
            (good if label == "good" else bad)[mid] = None
            if good and bad:
                models.append(_linear_model())
                lp.inject_live_model(dict(good), dict(bad), models[-1], threshold, beta=beta, **kwargs)
        series = lp.calculate_fbeta_over_time(clips, history, good, bad, beta=beta)
        cached, complete = lp.cached_indicator_history("smart", clips, history, good, bad, beta=beta)
        assert complete
        assert cached == series
        assert series
        return series, models, clips, good, bad

    @staticmethod
    def _want(model, clips, good, bad, beta):
        import torch

        ids = list(good) + list(bad)
        x = torch.tensor(np.stack([clips[i][EMBEDDINGS_KEY]["test"] for i in ids]))
        with torch.no_grad():
            scores = torch.sigmoid(model(x)).numpy().ravel()
        return round(best_fbeta(scores, [1.0] * len(good) + [0.0] * len(bad), beta), 4)

    def test_each_point_is_the_best_fbeta_on_the_current_votes(self):
        series, models, clips, good, bad = self._series(0.5, beta=1.0)
        assert [set(e) for e in series] == [{"time_index", "num_labels", "fbeta"}] * len(series)
        assert [e["fbeta"] for e in series] == [self._want(m, clips, good, bad, 1.0) for m in models]

    def test_cut_free_whatever_line_was_served(self):
        # Served where nothing clears it: a cut-priced point would read every Good missed.
        nothing, *_ = self._series(NO_GOOD_THRESHOLD, beta=1.0)
        half, *_ = self._series(0.5, beta=1.0)
        assert [e["fbeta"] for e in nothing] == [e["fbeta"] for e in half]

    def test_read_at_the_sessions_balance_and_f1_with_none(self):
        series4, models, clips, good, bad = self._series(0.5, beta=4.0)
        assert [e["fbeta"] for e in series4] == [self._want(m, clips, good, bad, 4.0) for m in models]
        none, models, clips, good, bad = self._series(0.5, beta=None)
        assert [e["fbeta"] for e in none] == [self._want(m, clips, good, bad, 1.0) for m in models]

    def test_the_smart_cut_is_deprecated_and_ignored(self):
        with pytest.warns(DeprecationWarning, match="smart_threshold"):
            cut, *_ = self._series(NO_GOOD_THRESHOLD, beta=1.0, smart_threshold=0.0)
        plain, *_ = self._series(NO_GOOD_THRESHOLD, beta=1.0)
        assert cut == plain

    def test_the_memo_keys_on_the_balance(self):
        """A status read at one beta is not served at another (#5134)."""
        series, _models, clips, good, bad = self._series(0.5, beta=1.0)
        cache = lp._active_cache()
        steps = lp._model_step_indices(cache)
        assert lp._smart_memo_key(cache, steps, 4, 4, 1.0) != lp._smart_memo_key(cache, steps, 4, 4, 4.0)
        assert lp._smart_memo_key(cache, steps, 4, 4, None) == lp._smart_memo_key(cache, steps, 4, 4, 1.0)


def test_harness_window_is_read_like_the_app(monkeypatch):
    """The harness's Smart window (#5134): the last SMART_WINDOW steps, read at the run's balance."""
    pytest.importorskip("torch")
    import vtscore.eval.voting_iterations as vi

    seen: list[tuple[int, float | None]] = []
    real = vi._labelset_window_readings

    def spy(model_steps, good, bad, clips, beta, **kwargs):
        seen.append((len(model_steps), beta))
        out = real(model_steps, good, bad, clips, beta, **kwargs)
        assert len(out["fbeta_best"]) in (0, len(model_steps))
        return out

    monkeypatch.setattr(vi, "_labelset_window_readings", spy)
    rng = np.random.RandomState(0)
    medias = {
        mid: {
            "id": mid,
            "embeddings": {"emb": rng.normal(0.3 if mid <= 20 else -0.3, 1.0, 16).astype(np.float32)},
            "category": "alpha" if mid <= 20 else "beta",
        }
        for mid in range(1, 41)
    }
    vi.simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1, beta=4.0, head="linear_svm")
    assert len(seen) > SMART_WINDOW
    assert [n for n, _ in seen] == [min(i + 1, SMART_WINDOW) for i in range(len(seen))]
    assert all(b == 4.0 for _, b in seen)


def test_cost_arm_window_is_priced_and_cut_like_the_app_before_5134(monkeypatch):
    """The ``cost`` arm's Smart window: priced at SMART_INCLUSION, at each step's ``smart_cut``.

    The app's light before #5134, kept as an arm.  Run at Inclusion 3, so the
    reporting line and the Smart cut are different lines wherever the step has a
    fold-anchored fit to re-cut.
    """
    pytest.importorskip("torch")
    import vtscore.eval.voting_iterations as vi

    windows: list[tuple[float, list[float]]] = []
    cuts: list[tuple[float, float, float]] = []
    real_costs = vi._labelset_error_costs
    real_cut = vi.smart_cut

    def spy_costs(model_steps, good, bad, clips, inclusion, **kwargs):
        windows.append((inclusion, [thr for _, thr in model_steps]))
        return real_costs(model_steps, good, bad, clips, inclusion, **kwargs)

    def spy_cut(served, served_inclusion, recut):
        out = real_cut(served, served_inclusion, recut)
        cuts.append((served, served_inclusion, out))
        return out

    monkeypatch.setattr(vi, "_labelset_error_costs", spy_costs)
    monkeypatch.setattr(vi, "smart_cut", spy_cut)

    rng = np.random.RandomState(0)
    medias = {}
    for mid in range(1, 41):
        category = "alpha" if mid <= 20 else "beta"
        centre = 0.3 if category == "alpha" else -0.3
        medias[mid] = {
            "id": mid,
            "embeddings": {"emb": rng.normal(centre, 1.0, 16).astype(np.float32)},
            "category": category,
        }

    # The Inclusion arm: a set balance (the default arm's, #4413) would draw the line instead. The
    # plain SVM's head: on these 40 points it gives the fold-anchored fit a cut to move, which is the
    # mechanism under test; the head itself is not (#4785 changed the single-vector default).
    vi.simulate_voting_iterations(
        medias, "alpha", seed=42, inclusion=3, calibrate_count=1, beta="off", head="linear_svm", smart_signal="cost"
    )

    assert windows
    assert len(windows) == len(cuts)
    assert all(inclusion == SMART_INCLUSION for inclusion, _ in windows)
    # The last SMART_WINDOW steps, which are the app's last SMART_WINDOW models
    # because every step trains one. The sync gate cannot see this slice (#5005).
    assert len(windows) > SMART_WINDOW
    assert [len(window) for _, window in windows] == [min(i + 1, SMART_WINDOW) for i in range(len(windows))]
    assert all(served_inclusion == 3 for _, served_inclusion, _ in cuts)
    # Each step's window ends with the cut ``smart_cut`` chose for that step.
    assert [window[-1] for _, window in windows] == [out for *_, out in cuts]
    # And the re-cut is real: at least one step is scored off its reporting line.
    assert any(out != served for served, _, out in cuts)
