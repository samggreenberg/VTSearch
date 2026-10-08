"""What the Smart indicator prices a model's error at (issue #4243).

Smart re-scores its window of recent models against the current labelset.  Every
cost in that window is ``FPR + FNR`` at :data:`~vtscore.detectors.cost_trend.SMART_INCLUSION`,
measured at each model's own cut for that inclusion
(:func:`~vtscore.detectors.cost_trend.smart_cut`), whatever Inclusion - or, once
#4224 lands, precision floor - set the line the user was shown.  A window scored at
the served lines would mix cut rules whenever the line changes rule mid-run, and
a rise in cost reads green.

Pinned three ways: the rule itself, the app's progress cache, and the eval
harness's Smart window, which has to price the way the app does for a study of
the phase machine to mean anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vtscore.detectors.labeling_progress as lp
from vtscore.detectors.cost_trend import SMART_INCLUSION, smart_cut
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


class TestProgressCacheScoresAtTheSmartCut:
    """The app's Smart series, through the same ``_eval_cached_models`` the status reads."""

    N_VOTES = 8

    def _series(self, threshold: float, smart_threshold: float | None = None, **inclusion: int):
        """The Smart series after a sort per vote, each model served at *threshold*."""
        pytest.importorskip("torch")
        clips = _clips(40)
        history = [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(self.N_VOTES)]
        good: dict[int, None] = {}
        bad: dict[int, None] = {}
        lp.clear_progress_cache()
        for mid, label, _ in history:
            (good if label == "good" else bad)[mid] = None
            if good and bad:
                kwargs = {} if smart_threshold is None else {"smart_threshold": smart_threshold}
                lp.inject_live_model(dict(good), dict(bad), _linear_model(), threshold, **kwargs)
        series = lp.calculate_error_cost_over_time(clips, history, good, bad, **inclusion)
        cached, complete = lp.cached_indicator_history("smart", clips, history, good, bad, **inclusion)
        assert complete
        assert cached == series
        assert series
        return series

    def test_scored_at_the_smart_cut_not_the_served_line(self):
        # Served where nothing clears it, re-cut where everything does: every
        # negative is a false alarm and nothing is missed.
        for entry in self._series(NO_GOOD_THRESHOLD, smart_threshold=0.0):
            assert (entry["fpr"], entry["fnr"]) == (1.0, 0.0)
            assert entry["error_cost"] == 1.0

    def test_the_smart_cut_defaults_to_the_served_line(self):
        for entry in self._series(NO_GOOD_THRESHOLD):
            assert (entry["fpr"], entry["fnr"]) == (0.0, 1.0)

    def test_priced_at_the_smart_inclusion_whatever_inclusion_is_passed(self):
        # Inclusion 4 would price this all-miss model at 16; Smart prices it at 1.
        # The argument is deprecated and ignored (#4361), so it says so.
        with pytest.warns(DeprecationWarning, match="inclusion_value"):
            series = self._series(NO_GOOD_THRESHOLD, inclusion_value=4)
        for entry in series:
            assert entry["fnr"] == 1.0
            assert entry["error_cost"] == 1.0


def test_harness_window_is_priced_and_cut_like_the_app(monkeypatch):
    """The harness's Smart window: priced at SMART_INCLUSION, at each step's ``smart_cut``.

    Run at Inclusion 3, so the reporting line and the Smart cut are different
    lines wherever the step has a fold-anchored fit to re-cut.
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

    # The Inclusion arm: a set balance (the default arm's, #4413) would draw the line instead.
    vi.simulate_voting_iterations(medias, "alpha", seed=42, inclusion=3, calibrate_count=1, beta="off")

    assert windows
    assert len(windows) == len(cuts)
    assert all(inclusion == SMART_INCLUSION for inclusion, _ in windows)
    assert all(served_inclusion == 3 for _, served_inclusion, _ in cuts)
    # Each step's window ends with the cut ``smart_cut`` chose for that step.
    assert [window[-1] for _, window in windows] == [out for *_, out in cuts]
    # And the re-cut is real: at least one step is scored off its reporting line.
    assert any(out != served for served, _, out in cuts)
