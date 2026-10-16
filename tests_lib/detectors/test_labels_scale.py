"""Issue #4910's fold scale: the labels line reads the folds' held-out scores on the final model's scale.

An experiment knob, not app behaviour: unset is the app's raw pooling, so the
default arm cannot drift.  The shadow fits the line on another scale beside the
arm's own, on the same head, and draws nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    FOLD_SCALES,
    fold_transfer_clipped,
    fold_transfer_shift,
    transfer_fold_orderings,
)
from vtscore.training.thresholds.anchored import rank_transfer


def _logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class TestTransferFoldOrderings:
    def test_each_score_is_its_rank_transfer(self):
        """The plain transfer is the fold-anchored cut's own, applied to every held-out score."""
        rng = np.random.default_rng(0)
        hay = [rng.beta(1, 8, 500), rng.beta(1, 6, 400)]
        final = rng.beta(1, 10, 600)
        orderings = [
            (rng.uniform(0, 1, 7).astype(np.float32), [1, 0, 1, 0, 0, 1, 0]),
            (list(rng.uniform(0, 1, 5)), [0.0, 0.0, 1.0, 1.0, 0.0]),
        ]
        out = transfer_fold_orderings(orderings, hay, final)
        for (s, y), (t, yy), h in zip(orderings, out, hay):
            assert np.array_equal(np.asarray(y, dtype=np.float64), yy)
            want = [rank_transfer(float(v), h, final) for v in np.asarray(s, dtype=np.float64)]
            assert np.array_equal(t, np.asarray(want))

    def test_a_monotone_rescale_of_the_same_items_is_undone(self):
        """Two models related by a monotone map: a fold's score lands where the final model scores the same rank."""
        rng = np.random.default_rng(1)
        fold_logits = rng.normal(-3.0, 1.0, 4000)
        hay = _sigmoid(fold_logits)
        final = _sigmoid(2.0 * fold_logits + 1.0)
        held = _sigmoid(rng.uniform(-5.0, -1.0, 50))
        ((t, _),) = transfer_fold_orderings([(held, np.zeros(50))], [hay], final)
        assert np.max(np.abs(_logit(t) - (2.0 * _logit(held) + 1.0))) < 0.05

    def test_saturates_past_the_haystack_unless_tails(self):
        hay = np.array([0.1, 0.2, 0.3, 0.4])
        final = np.array([0.5, 0.6, 0.7, 0.8])
        held = np.array([0.9, 0.05, 0.25])
        ((plain, _),) = transfer_fold_orderings([(held, [1, 0, 0])], [hay], final)
        assert plain[0] == 0.8 and plain[1] == 0.5
        ((tails, _),) = transfer_fold_orderings([(held, [1, 0, 0])], [hay], final, tails=True)
        assert tails[0] == pytest.approx(_sigmoid(_logit(0.8) + _logit(0.9) - _logit(0.4)))
        assert tails[1] == pytest.approx(_sigmoid(_logit(0.5) + _logit(0.05) - _logit(0.1)))
        # Inside the haystack's range the two agree.
        assert tails[2] == plain[2]

    def test_what_has_nothing_to_map_onto_keeps_its_score(self):
        hay = [np.array([0.1, 0.2, 0.3])]
        orderings = [(np.array([0.15, -1.0]), [1, 0]), (np.array([0.5, 0.6]), [1, 0])]
        out = transfer_fold_orderings(orderings, hay, np.array([0.4, 0.5, 0.6]))
        # The unscorable sentinel stays where class_score_model drops it; the
        # second fold has no haystack and is not moved.
        assert out[0][0][1] == -1.0 and out[0][0][0] != 0.15
        assert np.array_equal(out[1][0], [0.5, 0.6])
        # No final haystack: nothing moves.
        same = transfer_fold_orderings(orderings, hay, np.array([]))
        assert all(np.array_equal(a[0], np.asarray(b[0])) for a, b in zip(same, orderings))

    def test_gate_readings(self):
        hay = [np.array([0.1, 0.2, 0.3])]
        orderings = [(np.array([0.5, 0.15, 0.05, -1.0]), [1, 0, 1, 0])]
        side, good = fold_transfer_clipped(orderings, hay)
        assert side.tolist() == [1, 0, -1] and good.tolist() == [True, False, True]
        moved = transfer_fold_orderings(orderings, hay, np.array([0.1, 0.2, 0.3]))
        shift = fold_transfer_shift(orderings, moved)
        assert shift.size == 3 and shift[0] == pytest.approx(abs(_logit(0.3) - _logit(0.5)))

    def test_scales(self):
        assert FOLD_SCALES == ("raw", "quantile", "quantile_tails")


def _run(**kw):
    from tests_lib.detectors.test_safe_gmm_variant_rows import _run_safe_uncached

    rows = _run_safe_uncached("max_patch", beta=None, max_steps=16, **kw)
    return [r for r in rows if r["pool_variant"] == "max" and r["gmm_variant"] == ""]


_SHADOW_COLUMNS = (
    "shadow_train_threshold",
    "shadow_threshold",
    "shadow_n_flagged",
    "shadow_fbeta",
    "shadow_separation",
    "xfer_dlogit_med",
    "xfer_dlogit_p90",
    "xfer_clip_good",
    "xfer_dmu_pos",
    "xfer_dmu_neg",
    "xfer_dsigma",
)


def _same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))


def _compared(row):
    """The columns a run's outcome is: every one but the shadow's and the wall clocks."""
    return [c for c in row if c not in _SHADOW_COLUMNS and not c.endswith("_seconds")]


class TestLabelsScaleArm:
    def test_refuses_what_it_cannot_run(self):
        from tests_lib.detectors.test_safe_gmm_variant_rows import _run_safe_uncached

        with pytest.raises(ValueError, match="labels_scale must be one of"):
            _run_safe_uncached("max_patch", beta=None, max_steps=2, labels_scale="rank")
        with pytest.raises(ValueError, match="labels_scale_shadow must be one of"):
            _run_safe_uncached("max_patch", beta=None, max_steps=2, labels_scale_shadow="rank")
        # The Inclusion arm draws no labels line for a scale to move.
        with pytest.raises(ValueError, match="need safe_thresholds, a balance"):
            _run_safe_uncached("max_patch", max_steps=2, labels_scale="quantile")

    def test_the_shadow_draws_nothing(self):
        """The shadow's columns are the only difference: same picks, same line, same objective."""
        plain, shadowed = _run(), _run(labels_scale_shadow="quantile")
        assert len(plain) == len(shadowed)
        for a, b in zip(plain, shadowed):
            diff = [c for c in _compared(a) if not _same(a[c], b[c])]
            assert not diff, diff
        drawn = [r for r in shadowed if np.isfinite(r["train_prevalence"])]
        assert drawn, "the default arm never drew the labels line"
        assert all(np.isfinite(r["shadow_fbeta"]) or np.isnan(r["fbeta"]) for r in drawn)
        assert all(r["shadow_n_flagged"] >= 0 for r in drawn)
        # The gate reads the folds' held-out scores; a step whose folds fell back
        # (the in-sample stand-in draws its line) has none to move.
        folded = [r for r in drawn if np.isfinite(r["n_cal_scores"]) and r["n_cal_scores"] > 0]
        assert folded and all(np.isfinite(r["xfer_dlogit_med"]) for r in folded)
        assert all(np.isnan(r["xfer_dlogit_med"]) for r in drawn if r not in folded)
        assert all(np.isnan(r["shadow_fbeta"]) and r["shadow_n_flagged"] == -1 for r in plain)

    def test_raw_is_the_app(self):
        plain, raw = _run(), _run(labels_scale="raw")
        assert len(plain) == len(raw)
        assert all(_same(a[c], b[c]) for a, b in zip(plain, raw) for c in _compared(a))

    def test_the_arm_is_the_control_s_shadow_on_the_same_head(self):
        """At the first step the line is drawn both runs hold the same head; there each reads the other as its shadow."""
        control = _run(labels_scale_shadow="quantile")
        arm = _run(labels_scale="quantile", labels_scale_shadow="raw")
        i = next(k for k, r in enumerate(control) if np.isfinite(r["train_prevalence"]))
        c, a = control[i], arm[i]
        assert (c["seed"], c["t"]) == (a["seed"], a["t"])
        assert a["train_threshold"] == c["shadow_train_threshold"]
        assert c["train_threshold"] == a["shadow_train_threshold"]
        assert _same(a["fbeta"], c["shadow_fbeta"]) and _same(c["fbeta"], a["shadow_fbeta"])
        assert a["n_flagged"] == c["shadow_n_flagged"] and c["n_flagged"] == a["shadow_n_flagged"]
        # The gate reads the same transfer either way round.
        assert all(_same(a[k], c[k]) for k in _SHADOW_COLUMNS if k.startswith("xfer_"))
