"""#3546: Autopilot samples at a target pick precision; the old Inclusion origin stays a harness arm.

Since #3546 the app's acquisition cut under a balance is the score where the
labels line's corpus posterior falls below ``ACQUISITION_TARGET_PRECISION``
(:func:`~vtscore.training.thresholds.target_precision_threshold`), and the
harness's default arm resolves to it.  Pinned here: the cut is where it claims
to be, the knobs name one cut, the default arm is the app's, and ``"off"``
restores the line - 4 offset cut.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.voting_iterations import (
    ACQ_ORIGINS,
    _check_acquisition_arm,
    resolve_acquisition_target,
    simulate_voting_iterations,
)
from vtscore.training.thresholds import ACQUISITION_TARGET_PRECISION, target_precision_threshold
from vtscore.training.thresholds.labels_line import ClassScoreModel, LabelsLine

from .test_max_patch_style import _planted_dataset


def _line(n_pos: int = 30, n_neg: int = 970, seed: int = 0) -> LabelsLine:
    """A labels line over a corpus whose positives sit above a broad bulk."""
    rng = np.random.default_rng(seed)
    pos = 1 / (1 + np.exp(-rng.normal(2.0, 0.6, n_pos)))
    neg = 1 / (1 + np.exp(-rng.normal(-2.0, 0.9, n_neg)))
    scores = np.sort(np.concatenate([pos, neg]))[::-1]
    model = ClassScoreModel(mu_pos=2.0, mu_neg=-0.5, sigma=0.6, n_pos=8, n_neg=8)
    return LabelsLine(model=model, prevalence=n_pos / (n_pos + n_neg), unvoted_scores=scores)


class TestTheTargetPrecisionCut:
    def test_it_is_one_of_the_unvoted_scores(self):
        line = _line()
        cut = target_precision_threshold(line, 0.5)
        assert line.unvoted_scores is not None
        assert cut is not None and cut in set(line.unvoted_scores.tolist())

    def test_a_higher_target_samples_higher_up(self):
        line = _line()
        found = [target_precision_threshold(line, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
        cuts = [c for c in found if c is not None]
        assert len(cuts) == len(found)
        assert cuts == sorted(cuts), cuts
        assert cuts[0] < cuts[-1]

    def test_it_is_where_the_posterior_crosses_the_target(self):
        from vtscore.training.thresholds.labels_line import corpus_posteriors

        line = _line()
        assert line.unvoted_scores is not None
        post = corpus_posteriors(line.model, line.unvoted_scores)
        cut = target_precision_threshold(line, 0.5)
        i = int(np.flatnonzero(line.unvoted_scores == cut)[0])
        assert post[i] < 0.5 and (i == 0 or post[i - 1] >= 0.5)

    def test_no_line_or_no_scores_is_no_cut(self):
        assert target_precision_threshold(None, 0.5) is None
        empty = LabelsLine(model=_line().model, prevalence=0.01, unvoted_scores=np.zeros(0))
        assert target_precision_threshold(empty, 0.5) is None


class TestTheKnobsNameOneCut:
    def test_the_origins(self):
        assert ACQ_ORIGINS == ("line", "inclusion")
        with pytest.raises(ValueError, match="acq_origin must be one of"):
            _check_acquisition_arm(-4, None, None, "corpus")

    def test_the_inclusion_origin_needs_an_offset_to_move(self):
        with pytest.raises(ValueError, match="nonzero acq_inclusion_offset"):
            _check_acquisition_arm(0, None, None, "inclusion")
        _check_acquisition_arm(-4, None, None, "inclusion")

    @pytest.mark.parametrize(
        ("pct", "cross"), [(0.98, None), (None, 0.5)], ids=("with-a-rank-pin", "with-the-argmax-factor")
    )
    def test_a_target_precision_and_another_cut_conflict(self, pct, cross):
        with pytest.raises(ValueError, match="acq_target_p names the acquisition cut"):
            _check_acquisition_arm(0, pct, cross, "line", 0.5)

    def test_the_offset_is_the_targets_fallback_not_a_conflict(self):
        _check_acquisition_arm(-4, None, None, "line", 0.5)

    @pytest.mark.parametrize("p", (0.0, 1.0, -0.1, 1.5, "on"))
    def test_a_target_is_a_share_or_off(self, p):
        with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
            _check_acquisition_arm(0, None, None, "line", p)

    def test_the_off_factor_is_not_a_competing_cut(self):
        _check_acquisition_arm(0, None, "off", "line", 0.5)


class TestTheDefaultIsTheApps:
    def test_under_a_balance_the_target_is_the_shipped_one(self):
        assert ACQUISITION_TARGET_PRECISION == 0.5
        assert resolve_acquisition_target(None, 1.0) == ACQUISITION_TARGET_PRECISION
        assert resolve_acquisition_target(None, 0.25) == ACQUISITION_TARGET_PRECISION

    def test_off_and_the_inclusion_arm_are_the_offset_cut(self):
        assert resolve_acquisition_target("off", 1.0) is None
        assert resolve_acquisition_target(None, None) is None

    def test_a_pinned_target_is_that_target(self):
        assert resolve_acquisition_target(0.25, 1.0) == 0.25


class TestTheArmsRun:
    """Each arm runs end to end under the balance and moves the selector's cut."""

    def _acq(self, **kwargs) -> list[tuple[int, float]]:
        # The planted fixture test_acq_inclusion.py uses: big enough for the
        # fold-anchored scale to resolve offsets (a toy pool saturates it, and
        # every origin then cuts in the same place).
        medias, _ = _planted_dataset(n_per_cat=40, seed=0)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=3,
            dataset_name="synthetic",
            max_steps=30,
            head="linear",
            style="whole_image",
            emit_calibration_metrics=True,
            beta=1.0,
            **kwargs,
        )
        # Acquisition is the Train side: a row with no Bad yet (the Goods'
        # centroid's, #4643) has no head and so no acquisition cut.
        base = [
            r
            for r in rows
            if not str(r.get("gmm_variant") or "").strip()
            and str(r.get("pool_variant") or "") in ("", "max")
            and not str(r.get("schedule") or "").strip()
            and r["n_bad"] > 0
        ]
        assert base and all("acq_threshold" in r for r in base)
        return [(int(r["t"]), float(r["acq_threshold"])) for r in base]

    def test_the_default_arm_samples_at_the_target(self):
        default = self._acq()
        offset = self._acq(acq_target_p="off")
        assert default and offset
        assert all(math.isfinite(a) for _t, a in default)
        assert [a for _t, a in default] != [a for _t, a in offset]

    def test_naming_the_shipped_target_is_the_default(self):
        assert self._acq() == self._acq(acq_target_p=ACQUISITION_TARGET_PRECISION)

    def test_the_inclusion_origin_moves_the_offset_cut(self):
        line = self._acq(acq_target_p="off")
        old = self._acq(acq_target_p="off", acq_origin="inclusion")
        assert [a for _t, a in old] != [a for _t, a in line]
