"""#3546's two acquisition arms: the old Inclusion origin and a target pick precision.

Both are harness-only arms (the app still samples at the line - 4 offset), so
what is pinned here is that each names the cut it claims to and that the
default arm is untouched by either knob.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.voting_iterations import (
    ACQ_ORIGINS,
    _check_acquisition_arm,
    simulate_voting_iterations,
    target_precision_cut,
)
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
        cut = target_precision_cut(line, 0.5)
        assert line.unvoted_scores is not None
        assert cut is not None and cut in set(line.unvoted_scores.tolist())

    def test_a_higher_target_samples_higher_up(self):
        line = _line()
        found = [target_precision_cut(line, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
        cuts = [c for c in found if c is not None]
        assert len(cuts) == len(found)
        assert cuts == sorted(cuts), cuts
        assert cuts[0] < cuts[-1]

    def test_it_is_where_the_posterior_crosses_the_target(self):
        from vtscore.training.thresholds.labels_line import corpus_posteriors

        line = _line()
        assert line.unvoted_scores is not None
        post = corpus_posteriors(line.model, line.unvoted_scores)
        cut = target_precision_cut(line, 0.5)
        i = int(np.flatnonzero(line.unvoted_scores == cut)[0])
        assert post[i] < 0.5 and (i == 0 or post[i - 1] >= 0.5)

    def test_no_line_or_no_scores_is_no_cut(self):
        assert target_precision_cut(None, 0.5) is None
        empty = LabelsLine(model=_line().model, prevalence=0.01, unvoted_scores=np.zeros(0))
        assert target_precision_cut(empty, 0.5) is None


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
        ("offset", "pct", "cross"),
        [(-4, None, None), (0, 0.98, None), (0, None, 0.5)],
        ids=("with-the-offset", "with-a-rank-pin", "with-the-argmax-factor"),
    )
    def test_a_target_precision_replaces_every_other_cut(self, offset, pct, cross):
        with pytest.raises(ValueError, match="acq_target_p replaces the acquisition cut"):
            _check_acquisition_arm(offset, pct, cross, "line", 0.5)

    @pytest.mark.parametrize("p", (0.0, 1.0, -0.1, 1.5))
    def test_a_target_is_a_share(self, p):
        with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
            _check_acquisition_arm(0, None, None, "line", p)

    def test_the_off_factor_is_not_a_competing_cut(self):
        _check_acquisition_arm(0, None, "off", "line", 0.5)


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
        base = [
            r
            for r in rows
            if not str(r.get("gmm_variant") or "").strip()
            and str(r.get("pool_variant") or "") in ("", "max")
            and not str(r.get("schedule") or "").strip()
        ]
        assert base and all("acq_threshold" in r for r in base)
        return [(int(r["t"]), float(r["acq_threshold"])) for r in base]

    def test_the_target_precision_arm_moves_the_selectors_cut(self):
        control = self._acq()
        target = self._acq(acq_inclusion_offset=0, acq_target_p=0.5)
        assert control and target
        assert all(math.isfinite(a) for _t, a in target)
        assert [a for _t, a in target] != [a for _t, a in control]

    def test_the_inclusion_origin_moves_the_selectors_cut(self):
        control = self._acq()
        old = self._acq(acq_origin="inclusion")
        assert control and old
        assert [a for _t, a in old] != [a for _t, a in control]

    def test_the_default_is_unchanged_by_naming_it(self):
        assert self._acq() == self._acq(acq_origin="line", acq_target_p=None)
