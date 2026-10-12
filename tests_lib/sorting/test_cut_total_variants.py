"""#4798's candidate cuts: the class model's cut with its positives total held."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.arms_cut_total import CUT_VARIANTS, chances_sum, variant_counts, variant_threshold
from vtscore.training.thresholds import ClassScoreModel, LabelsLine, corpus_cut


def _diffuse_line(total: float = 3.0) -> tuple[LabelsLine, np.ndarray]:
    """Three items at chance 0.9 above 5,000 at 0.05: a 2-part total of *total*, chances summing to ~253."""
    s = np.linspace(0.99, 0.01, 5003)
    p = np.r_[[0.9, 0.9, 0.9], np.full(5000, 0.05)]
    model = ClassScoreModel(2.0, -1.0, 1.0, 4, 4)
    line = LabelsLine(model, 0.001, None, total / s.size, s, p, 1.0)
    return line, s


class TestTheDiffuseCase:
    def test_todays_cut_runs_to_the_end(self):
        line, s = _diffuse_line()
        assert variant_counts(line, s, 4.0)["class"] == s.size
        assert variant_threshold("class", line, 4.0) == corpus_cut(line.unvoted_posteriors, s, 3.0, 4.0)

    # Not ``capsum``: its cap is the counted bound, which an evenly spread toy corpus never tightens
    # (it needs a bulk concentrated below the Goods); the sessions price it.
    @pytest.mark.parametrize("name", ["ptotal", "pmin", "captotal", "ncap"])
    def test_each_candidate_keeps_the_standouts(self, name):
        line, s = _diffuse_line()
        k = variant_counts(line, s, 4.0)[name]
        assert 1 <= k <= 20, (name, k)

    def test_the_count_cap_is_the_count_lines_multiple(self):
        line, s = _diffuse_line(total=3.0)
        assert variant_counts(line, s, 4.0)["ncap"] <= math.ceil(4.0**0.708 * 3.0)

    def test_the_chances_sum_is_recorded(self):
        line, _s = _diffuse_line()
        assert chances_sum(line) == pytest.approx(0.9 * 3 + 0.05 * 5000)


class TestWhereTheTotalHolds:
    def test_concentrated_chances_leave_every_candidate_at_todays_cut(self):
        """Chances that sum to about the 2-part total: nothing to hold, every cut the same."""
        s = np.linspace(0.99, 0.01, 1003)
        p = np.r_[[0.95, 0.9, 0.85], np.full(1000, 1e-5)]
        line = LabelsLine(ClassScoreModel(2.0, -1.0, 1.0, 10, 40), 0.003, None, 3.0 / s.size, s, p, 1.0)
        counts = variant_counts(line, s, 1.0)
        assert len(set(counts.values())) == 1, counts

    def test_no_corpus_side_counts_minus_one(self):
        line = LabelsLine(ClassScoreModel(2.0, -1.0, 1.0, 4, 4), 0.01)
        assert set(variant_counts(line, np.array([0.5, 0.4]), 4.0).values()) == {-1}
        assert variant_threshold("ptotal", None, 4.0) is None

    def test_an_unknown_candidate_is_refused(self):
        line, _s = _diffuse_line()
        with pytest.raises(ValueError, match="cut variant"):
            variant_threshold("nope", line, 4.0)
        assert "class" in CUT_VARIANTS


class TestARunawayTotal:
    """The other way round: a 2-part total of hundreds over chances that sum to a few."""

    def _line(self):
        s = np.linspace(0.99, 0.01, 5003)
        p = np.r_[[0.9, 0.8, 0.7], np.full(5000, 0.0007)]
        return LabelsLine(ClassScoreModel(2.0, -1.0, 1.0, 10, 60), 0.1, None, 500.0 / s.size, s, p, 1.0), s

    def test_todays_cut_runs_deep_and_ptotal_with_it(self):
        line, s = self._line()
        counts = variant_counts(line, s, 4.0)
        assert counts["class"] > 1000 and counts["ptotal"] > 1000, counts

    @pytest.mark.parametrize("name", ["pchances", "pmin"])
    def test_the_fits_own_total_keeps_the_standouts(self, name):
        line, s = self._line()
        assert 1 <= variant_counts(line, s, 4.0)[name] <= 20
