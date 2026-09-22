"""The detection-cap benchmark's sampling and labelling (#3908).

No detection runs here -- the GRID sweep is the measurement.  What is checked is
the part that decides *which* pages get priced, because a cost table built from a
uniform sample of DocMarks would be a table about 2 MP pages and would miss the
63 MP tail that the cap exists to bound.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield importlib.import_module("bench_detect_cap")
    finally:
        sys.path.remove(str(_DOCMARKS))


class TestCapLabel:
    def test_zero_and_negative_read_as_uncapped(self, bench):
        assert bench.cap_label(0) == "uncapped"
        assert bench.cap_label(-1) == "uncapped"

    def test_megapixels_drop_trailing_zeros(self, bench):
        assert bench.cap_label(2_000_000) == "2MP"
        assert bench.cap_label(500_000) == "0.5MP"
        assert bench.cap_label(250_000) == "0.25MP"


class TestStratifiedSample:
    def _sizes(self, n):
        # 0.1 MP .. n/10 MP, shuffled so the function cannot rely on input order
        vals = [((i + 1) / 10.0, f"p{i:04d}") for i in range(n)]
        return vals[1::2] + vals[0::2]

    def test_every_page_when_the_sample_is_larger_than_the_corpus(self, bench):
        got = bench.stratified_sample(self._sizes(5), 50)
        assert sorted(got) == [f"p{i:04d}" for i in range(5)]

    def test_the_largest_pages_are_always_kept(self, bench):
        got = set(bench.stratified_sample(self._sizes(500), 40))
        biggest = {f"p{i:04d}" for i in range(500 - bench.N_LARGEST, 500)}
        assert biggest <= got, "the tail is the reason the cap exists; a stride must not skip it"

    def test_asks_for_n_and_gets_about_n(self, bench):
        got = bench.stratified_sample(self._sizes(500), 40)
        assert len(got) == len(set(got)) and 36 <= len(got) <= 40

    def test_spans_the_distribution_rather_than_clustering(self, bench):
        got = bench.stratified_sample(self._sizes(500), 40)
        idx = sorted(int(p[1:]) for p in got)
        # something from the bottom decile, something from the middle
        assert idx[0] < 50 and any(200 <= i <= 300 for i in idx)

    def test_a_degenerate_corpus_does_not_crash(self, bench):
        assert bench.stratified_sample([], 10) == []
        assert bench.stratified_sample([(1.0, "only")], 10) == ["only"]
