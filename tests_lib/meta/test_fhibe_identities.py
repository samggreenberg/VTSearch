"""The FHIBE identity sampler (#4699): who a study can run, and that every arm runs the same people."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fhibe" / "identities.py"


@pytest.fixture(scope="module")
def ids():
    spec = importlib.util.spec_from_file_location("_fhibe_identities", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def test_admission_is_the_stratified_split_s_voting_half(ids):
    """Six photos keep three to vote on, so four examples need seven."""
    counts = Counter({"a": 2, "b": 6, "c": 7, "d": 1})
    assert ids.eligible(counts, 1, 0.5) == {"a", "b", "c"}
    assert ids.eligible(counts, 3, 0.5) == {"b", "c"}
    assert ids.eligible(counts, 4, 0.5) == {"c"}


def test_a_lone_photo_is_never_admitted(ids):
    """One positive cannot sit on both sides of the split."""
    assert ids.eligible(Counter({"a": 1}), 1, 0.5) == set()


def test_every_cell_must_admit_an_identity(ids):
    """The crop arms lose people whose faces MTCNN missed; the photo arms run the same people anyway."""
    pools = {"photo": {"a", "b", "c"}, "face": {"a", "c"}}
    chosen, common = ids.sample(pools, None, seed=0)
    assert chosen == ["a", "c"] and common == {"a", "c"}


def test_the_sample_is_seeded_and_sorted(ids):
    pools = {"photo": {f"s{i:03d}" for i in range(100)}}
    a, _ = ids.sample(pools, 10, seed=1)
    b, _ = ids.sample(pools, 10, seed=1)
    assert a == b == sorted(a)
    assert len(a) == 10


def test_strata_parse_as_inclusive_bands(ids):
    assert ids.parse_strata("2-4,5-6,7-") == [("2-4", 2, 4), ("5-6", 5, 6), ("7-", 7, float("inf"))]


@pytest.mark.parametrize("spec", ["2-4,4-6", "5-6,2-4", "4-2", "two-4", "4"])
def test_bad_strata_are_refused(ids, spec):
    with pytest.raises(ValueError):
        ids.parse_strata(spec)


def test_each_band_draws_its_own_identities(ids):
    """#4731: a band of few photos and a band of many, drawn apart, from people every cell admits."""
    counts = Counter({f"s{i:03d}": 2 + i % 8 for i in range(200)})
    pools = {"photo": set(counts), "face": set(counts) - {"s002"}}
    strata = ids.parse_strata("2-4,5-6,7-")
    chosen, band_of, common = ids.sample_strata(pools, counts, strata, 5, seed=0)
    assert len(chosen) == 15 and chosen == sorted(chosen) and "s002" not in chosen
    for c in chosen:
        lo, hi = {"2-4": (2, 4), "5-6": (5, 6), "7-": (7, 99)}[band_of[c]]
        assert lo <= counts[c] <= hi
    # A band's draw does not depend on the bands after it.
    first, _, _ = ids.sample_strata(pools, counts, strata[:1], 5, seed=0)
    assert set(first) == {c for c in chosen if band_of[c] == "2-4"}
