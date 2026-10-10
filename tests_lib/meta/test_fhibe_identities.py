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
