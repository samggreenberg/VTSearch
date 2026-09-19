"""The export layer: prevalence as a parameter, negatives per class (#3986, #3987).

`SCALE_PREVALENCE`, `SCALE_N_POS` and `SCALE_N_NEG` used to be fixed when a pile
was built. Over the embedded corpus they are filters, and these pin the three
properties that make the filter trustworthy: it reproduces the designated draw,
it refuses what the corpus cannot supply, and its default pool is the per-class
one rather than the barren shared pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def qe():
    for p in (_PILE_DIR, _PILE_DIR.parent / "calibration"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import quarry_export

    return quarry_export


def _members():
    """Three cells over a tiny corpus, in the shape `load_membership` returns."""
    m = {}
    for i in range(1, 11):  # book@small positives
        m[i] = {"cells": ["book@small"], "evaluable": ["book@small", "bus@small"]}
    for i in range(11, 21):  # hold bus, so scorable as book negatives
        m[i] = {"cells": ["bus@small"], "evaluable": ["book@small", "bus@small"]}
    for i in range(21, 41):  # hold nothing in C
        m[i] = {"cells": [], "evaluable": ["book@small", "bus@small"]}
    return m


def test_negatives_for_is_the_prevalence_identity(qe):
    assert qe.negatives_for(100, 0.01) == 9900
    assert qe.negatives_for(100, 0.5) == 100
    assert qe.negatives_for(1, 0.001) == 999


def test_per_class_pool_keeps_images_holding_another_class(qe):
    """A negative for `book` need only lack `book` -- #3986's whole argument."""
    pos, neg = qe.candidates(_members(), "book@small", "per_class")
    assert len(pos) == 10
    assert len(neg) == 30, "20 barren + 10 holding bus"
    assert 15 in neg, "an image holding bus is a legitimate book negative"


def test_shared_pool_is_the_barren_one(qe):
    """The old shape, kept for arms compared against a published number."""
    _, neg = qe.candidates(_members(), "book@small", "shared")
    assert len(neg) == 20
    assert all(i >= 21 for i in neg), "every shared-pool negative holds nothing in C"


def test_an_impossible_prevalence_is_refused_with_both_ways_out(qe):
    with pytest.raises(SystemExit) as err:
        qe.select(_members(), "book@small", 10, 0.001)
    msg = str(err.value)
    assert "9,990 negatives" in msg and "30 are available" in msg
    assert "SHORT 9,960" in msg, "name the shortfall: 'drop to 3 positives' reads as a small deficit"
    assert "999 more of them" in msg, "and that adding positives makes it worse, not better"
    assert "drop to" in msg and "raise pi" in msg, "a refusal must name the fix, not just the problem"


def test_asking_for_more_positives_than_exist_is_refused(qe):
    with pytest.raises(SystemExit, match="asked for 50"):
        qe.select(_members(), "book@small", 50, 0.5)


def test_the_draw_is_stable_when_the_corpus_changes(qe):
    """Hash-ordered, so adding an image changes only that image's membership.

    `rng.sample` is deterministic and still reshuffles everything when the pool
    moves, which silently retires images a human already reviewed (#3726).
    """
    base = _members()
    first = qe.select(base, "book@small", 5, 0.5)["positives"]
    grown = dict(base)
    grown[999] = {"cells": ["book@small"], "evaluable": ["book@small"]}
    second = qe.select(grown, "book@small", 5, 0.5)["positives"]
    assert len(set(first) & set(second)) >= 4, "a new candidate must not reshuffle the draw"


def test_prevalence_is_actually_achieved(qe):
    sel = qe.select(_members(), "book@small", 3, 0.1)
    got = sel["n_positives"] / (sel["n_positives"] + sel["n_negatives"])
    assert abs(got - 0.1) < 1e-9
