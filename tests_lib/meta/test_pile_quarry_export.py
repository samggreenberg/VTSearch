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


def _banded():
    """One class at three bands, sharing one negative pool -- the cross-band shape.

    A positive of any band is evaluable **only** in its own cell, which is what
    `scale_core._evaluable` does: an image holding a large car is not a small-car
    negative.  Every barren image is evaluable in all three.
    """
    cells = ["car@small", "car@medium", "car@large"]
    m = {}
    nxt = 1
    for cell, n in zip(cells, (10, 14, 6)):
        for _ in range(n):
            m[nxt] = {"cells": [cell], "evaluable": [cell]}
            nxt += 1
    for _ in range(40):
        m[nxt] = {"cells": [], "evaluable": list(cells)}
        nxt += 1
    return m


class TestCrossBandTesting:
    """#4044: one arm, one FPR, an FNR per size."""

    def test_the_bands_are_read_off_the_membership(self, qe):
        assert qe.cells_of_class(_banded(), "car") == ["car@large", "car@medium", "car@small"]

    def test_the_bands_share_one_negative_pool(self, qe):
        """The premise of the whole breakdown: three FNRs are readable against a
        single FPR only if the images behind that FPR are the same for each."""
        ok, note = qe.check_shared_negatives(_banded(), "car", "per_class")
        assert ok, note
        assert "share 40 negatives" in note

    def test_a_broken_shared_pool_refuses_the_export(self, qe):
        """True by construction today, and a construction is exactly what stops
        being true without anyone editing the file that states it."""
        members = _banded()
        members[500] = {"cells": [], "evaluable": ["car@small"]}  # a negative for one band only
        sel = qe.select(members, "car@small", 5, 0.5)
        with pytest.raises(SystemExit, match="do NOT share a negative pool"):
            qe.add_test_bands(members, sel, None)

    def test_sibling_positives_ride_along_without_joining_the_positives(self, qe):
        """An image with a large car is not a `car@small` positive; folding it in
        would move the very FNR the breakdown keeps separate."""
        members = _banded()
        sel = qe.add_test_bands(members, qe.select(members, "car@small", 5, 0.5), None)
        assert sel["n_positives"] == 5
        assert sel["n_test_positives_by_band"] == {"medium": 14, "large": 6}
        assert set(sel["positives"]).isdisjoint(sel["test_positives_by_band"]["medium"])

    def test_a_named_subset_takes_only_those_bands(self, qe):
        members = _banded()
        sel = qe.add_test_bands(members, qe.select(members, "car@small", 5, 0.5), ["large"])
        assert list(sel["test_positives_by_band"]) == ["large"]

    def test_the_natural_mix_rides_along(self, qe):
        """The proportions "in the wild", which a train-side mix is set against."""
        members = _banded()
        sel = qe.add_test_bands(members, qe.select(members, "car@small", 5, 0.5), None)
        assert sel["natural_mix"] == pytest.approx({"small": 1 / 3, "medium": 14 / 30, "large": 0.2}, abs=1e-6)


class TestTrainSideMix:
    """#4044: training is not linear in its input, so each mix is a real arm."""

    def test_the_quota_follows_the_requested_shares(self, qe):
        sel = qe.select_mix(_banded(), "car", {"small": 1, "medium": 2, "large": 1}, 8, 0.5)
        assert sel["positives_by_band"] == {"small": 2, "medium": 4, "large": 2}
        assert sel["n_positives"] == 8

    def test_the_realised_size_is_exactly_what_was_asked(self, qe):
        """Largest-remainder: a mixed arm whose size drifts with the mix cannot
        be compared with the pure arms it exists to be compared against."""
        sel = qe.select_mix(_banded(), "car", {"small": 1, "medium": 1, "large": 1}, 15, 0.5)
        assert sel["n_positives"] == 15

    def test_natural_reads_the_shares_off_the_corpus(self, qe):
        sel = qe.select_mix(_banded(), "car", "natural", 15, 0.5)
        assert sel["requested_mix"]["medium"] == pytest.approx(14 / 30)

    def test_a_short_band_makes_the_set_smaller_never_a_different_mix(self, qe):
        """`large` has 6 of the 8 an equal-thirds 24 wants. Its two seats must NOT
        go to `small` and `medium`: that keeps the count round while quietly
        running a mix nobody asked for."""
        sel = qe.select_mix(_banded(), "car", {"small": 1, "medium": 1, "large": 1}, 24, 0.5)
        assert sel["positives_by_band"] == {"small": 8, "medium": 8, "large": 6}
        assert sel["n_positives"] == 22 < sel["n_positives_requested"]
        assert sel["realised_mix"]["large"] < sel["requested_mix"]["large"]

    def test_a_mix_draws_the_pure_cells_own_first_n(self, qe):
        """So a mix's small slice is a subset of `car@small`, not a second draw."""
        members = _banded()
        few = qe.select_mix(members, "car", {"small": 1}, 3, 0.5)
        more = qe.select_mix(members, "car", {"small": 1}, 6, 0.5)
        pure = qe.select(members, "car@small", 6, 0.5)
        assert set(few["positives"]) < set(more["positives"]) == set(pure["positives"])

    def test_every_bands_positives_are_excluded_from_the_negatives(self, qe):
        sel = qe.select_mix(_banded(), "car", "natural", 9, 0.25)
        assert sel["available_negatives"] == 40
        assert set(sel["negatives"]).isdisjoint(sel["positives"])

    def test_an_unknown_band_is_refused(self, qe):
        with pytest.raises(SystemExit, match="no such band"):
            qe.select_mix(_banded(), "car", {"enormous": 1}, 3, 0.5)

    def test_an_impossible_prevalence_is_refused_with_both_ways_out(self, qe):
        with pytest.raises(SystemExit) as err:
            qe.select_mix(_banded(), "car", "natural", 30, 0.001)
        assert "SHORT" in str(err.value) and "drop to" in str(err.value)

    def test_the_mix_spec_parses_both_spellings(self, qe):
        assert qe.parse_mix("natural") == "natural"
        assert qe.parse_mix("small=1,large=3") == {"small": 1.0, "large": 3.0}
        with pytest.raises(SystemExit, match="bad --train-mix"):
            qe.parse_mix("small")

    def test_the_census_reports_every_class(self, qe):
        rows = qe.mix_census(_banded())
        assert [r["class"] for r in rows] == ["car"]
        assert rows[0]["counts"] == {"small": 10, "medium": 14, "large": 6}


class TestTheRuleIsOneRule:
    """Two tiers restate the same draw; these pin them to each other."""

    def test_the_hash_rank_matches_the_pile_builders(self, qe):
        """A mix's small slice is only a subset of `car@small` while both order
        candidates by the same key. `scale_core` is experiment-tier and cannot be
        imported from the library, so the library restates the rule -- and a
        restatement that nothing compares is a copy waiting to drift."""
        import sys

        sys.path.insert(0, str(_PILE_DIR))
        from pilebuild.scale_core import rank

        from vtscore.eval.scale_bands import _rank

        for cell in ("car@small", "stop sign@large", "book@medium"):
            for iid in (1, 7, 4242):
                assert _rank(cell, iid) == rank(cell, iid)

    def test_the_two_quota_rules_agree(self, qe):
        """The export cuts a manifest and the harness retags a pool; a mix that
        means two different things depending on which one built it is worse than
        either."""
        from vtscore.eval.scale_bands import _quota

        shares = {"small": 0.25, "medium": 0.5, "large": 0.25}
        for n in (4, 10, 17, 24, 100):
            for avail in ({"small": 99, "medium": 99, "large": 99}, {"small": 3, "medium": 99, "large": 6}):
                assert qe.quota(shares, n, avail) == _quota(shares, n, avail)
