"""Whether VG was silent, or the build did not listen (#3696).

`silence_rate.py` has to call a pair silent when the queue does not *designate*
it, because a designation is all a cell pickle carries (#3678). `silence_source`
is the correction, and it exists because the gap turned out to be most of the
number: 524 of 815 confirmed errors are images VG **did** name, 441 under a
spelling the build folds and 83 under one it deliberately refuses.

Two things have to hold for that correction to be worth anything, and neither is
visible in the rate it prints:

* **the three readings are decided by the tables, never by resemblance.** A
  string-similarity rule would turn `bike rack` into a `bicycle` and manufacture
  the finding;
* **both halves of the fraction move.** Splitting only the numerator divides
  `coverage` errors by a denominator that still counts every pair VG named, which
  understates the corrected rate by exactly the share just removed from the top.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def ss():
    """``silence_source``, which defers ``setup_env`` into ``main``."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import silence_source

    return silence_source


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


class TestTheTablesDecide:
    def test_the_classs_own_name_is_folded(self, ss):
        assert ss.classify("bicycle", {"bicycle", "tree"}) == "folded"

    def test_a_spelling_the_build_folds_is_folded(self, ss, pc):
        (alias,) = pc.SCALE_VG_NAMES["bicycle"][:1]
        assert ss.classify("bicycle", {alias}) == "folded"

    def test_a_spelling_the_build_refuses_is_withheld(self, ss, pc):
        """`bike` is in `SCALE_VG_AMBIGUOUS` on purpose (#3605). VG spoke; the
        build declined to listen, which is a ruling and not an error."""
        assert "bike" in pc.SCALE_VG_AMBIGUOUS["bicycle"]
        assert ss.classify("bicycle", {"bike", "street"}) == "withheld"

    def test_folded_wins_over_withheld_when_both_are_present(self, ss):
        """An image carrying both is one the build already puts in the class, so
        the reason it is not designated is downstream of any name ruling."""
        assert ss.classify("bicycle", {"bicycle", "bike"}) == "folded"

    def test_a_lookalike_name_is_not_guessed_at(self, ss):
        """`bike rack` is not in either table, and inventing a substring rule
        here would classify the rack as the bicycle."""
        assert ss.classify("bicycle", {"bike rack", "sidewalk"}) == "coverage"

    def test_nothing_recognisable_is_coverage(self, ss):
        assert ss.classify("bench", {"grass", "woman", "ocean"}) == "coverage"


class TestBothHalvesOfTheFractionMove:
    def _census(self, ss, names_by_image):
        rows = [{"class": "bench", "silent_pairs": 4, "rate": 3 / 4}]
        answers = {"bench": {1: True, 2: True, 3: True, 4: False}}
        silent = {"bench": {1, 2, 3, 4}}
        return ss.census(rows, answers, silent, names_by_image, {"bench"})

    def test_the_denominator_drops_every_pair_vg_named(self, ss):
        """Images 2 and 4 carry the word, so neither is a pair a negative pool
        could ever draw from -- 4 silent pairs, 2 VG-silent."""
        rows, pooled = self._census(
            ss,
            {
                1: Counter({"grass": 1}),
                2: Counter({"bench": 1}),
                3: Counter({"ocean": 1}),
                4: Counter({"bench": 2}),
            },
        )
        (row,) = rows
        assert (row["silent_pairs"], row["vg_silent_pairs"]) == (4, 2)
        assert (row["coverage"], row["folded"]) == (2, 1)
        assert row["designation_rate"] == pytest.approx(0.75)
        assert row["coverage_rate"] == pytest.approx(1.0)  # 2 of 2, not 2 of 4
        assert pooled["vg_silent_pairs"] == 2

    def test_correcting_only_the_numerator_would_halve_the_rate(self, ss):
        """The regression this guards: 2/4 instead of 2/2. Stated as an
        inequality against the uncorrected denominator so it cannot pass by
        coincidence."""
        rows, _ = self._census(
            ss,
            {1: Counter({"grass": 1}), 2: Counter({"bench": 1}), 3: Counter({"ocean": 1}), 4: Counter()},
        )
        (row,) = rows
        assert row["coverage_rate"] > row["coverage"] / row["silent_pairs"]

    def test_an_image_the_object_table_does_not_hold_reads_as_silent(self, ss):
        """Absent is not "VG named it": a missing record cannot be evidence that
        VG spoke, and reading it as such would shrink the denominator for free."""
        rows, _ = self._census(ss, {})
        (row,) = rows
        assert row["vg_silent_pairs"] == 4
        assert row["coverage"] == 3

    def test_a_class_outside_the_pooled_set_is_not_censused(self, ss):
        rows, pooled = ss.census([{"class": "bench", "silent_pairs": 1, "rate": 0.0}], {}, {"bench": {1}}, {}, set())
        assert rows == []
        assert pooled["vg_silent_pairs"] == 0


class TestReadingVgsNames:
    def test_every_name_is_read_not_only_the_ones_the_loader_folds(self, ss):
        """The question is what VG *did* say; filtering to the fold's own
        vocabulary would answer it with the answer already assumed."""
        got = ss.vg_names_by_image([{"image_id": 7, "objects": [{"names": ["Bike Rack", "  TREE "]}]}], {7})
        assert got[7] == Counter({"bike rack": 1, "tree": 1})

    def test_images_outside_the_ask_are_skipped(self, ss):
        recs = [{"image_id": 1, "objects": []}, {"image_id": 2, "objects": []}]
        assert set(ss.vg_names_by_image(recs, {2})) == {2}

    def test_an_object_with_several_names_counts_all_of_them(self, ss):
        """VG's `names` is a list and the loader reads only the first; here the
        question is whether the word appears at all, so all of them count."""
        got = ss.vg_names_by_image([{"image_id": 1, "objects": [{"names": ["seat", "bench"]}]}], {1})
        assert set(got[1]) == {"seat", "bench"}
