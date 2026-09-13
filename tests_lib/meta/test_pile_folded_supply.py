"""Why a confirmed positive is not a designated one (#3818).

`folded_supply.py` apportions #3696's 441 `folded` errors between the reasons the
build did not designate them. Its whole value is in two properties that the
printed table cannot show, and both have already been got wrong once:

* **the cascade charges the FIRST pass that drops a pair**, not the first test
  that happens to match. A pair lost to the COCO anchor must not be reported as
  one `apply_corrections` answered, because the two imply different repairs;
* **the control is conditioned the same way the sample is.** The 441 are
  undesignated by construction and a scattered image can never be designated, so
  dividing by every queue image holding the class manufactures a 3.6x excess out
  of the conditioning alone.

A third is about the examples rather than the numbers: taking the first rows of
each *outcome* returns whichever class sorts first and nothing else, which is how
the first draft illustrated the corpus with nine `backpack` rows.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def fs():
    """``folded_supply``, which defers ``setup_env`` into ``main``."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import folded_supply

    return folded_supply


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


#: A 100x100 frame, so a box's area in px is its area fraction x 10,000.
DIMS = (100, 100)
SMALL = [0.0, 0.0, 5.0, 5.0]  # 0.25% of the frame
MEDIUM = [0.0, 0.0, 20.0, 20.0]  # 4%
HUGE = [0.0, 0.0, 95.0, 95.0]  # 90%, past MAX_VOTED_AREA
FAR = [90.0, 90.0, 100.0, 100.0]  # with SMALL, a union covering the frame


def _snaps(pairs, drop_at=None):
    """Stage snapshots where every pair survives, except one dropped at *drop_at*.

    The stages are a dict of flags by design -- `apportion` is pure given them,
    so a whole build can be simulated without VG's 350 MB object table.
    """
    stages = ("read", "anchor", "fold", "corrections", "lift")
    out = {}
    seen = False
    for stage in stages:
        seen = seen or stage == drop_at
        out[stage] = dict.fromkeys(pairs, not seen)
    return out


def _apportion(fs, pc, pairs, snaps, *, labels, supply=None, chosen=None, unbanded=(), in_read=None, designated=None):
    return fs.apportion(
        pairs,
        snaps,
        set(unbanded),
        labels,
        designated or {},
        dict.fromkeys({i for _, i in pairs}, DIMS),
        supply or {c: {b: [] for b in pc.BOX_BANDS} for c, _ in pairs},
        chosen or {},
        in_read if in_read is not None else set(labels),
    )


class TestTheCascadeChargesTheFirstPass:
    """A pair that leaves early must be charged where it left, not later."""

    @pytest.mark.parametrize(
        ("drop_at", "outcome"),
        [("anchor", "anchor_absent"), ("fold", "fold_lost"), ("lift", "withheld_lift")],
    )
    def test_each_pass_owns_the_pairs_it_drops(self, fs, pc, drop_at, outcome):
        pairs = [("bench", 7)]
        rows = _apportion(fs, pc, pairs, _snaps(pairs, drop_at), labels={7: {}})
        assert rows[0]["outcome"] == outcome

    def test_a_pair_the_anchor_dropped_is_not_blamed_on_corrections(self, fs, pc):
        """The two imply different repairs -- re-audit a VG spelling against COCO,
        versus revisit a human verdict -- so conflating them is not cosmetic."""
        pairs = [("bench", 7)]
        # Present in `unbanded`, i.e. the shape `reviewed_boxless` looks for. The
        # anchor still owns it, because the anchor is asked first.
        rows = _apportion(fs, pc, pairs, _snaps(pairs, "anchor"), labels={7: {}}, unbanded={(7, "bench")})
        assert rows[0]["outcome"] == "anchor_absent"

    def test_a_reviewed_pair_splits_on_whether_a_box_was_drawn(self, fs, pc):
        pairs = [("bench", 7), ("bench", 8)]
        snaps = _snaps(pairs, "corrections")
        rows = _apportion(fs, pc, pairs, snaps, labels={7: {}, 8: {}}, unbanded={(7, "bench")})
        assert [r["outcome"] for r in rows] == ["reviewed_boxless", "reviewed_absent"]

    def test_an_image_the_build_never_read_is_not_a_degenerate_box(self, fs, pc):
        """Two different facts about VG: an image with no JPEG, and a named object
        whose box has no area. Only the second says anything about the annotation."""
        pairs = [("bench", 7), ("bench", 8)]
        snaps = _snaps(pairs, "read")
        rows = _apportion(fs, pc, pairs, snaps, labels={8: {}}, in_read={8})
        assert [r["outcome"] for r in rows] == ["image_dropped", "degenerate_box"]


class TestTheBandDecidesTheRest:
    def test_a_scattered_pair_never_reaches_a_cell(self, fs, pc):
        pairs = [("bench", 7)]
        rows = _apportion(fs, pc, pairs, _snaps(pairs), labels={7: {"bench": [SMALL, FAR]}})
        assert rows[0]["outcome"] == "scattered"
        assert rows[0]["inflation"] > pc.BAND_MAX_INFLATION
        assert "cell" not in rows[0]

    def test_scatter_is_asked_before_size(self, fs, pc):
        """`band_for` returns SCATTERED first, so an image that is both scattered
        and frame-filling is a scatter finding. Reporting it as `oversize` would
        put it in the bucket a band-edge change could fix, and it could not."""
        pairs = [("bench", 7)]
        rows = _apportion(fs, pc, pairs, _snaps(pairs), labels={7: {"bench": [SMALL, FAR]}})
        assert rows[0]["union_area"] > pc.MAX_VOTED_AREA
        assert rows[0]["outcome"] == "scattered"

    def test_a_frame_filling_box_missed_every_band(self, fs, pc):
        pairs = [("bench", 7)]
        rows = _apportion(fs, pc, pairs, _snaps(pairs), labels={7: {"bench": [HUGE]}})
        assert rows[0]["outcome"] == "oversize"
        assert rows[0]["union_area"] >= pc.MAX_VOTED_AREA

    def test_a_banded_pair_is_cell_full_only_when_the_cell_did_not_take_it(self, fs, pc):
        pairs = [("bench", 7)]
        labels = {7: {"bench": [MEDIUM]}}
        supply = {"bench": {b: [] for b in pc.BOX_BANDS}}
        supply["bench"]["medium"] = [7, 8, 9]
        cell = pc.scale_cell("bench", "medium")

        full = _apportion(fs, pc, pairs, _snaps(pairs), labels=labels, supply=supply, chosen={cell: [8, 9]})
        assert full[0]["outcome"] == "cell_full"
        assert full[0]["cell"] == cell
        assert full[0]["cell_supply"] == 3

        taken = _apportion(fs, pc, pairs, _snaps(pairs), labels=labels, supply=supply, chosen={cell: [7, 8]})
        assert taken[0]["outcome"] == "designated"

    def test_a_reviewers_box_decides_the_band(self, fs, pc):
        """#3726: a designation bands the cell while the instances stay behind it.
        Without this the drawn box would be unioned with the rest and a scattered
        set would still leave every band -- the finding would be about VG's
        instances rather than about the image the reviewer picked."""
        pairs = [("bench", 7)]
        labels = {7: {"bench": [SMALL, FAR]}}
        rows = _apportion(fs, pc, pairs, _snaps(pairs), labels=labels, designated={(7, "bench"): [MEDIUM]})
        assert rows[0]["outcome"] != "scattered"
        assert rows[0]["designated_box"] is True


class TestTheCensus:
    def test_every_outcome_is_printed_even_at_zero(self, fs):
        rows = [{"class": "bench", "outcome": "cell_full"}]
        per_class, _pooled = fs.census(rows)
        assert set(fs.OUTCOMES) <= set(per_class[0])
        assert per_class[0]["scattered"] == 0

    def test_actionable_excludes_the_benign_outcome(self, fs):
        rows = [
            {"class": "bench", "outcome": "cell_full"},
            {"class": "bench", "outcome": "cell_full"},
            {"class": "boat", "outcome": "scattered"},
            {"class": "boat", "outcome": "oversize"},
        ]
        _, pooled = fs.census(rows)
        assert pooled["folded"] == 4
        assert pooled["actionable"] == 2  # scattered + oversize; cell_full is not a fault
        assert pooled["benign_share"] == 0.5


class TestTheControlIsConditionedLikeTheSample:
    """The bug that would invent a finding out of nothing."""

    def test_rates_divide_by_the_undesignated_count_not_the_held_count(self, fs, pc):
        labels = {
            1: {"bench": [MEDIUM]},  # designated
            2: {"bench": [MEDIUM]},  # designated
            3: {"bench": [MEDIUM]},  # designated
            4: {"bench": [SMALL, FAR]},  # scattered, and undesignatable BY DEFINITION
        }
        box_dims = dict.fromkeys(labels, DIMS)
        cell = pc.scale_cell("bench", "medium")
        (row,) = fs.control(labels, {}, box_dims, {cell: [1, 2, 3]}, ["bench"], set(labels))

        assert row["queue_held"] == 4
        assert row["designated"] == 3
        assert row["undesignated"] == 1
        # 1/1, not 1/4. The three designated images could never have been
        # scattered, so counting them in the denominator says the filter fires
        # four times less often than it does among the pairs being apportioned.
        assert row["scattered_rate"] == 1.0

    def test_images_outside_the_queue_are_not_counted(self, fs):
        labels = {1: {"bench": [MEDIUM]}, 2: {"bench": [MEDIUM]}}
        (row,) = fs.control(labels, {}, dict.fromkeys(labels, DIMS), {}, ["bench"], {1})
        assert row["queue_held"] == 1

    def test_it_uses_the_same_three_keys_the_census_does(self, fs):
        """So the report's two bars are the same split, not two vocabularies."""
        labels = {1: {"bench": [MEDIUM]}}
        (row,) = fs.control(labels, {}, dict.fromkeys(labels, DIMS), {}, ["bench"], {1})
        assert {"scattered", "oversize", "cell_full"} <= set(row)


class TestTheExamplesSpanTheClasses:
    def test_one_class_cannot_take_every_row_of_an_outcome(self, fs):
        rows = [{"class": "backpack", "outcome": "scattered", "image_id": i} for i in range(5)]
        rows.append({"class": "boat", "outcome": "scattered", "image_id": 99})
        got = fs.pick_examples(rows, {}, 1)
        assert [(r["class"], r["image_id"]) for r in got] == [("backpack", 0), ("boat", 99)]

    def test_each_row_carries_what_vg_said(self, fs):
        """An example a reader cannot check is a claim, not an example: the row
        has to show the spelling that put the image in the `folded` bucket."""
        rows = [{"class": "bicycle", "outcome": "scattered", "image_id": 4}]
        (got,) = fs.pick_examples(rows, {4: Counter({"bicycle": 2, "sky": 9})}, 1)
        assert got["matched"] == ["bicycle"]
        assert got["vg_names_total"] == 11
        assert "sky" not in got["vg_named"]  # carried in the count, dropped from the list


class TestSpellings:
    def test_an_ambiguous_name_is_not_one_of_them(self, fs, pc):
        """`lift_ambiguous` dropping a spelling has to be *visible* as a stage
        losing the class. Counting `bike` as a way of holding `bicycle` would
        make `withheld_lift` unreachable and charge the pair to the band instead."""
        (ambiguous,) = pc.SCALE_VG_AMBIGUOUS["bicycle"][:1]
        assert ambiguous not in fs.spellings("bicycle")
        assert "bicycle" in fs.spellings("bicycle")

    def test_a_folded_spelling_is_one_of_them(self, fs, pc):
        (alias,) = pc.SCALE_VG_NAMES["bicycle"][:1]
        assert alias in fs.spellings("bicycle")

    def test_held_reads_any_spelling(self, fs, pc):
        (alias,) = pc.SCALE_VG_NAMES["bicycle"][:1]
        assert fs.held({1: {alias: [MEDIUM]}}, 1, "bicycle")
        assert not fs.held({1: {"tree": [MEDIUM]}}, 1, "bicycle")
        assert not fs.held({}, 1, "bicycle")
