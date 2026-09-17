"""DocMarks review as VTSearch Good/Bad queues: votes map back to each audit's verdicts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def br():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield importlib.import_module("binary_review")
    finally:
        sys.path.remove(str(_DOCMARKS))


def _q(task, key, fn):
    return fn, {"task": task, "key": key}


class TestBankOnlyReadsDocmarks:
    def test_vg_scale_detectors_are_ignored(self, br):
        dets = [
            {"name": "docmarks pm_crest -- right logo same as left?"},
            {"name": "bottle incl jars -- seat check"},
            {"name": "docmarksish not ours"},
        ]
        assert [d["name"] for d in br.docmarks_detectors(dets)] == ["docmarks pm_crest -- right logo same as left?"]

    def test_queue_names_must_carry_the_prefix(self, br, tmp_path):
        with pytest.raises(ValueError):
            br.emit([], tmp_path, "pm_crest -- same?", pages={}, corpus=tmp_path)

    def test_a_file_voted_both_ways_is_refused(self, br):
        with pytest.raises(ValueError):
            br.votes_from_labels({"good": [{"filename": "a.jpg"}], "bad": [{"filename": "a.jpg"}]})


class TestUcsf:
    ROWS = [
        {
            "task": "ucsf_classes",
            "proposal": "p",
            "sheet": "p__a_00.png",
            "cells": [{"index": 0}, {"index": 1}, {"index": 2}],
            "verdict": "",
        },
        {
            "task": "ucsf_classes",
            "proposal": "p",
            "sheet": "p__b_00.png",
            "cells": [{"index": 0}, {"index": 1}],
            "verdict": "",
        },
        {"task": "ucsf_classes_relation", "proposal": "p", "relation": ""},
    ]
    QS = dict(
        [
            _q("ucsf_classes", {"sheet": "p__a_00.png", "cell": 0}, "a0"),
            _q("ucsf_classes", {"sheet": "p__a_00.png", "cell": 1}, "a1"),
            _q("ucsf_classes", {"sheet": "p__a_00.png", "cell": 2}, "a2"),
            _q("ucsf_classes", {"sheet": "p__b_00.png", "cell": 0}, "b0"),
            _q("ucsf_classes", {"sheet": "p__b_00.png", "cell": 1}, "b1"),
            _q("ucsf_classes_relation", {"proposal": "p", "target": "tobacco800/logo_x"}, "rel"),
        ]
    )

    def test_answered_sheets_become_cell_lists_and_partial_ones_stay_blank(self, br):
        votes = {"a0": "good", "a1": "bad", "a2": "good", "b0": "good", "rel": "good"}
        rows, unanswered = br.translate_ucsf(self.ROWS, self.QS, votes)
        assert rows[0]["verdict"] == "0,2"
        assert rows[1]["verdict"] == "" and any("p__b_00" in u for u in unanswered)
        assert rows[2]["relation"] == "extends tobacco800/logo_x"
        assert self.ROWS[0]["verdict"] == ""  # originals untouched

    def test_all_none_and_new(self, br):
        votes = {"a0": "good", "a1": "good", "a2": "good", "b0": "bad", "b1": "bad", "rel": "bad"}
        rows, unanswered = br.translate_ucsf(self.ROWS, self.QS, votes)
        assert [r.get("verdict") for r in rows[:2]] == ["all", "none"] and rows[2]["relation"] == "new"
        assert unanswered == []


class TestQueryCropsAndBoxes:
    def test_query_crops_keep_rank_order_capped(self, br):
        qs = dict(_q("query_crops", {"class_id": "c", "index": i}, f"q{i}") for i in range(6))
        votes = {f"q{i}": ("good" if i != 1 else "bad") for i in range(6)}
        rows, unanswered = br.translate_query_crops([{"class_id": "c", "verdict": ""}], qs, votes, cap=3)
        assert rows[0]["verdict"] == "0,2,3" and unanswered == []

    def test_query_crops_unanswered_is_reported_not_guessed(self, br):
        qs = dict(_q("query_crops", {"class_id": "c", "index": i}, f"q{i}") for i in range(2))
        rows, unanswered = br.translate_query_crops([{"class_id": "c", "verdict": ""}], qs, {"q0": "good"})
        assert rows[0]["verdict"] == "" and unanswered

    def test_boxes_only_ask_about_changed_members(self, br):
        assert br.box_questionable({"flags": [], "new_box": [1, 2, 3, 4]})
        assert not br.box_questionable({"flags": ["unchanged"], "new_box": [1, 2, 3, 4]})
        assert not br.box_questionable({"flags": ["no fit"], "new_box": None})
        qs = dict(
            [
                _q("box_tighten", {"class_id": "c", "index": 5}, "m5"),
                _q("box_tighten", {"class_id": "c", "index": 9}, "m9"),
            ]
        )
        rows, _ = br.translate_box_tighten([{"class_id": "c", "verdict": ""}], qs, {"m5": "good", "m9": "bad"})
        assert rows[0]["verdict"] == "5"


class TestRendering:
    def test_region_and_scale(self, br):
        assert br.expand((100, 100, 40, 20), None, 1000, 1000) == (90, 90, 150, 130)
        assert br.expand((0, 0, 40, 20), (0, 0, 80, 20), 60, 1000) == (0, 0, 60, 40)
        # a 50 px mark is enlarged toward 400 px but never past the panel
        assert br.panel_scale((0, 0, 60, 60), 50, (840, 580)) == 8.0
        assert br.panel_scale((0, 0, 600, 600), 500, (840, 580)) == pytest.approx(580 / 600)

    def test_class_name_leads_the_queue_name(self, br):
        assert br.short_class("tobacco800/logo_ajj10e00_1") == "t800 logo_ajj10e00_1"


class TestCompleteness2:
    def test_tile_box_goods_are_held_back_for_a_tight_box(self, br):
        qs = {
            "c0": {"task": "completeness2", "key": {"class_id": "c", "index": 0, "tile_box": False}},
            "c1": {"task": "completeness2", "key": {"class_id": "c", "index": 1, "tile_box": True}},
            "c2": {"task": "completeness2", "key": {"class_id": "c", "index": 2, "tile_box": False}},
        }
        rows, notes = br.translate_completeness2(
            [{"class_id": "c", "verdict": "none"}], qs, {"c0": "good", "c1": "good", "c2": "bad"}
        )
        assert rows[0]["verdict"] == "0" and rows[0]["needs_tight_box"] == [1]
        assert any("tile box" in n for n in notes)

    def test_sig_only_candidates_are_tiles(self, br):
        assert br.tile_box({"methods": "SIG"})
        assert not br.tile_box({"methods": "OCR+SIG"})
