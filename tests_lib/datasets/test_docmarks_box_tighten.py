"""Hand-reviewed tighter boxes: proposals, verdicts, and the store a rebuild replays."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield {"b": importlib.import_module("box_tighten"), "common": importlib.import_module("sources._common")}
    finally:
        sys.path.remove(str(_DOCMARKS))


def _page(mods, page_id, marks, w=1000, h=1000):
    Mark, Page = mods["common"].Mark, mods["common"].Page
    return Page(page_id=page_id, source="staver", path="x.png", width=w, height=h, marks=[Mark(*m) for m in marks])


def _corners(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _spread(x0, y0, x1, y1, n=20):
    return [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)) for i in range(n)]


class TestPropose:
    def test_projected_query_outline_becomes_the_box(self, mods):
        b = mods["b"]
        prop = b.Proposal("staver/p", 1, (100, 100, 600, 400))
        b.propose(prop, _corners(110, 105, 660, 385), _spread(120, 110, 650, 380), (1000, 1000))
        assert prop.new_box == (110, 105, 550, 280)
        assert prop.inliers == 20 and prop.flags == []

    def test_overshoot_is_clamped_and_flagged(self, mods):
        b = mods["b"]
        prop = b.Proposal("staver/p", 0, (100, 100, 400, 200))
        b.propose(prop, _corners(0, 150, 250, 300), _spread(60, 160, 240, 290), (1000, 1000))
        assert prop.flags == ["clamped: fit reaches far past old box"]
        assert prop.new_box[0] == 40  # old x minus 15% of its width
        grown = b.propose(
            b.Proposal("p", 0, (100, 100, 400, 200)),
            _corners(90, 150, 250, 300),
            _spread(95, 160, 240, 290),
            (1000, 1000),
        )
        assert grown.flags == ["grows the old box"] and grown.new_box[0] == 90

    def test_weak_fits_are_flagged_and_no_fit_has_no_box(self, mods):
        b = mods["b"]
        few = b.propose(
            b.Proposal("p", 0, (0, 0, 400, 400)), _corners(0, 0, 200, 200), _spread(90, 90, 100, 100, n=5), (1000, 1000)
        )
        assert few.flags == ["few inliers", "inliers span little of the box"]
        # A projection onto an already tight box lands a little larger: no decision, no warnings.
        same = b.propose(
            b.Proposal("p", 0, (100, 100, 400, 200)),
            _corners(95, 95, 505, 305),
            _spread(96, 96, 504, 304, n=3),
            (1000, 1000),
        )
        assert same.flags == ["unchanged"] and same.new_box is not None
        # ...but one edge moving 12% is a finding, however high the IoU.
        edge = b.propose(
            b.Proposal("p", 0, (810, 563, 285, 185)),
            _corners(776, 563, 1093, 747),
            _spread(780, 570, 1090, 740),
            (2000, 2000),
        )
        assert edge.flags == ["grows the old box"]
        none = b.propose(b.Proposal("p", 0, (0, 0, 10, 10)), None, None, (1000, 1000))
        assert none.new_box is None and none.flags == ["no fit"]

    def test_ink_points_ignore_the_paper(self, mods):
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (100, 60), (240, 225, 225))  # pinkish paper
        ImageDraw.Draw(im).rectangle([20, 10, 80, 40], outline=(90, 110, 220), width=2)
        pts = mods["b"].ink_points(im)
        assert pts[:, 0].min() == 20 and pts[:, 0].max() == 80
        assert pts[:, 1].min() == 10 and pts[:, 1].max() == 40

    def test_members_are_the_class_marks_with_their_index(self, mods):
        b = mods["b"]
        pages = {
            "staver/a": _page(mods, "staver/a", [("stamp", (1, 1, 5, 5), "other"), ("stamp", (10, 10, 50, 30), "c")]),
            "staver/b": _page(mods, "staver/b", [("stamp", (0, 0, 40, 20), "c")]),
        }
        got = b.members("c", {"page_ids": ["staver/b", "staver/a"]}, pages)
        assert [(p.page_id, p.mark_index) for p in got] == [("staver/a", 1), ("staver/b", 0)]


ROW = {
    "class_id": "c",
    "members": [
        {"index": 0, "page_id": "staver/a", "mark_index": 1, "old_box": [10, 10, 50, 30], "new_box": [12, 12, 30, 20]},
        {"index": 1, "page_id": "staver/b", "mark_index": 0, "old_box": [0, 0, 40, 20], "new_box": None},
    ],
}


def _corpus(mods):
    pages = [
        _page(mods, "staver/a", [("stamp", (1, 1, 5, 5), "other"), ("stamp", (10, 10, 50, 30), "c")]),
        _page(mods, "staver/b", [("stamp", (0, 0, 40, 20), "c")]),
    ]
    classes = {"c": {"page_ids": ["staver/a", "staver/b"], "query_page_id": "staver/a", "query_crop": "q.png"}}
    return pages, classes


class TestApply:
    def test_accepted_box_replaces_the_mark_in_place_and_is_stored(self, mods):
        pages, classes = _corpus(mods)
        store: list = []
        changes, problems, stale = mods["b"].apply_box_tighten(
            pages, classes, [dict(ROW, verdict="0")], store, reviewer="sam"
        )
        assert problems == [] and len(changes) == 1
        mark = pages[0].marks[1]
        assert (mark.box, mark.class_id) == ((12, 12, 30, 20), "c")
        assert pages[0].marks[0].box == (1, 1, 5, 5)  # index 1 still means the same mark
        assert store == [
            {
                "page_id": "staver/a",
                "mark_index": 1,
                "class_id": "c",
                "old_box": [10, 10, 50, 30],
                "new_box": [12, 12, 30, 20],
                "provenance": "gt",
                "reviewed_by": "sam",
            }
        ]
        assert stale == {"c"}  # the query page's box moved, so its crop must be re-cut
        assert classes["c"]["audit"]["box_tighten_checked"]["accepted"] == 1

    def test_problems_blank_and_a_corpus_that_moved(self, mods):
        b = mods["b"]
        pages, classes = _corpus(mods)
        assert b.apply_box_tighten(pages, classes, [dict(ROW, verdict="")], []) == ([], [], set())
        _, problems, _ = b.apply_box_tighten(pages, classes, [dict(ROW, verdict="1")], [])
        assert problems and "no proposed box" in problems[0]
        pages[0].marks[1] = mods["common"].Mark("stamp", (11, 10, 50, 30), "c")
        store: list = []
        _, problems, _ = b.apply_box_tighten(pages, classes, [dict(ROW, verdict="0")], store)
        assert problems and "no longer has the box" in problems[0] and store == []
        for bad in ("2", "0,0", "yes"):
            assert b.apply_box_tighten(pages, classes, [dict(ROW, verdict=bad)], [])[1]

    def test_reaccepting_a_mark_replaces_its_store_row(self, mods):
        b = mods["b"]
        pages, classes = _corpus(mods)
        store = [{"page_id": "staver/a", "mark_index": 1, "old_box": [10, 10, 50, 30], "new_box": [0, 0, 1, 1]}]
        b.apply_box_tighten(pages, classes, [dict(ROW, verdict="0")], store)
        assert len(store) == 1 and store[0]["new_box"] == [12, 12, 30, 20]


class TestReplay:
    def test_replay_restores_the_box_by_index_is_idempotent_and_warns_on_drift(self, mods, tmp_path):
        b = mods["b"]
        pages, classes = _corpus(mods)
        store: list = []
        b.apply_box_tighten(pages, classes, [dict(ROW, verdict="0")], store)
        b.save_store(store, tmp_path / b.STORE)

        rebuilt, _ = _corpus(mods)
        warnings: list[str] = []
        rows = b.load_store(tmp_path / b.STORE)
        assert b.replay_box_overrides(rebuilt, rows, warnings) == 1
        assert rebuilt[0].marks[1].box == (12, 12, 30, 20) and rebuilt[0].marks[0].box == (1, 1, 5, 5)
        assert b.replay_box_overrides(rebuilt, rows, warnings) == 0 and warnings == []

        drifted, _ = _corpus(mods)
        drifted[0].marks[1] = mods["common"].Mark("stamp", (9, 9, 9, 9), None)
        assert (
            b.replay_box_overrides(
                drifted,
                rows + [{"page_id": "staver/gone", "mark_index": 0, "old_box": [0, 0, 1, 1], "new_box": [0, 0, 1, 1]}],
                warnings,
            )
            == 0
        )
        assert len(warnings) == 2 and "no longer has box" in warnings[0] and "gone" in warnings[1]


class TestBandLocated:
    """#4109: a mark located only by its letterhead band gets a real box, and says so."""

    def _band_corpus(self, mods):
        Mark, Page = mods["common"].Mark, mods["common"].Page
        page = Page(
            page_id="ucsf/a#0",
            source="ucsf",
            path="x.png",
            width=1240,
            height=1680,
            marks=[
                Mark("logo", (0, 0, 1240, 370), None, "candidate"),
                Mark("logo", (0, 0, 1240, 370), "ucsf/logo_x", "ucsf_classes_band"),
            ],
        )
        tight = Page(
            page_id="ucsf/b#0",
            source="ucsf",
            path="x.png",
            width=1240,
            height=1680,
            marks=[Mark("logo", (500, 100, 60, 40), "ucsf/logo_x", "ucsf_classes")],
        )
        return [page, tight], {"ucsf/logo_x": {"page_ids": ["ucsf/a#0", "ucsf/b#0"], "query_page_id": "ucsf/q#0"}}

    def test_band_only_proposes_just_the_band_located_marks(self, mods):
        pages, classes = self._band_corpus(mods)
        by_id = {p.page_id: p for p in pages}
        got = mods["b"].members("ucsf/logo_x", classes["ucsf/logo_x"], by_id, band_only=True)
        assert [(p.page_id, p.mark_index) for p in got] == [("ucsf/a#0", 1)]
        assert len(mods["b"].members("ucsf/logo_x", classes["ucsf/logo_x"], by_id)) == 2

    def test_an_accepted_box_makes_the_mark_located_and_a_rebuild_keeps_it(self, mods):
        b = mods["b"]
        pages, classes = self._band_corpus(mods)
        row = {
            "class_id": "ucsf/logo_x",
            "members": [
                {"page_id": "ucsf/a#0", "mark_index": 1, "old_box": [0, 0, 1240, 370], "new_box": [480, 90, 70, 50]}
            ],
            "verdict": "0",
        }
        store: list = []
        _, problems, _ = b.apply_box_tighten(pages, classes, [row], store)
        assert problems == []
        assert pages[0].marks[1].provenance == "ucsf_classes" and pages[0].marks[1].box == (480, 90, 70, 50)
        assert pages[0].marks[0].provenance == "candidate"  # the band itself is untouched
        rebuilt, _ = self._band_corpus(mods)
        assert b.replay_box_overrides(rebuilt, store) == 1
        assert rebuilt[0].marks[1].provenance == "ucsf_classes"

    def test_a_located_provenance_is_left_alone(self, mods):
        assert mods["b"].located_provenance("gt") == "gt"
        assert mods["b"].located_provenance("ucsf_classes_band") == "ucsf_classes"


class TestLeftovers:
    """#4125: marks a box pass left without a reviewed box."""

    @pytest.fixture
    def lo(self):
        sys.path.insert(0, str(_DOCMARKS))
        try:
            yield importlib.import_module("box_leftovers")
        finally:
            sys.path.remove(str(_DOCMARKS))

    ROWS = [
        {
            "class_id": "ucsf/logo_x",
            "verdict": "0",
            "members": [
                {
                    "index": 0,
                    "page_id": "u/ok",
                    "mark_index": 1,
                    "old_box": [1, 1, 5, 5],
                    "new_box": [2, 2, 4, 4],
                    "flags": [],
                },
                {
                    "index": 1,
                    "page_id": "u/q",
                    "mark_index": 1,
                    "old_box": [1, 1, 5, 5],
                    "new_box": [3, 3, 4, 4],
                    "flags": [],
                },
                {
                    "index": 2,
                    "page_id": "u/nofit",
                    "mark_index": 1,
                    "old_box": [0, 0, 900, 300],
                    "new_box": None,
                    "flags": ["no fit"],
                },
                {
                    "index": 3,
                    "page_id": "u/same",
                    "mark_index": 1,
                    "old_box": [1, 1, 5, 5],
                    "new_box": [1, 1, 5, 5],
                    "flags": ["unchanged"],
                },
                {
                    "index": 4,
                    "page_id": "u/rej",
                    "mark_index": 1,
                    "old_box": [7, 7, 5, 5],
                    "new_box": [8, 8, 4, 4],
                    "flags": [],
                },
            ],
        }
    ]

    def test_the_query_page_gets_its_crop_extent_and_the_rest_are_drawn(self, lo):
        classes = {"ucsf/logo_x": {"query_page_id": "u/q"}}
        query_rows, draw_rows = lo.leftovers(self.ROWS, classes, {"ucsf/logo_x": [10, 20, 30, 40]})
        (q,) = query_rows
        assert (
            q["verdict"] == "0"
            and q["members"][0]["page_id"] == "u/q"
            and q["members"][0]["new_box"] == [10, 20, 30, 40]
        )
        (d,) = draw_rows
        assert [(m["page_id"], m["why"]) for m in d["members"]] == [("u/nofit", "no fit"), ("u/rej", "rejected")]
        assert all(m["new_box"] is None for m in d["members"]) and d["verdict"] == ""

    def test_a_query_page_with_no_known_crop_box_is_drawn_instead(self, lo):
        classes = {"ucsf/logo_x": {"query_page_id": "u/q"}}
        query_rows, draw_rows = lo.leftovers(self.ROWS, classes, {})
        assert query_rows == [] and "u/q" in [m["page_id"] for m in draw_rows[0]["members"]]
