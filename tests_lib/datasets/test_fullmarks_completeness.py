"""The completeness pass (#3927): proposing missing members and folding verdicts back.

No matcher runs here; the GRID slate is the measurement.  What is checked is the
part that decides whether a verdict lands on the right mark and survives a
rebuild.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield {"c": importlib.import_module("completeness"), "common": importlib.import_module("sources._common")}
    finally:
        sys.path.remove(str(_FULLMARKS))


def _page(mods, page_id, source="tobacco800", marks=()):
    Mark, Page = mods["common"].Mark, mods["common"].Page
    return Page(page_id=page_id, source=source, path="x.png", width=1000, height=2000, marks=[Mark(*m) for m in marks])


def _corpus(mods):
    pages = [
        _page(mods, "tobacco800/member", marks=[("logo", (100, 100, 200, 200), "tobacco800/logo_a")]),
        _page(
            mods,
            "tobacco800/offroster",
            marks=[("signature", (0, 0, 50, 50), None), ("logo", (500, 500, 100, 100), "tobacco800/logo_z")],
        ),
        _page(mods, "tobacco800/nobox"),
        _page(mods, "tobacco800/other", marks=[("logo", (10, 10, 50, 50), "tobacco800/logo_q")]),
        _page(mods, "ucsf/leak", source="ucsf"),
    ]
    classes = {
        "tobacco800/logo_a": {
            "on_roster": True,
            "kind": "logo",
            "page_ids": ["tobacco800/member"],
            "n_instances": 1,
            "audit": {},
        },
        "tobacco800/logo_b": {
            "on_roster": True,
            "kind": "logo",
            "page_ids": ["tobacco800/other"],
            "n_instances": 1,
            "audit": {},
        },
    }
    return pages, classes


class TestCandidates:
    def test_members_ucsf_and_weak_matches_are_not_candidates(self, mods):
        pages, classes = _corpus(mods)
        by_id = {p.page_id: p for p in pages}
        inliers = {
            "tobacco800/logo_a": {
                "tobacco800/member": [90, 0],
                "tobacco800/offroster": [40, 3],
                "tobacco800/nobox": [60, 1],
                "tobacco800/other": [7, 0],
                "ucsf/leak": [99, 0],
            }
        }
        (entry,) = mods["c"].candidates(classes, by_id, inliers)
        assert [c.page_id for c in entry.candidates] == ["tobacco800/nobox", "tobacco800/offroster"]
        assert entry.positive_inliers == [90]

    def test_top_caps_the_list(self, mods):
        pages, classes = _corpus(mods)
        by_id = {p.page_id: p for p in pages}
        inliers = {"tobacco800/logo_a": {"tobacco800/offroster": [40, 0], "tobacco800/nobox": [60, 0]}}
        (entry,) = mods["c"].candidates(classes, by_id, inliers, top=1)
        assert [c.page_id for c in entry.candidates] == ["tobacco800/nobox"]


class TestLocate:
    def test_normalised_match_becomes_a_pixel_box_on_the_mark_it_overlaps(self, mods):
        pages, classes = _corpus(mods)
        by_id = {p.page_id: p for p in pages}
        entry = mods["c"].ClassCandidates(
            "tobacco800/logo_a", candidates=[mods["c"].Candidate("tobacco800/offroster", 40)]
        )
        # 0.52..0.58 x 0.26..0.29 of a 1000x2000 page sits inside the (500,500,100,100) logo
        mods["c"].locate(entry, by_id, lambda pid: (0.58, 0.29, 0.52, 0.26))
        (cand,) = entry.candidates
        assert cand.box == (520, 520, 60, 60)
        assert cand.mark_index == 1 and cand.mark_class_id == "tobacco800/logo_z"

    def test_a_match_that_cannot_be_refitted_is_dropped(self, mods):
        pages, classes = _corpus(mods)
        entry = mods["c"].ClassCandidates("tobacco800/logo_a", candidates=[mods["c"].Candidate("tobacco800/nobox", 60)])
        mods["c"].locate(entry, {p.page_id: p for p in pages}, lambda pid: None)
        assert entry.candidates == []

    def test_signatures_are_never_the_overlapping_mark(self, mods):
        page = _page(mods, "t/p", marks=[("signature", (0, 0, 100, 100), "sig")])
        assert mods["c"].overlapping_mark(page, (10, 10, 20, 20)) is None


def _row(cands, verdict):
    return {"class_id": "tobacco800/logo_a", "candidates": cands, "verdict": verdict}


OFFROSTER = {
    "page_id": "tobacco800/offroster",
    "inliers": 40,
    "box": [520, 520, 60, 60],
    "mark_index": 1,
    "mark_class_id": "tobacco800/logo_z",
}
NOBOX = {
    "page_id": "tobacco800/nobox",
    "inliers": 60,
    "box": [300, 400, 80, 90],
    "mark_index": None,
    "mark_class_id": None,
}


class TestApply:
    def test_accepted_offroster_mark_is_reassigned_and_must_linked(self, mods):
        pages, classes = _corpus(mods)
        changes, problems, merges, seps, added = mods["c"].apply_completeness(
            pages, classes, [_row([OFFROSTER], "0")], reviewer="sam"
        )
        assert problems == []
        assert pages[1].marks[1].class_id == "tobacco800/logo_a"
        assert classes["tobacco800/logo_a"]["page_ids"] == ["tobacco800/member", "tobacco800/offroster"]
        assert classes["tobacco800/logo_a"]["n_instances"] == 2
        assert merges[0]["right_page_id"] == "tobacco800/offroster" and merges[0]["right_mark_index"] == 1
        assert merges[0]["left_page_id"] == "tobacco800/member" and merges[0]["left_mark_index"] == 0
        assert added == [] and seps == []

    def test_accepted_unboxed_page_gains_a_marked_new_box(self, mods):
        pages, classes = _corpus(mods)
        _, problems, merges, _, added = mods["c"].apply_completeness(pages, classes, [_row([NOBOX], "0")])
        assert problems == []
        mark = pages[2].marks[0]
        assert (mark.box, mark.class_id, mark.provenance) == ((300, 400, 80, 90), "tobacco800/logo_a", "completeness")
        assert added[0]["page_id"] == "tobacco800/nobox" and added[0]["box"] == [300, 400, 80, 90]
        assert merges[0]["right_mark_index"] == 0

    def test_rejections_are_recorded_either_as_separations_or_checked_pages(self, mods):
        pages, classes = _corpus(mods)
        _, problems, merges, seps, _ = mods["c"].apply_completeness(pages, classes, [_row([OFFROSTER, NOBOX], "none")])
        assert problems == [] and merges == []
        assert seps[0]["left_page_id"] == "tobacco800/offroster" and seps[0]["right_class_id"] == "tobacco800/logo_a"
        audit = classes["tobacco800/logo_a"]["audit"]["completeness_checked"]
        assert audit["unboxed_rejected_page_ids"] == ["tobacco800/nobox"] and audit["accepted"] == 0

    def test_a_held_candidate_is_neither_accepted_nor_rejected(self, mods):
        """#4040: a Good vote parked for a hand-drawn box must not become a rejection."""
        pages, classes = _corpus(mods)
        row = dict(_row([NOBOX], "none"), needs_tight_box=[0])
        _, problems, merges, seps, added = mods["c"].apply_completeness(pages, classes, [row])
        assert problems == [] and merges == [] and seps == [] and added == []
        audit = classes["tobacco800/logo_a"]["audit"]["completeness_checked"]
        # the page must NOT be written off, or decided() never re-proposes the mark
        assert audit["unboxed_rejected_page_ids"] == []
        assert audit["held_for_box"] == [0]
        assert pages[2].marks == []

    def test_a_held_candidate_over_a_mark_is_not_a_cannot_link(self, mods):
        pages, classes = _corpus(mods)
        row = dict(_row([OFFROSTER], "none"), needs_tight_box=[0])
        _, problems, merges, seps, _ = mods["c"].apply_completeness(pages, classes, [row])
        assert problems == [] and merges == [] and seps == []
        assert pages[1].marks[1].class_id == "tobacco800/logo_z"

    def test_an_instance_of_another_roster_class_is_a_merge_not_a_member(self, mods):
        pages, classes = _corpus(mods)
        other = {
            "page_id": "tobacco800/other",
            "inliers": 30,
            "box": [10, 10, 50, 50],
            "mark_index": 0,
            "mark_class_id": "tobacco800/logo_q",
        }
        classes["tobacco800/logo_q"] = {"on_roster": True, "page_ids": ["tobacco800/other"], "audit": {}}
        _, problems, merges, _, _ = mods["c"].apply_completeness(pages, classes, [_row([other], "0")])
        assert problems and "merge" in problems[0]
        assert pages[3].marks[0].class_id == "tobacco800/logo_q" and merges == []

    def test_blank_verdict_is_skipped_and_garbage_is_a_problem(self, mods):
        pages, classes = _corpus(mods)
        before = copy.deepcopy(classes)
        changes, problems, *_ = mods["c"].apply_completeness(pages, classes, [_row([NOBOX], "")])
        assert changes == [] and problems == [] and classes == before
        _, problems, *_ = mods["c"].apply_completeness(pages, classes, [_row([NOBOX], "yes please")])
        assert problems
        _, problems, *_ = mods["c"].apply_completeness(pages, classes, [_row([NOBOX], "4")])
        assert problems and "outside" in problems[0]


class TestAddedMarksSurviveARebuild:
    def test_replay_puts_the_new_box_at_the_index_the_must_link_names(self, mods, tmp_path):
        pages, classes = _corpus(mods)
        _, _, merges, _, added = mods["c"].apply_completeness(pages, classes, [_row([NOBOX], "0")])
        store = tmp_path / mods["c"].ADDED_MARKS
        mods["c"].save_added_marks(added, store)

        rebuilt, _ = _corpus(mods)  # the sources again: no hand-added box
        warnings: list[str] = []
        assert mods["c"].replay_added_marks(rebuilt, mods["c"].load_added_marks(store), warnings) == 1
        page = next(p for p in rebuilt if p.page_id == merges[0]["right_page_id"])
        mark = page.marks[merges[0]["right_mark_index"]]
        assert mark.box == (300, 400, 80, 90) and mark.class_id is None and mark.provenance == "completeness"
        # replaying twice does not duplicate
        assert mods["c"].replay_added_marks(rebuilt, mods["c"].load_added_marks(store), warnings) == 0

    def test_a_page_missing_from_the_build_warns(self, mods):
        pages, _ = _corpus(mods)
        warnings: list[str] = []
        rows = [{"page_id": "tobacco800/gone", "box": [0, 0, 1, 1]}]
        assert mods["c"].replay_added_marks(pages, rows, warnings) == 0
        assert warnings and "gone" in warnings[0]


class TestReApplyingDoesNotGrowTheStore:
    """#4042: re-applying a class re-proposes every box it already contributed."""

    def _rows(self, mods):
        pages, classes = _corpus(mods)
        _, _, _, _, added = mods["c"].apply_completeness(pages, classes, [_row([NOBOX], "0")])
        return added

    def test_saving_the_same_row_twice_writes_it_once(self, mods, tmp_path):
        added = self._rows(mods)
        store = tmp_path / mods["c"].ADDED_MARKS
        mods["c"].save_added_marks(added, store)
        written = mods["c"].save_added_marks(mods["c"].load_added_marks(store) + added, store)

        assert len(written) == 1
        assert mods["c"].load_added_marks(store) == written

    def test_a_differing_note_is_still_the_same_mark_and_the_first_is_kept(self, mods):
        first = self._rows(mods)[0]
        again = dict(first, note="a later re-apply")
        kept = mods["c"].dedupe_added_marks([first, again])
        assert kept == [first]

    def test_a_different_box_or_class_is_a_different_mark(self, mods):
        row = self._rows(mods)[0]
        moved = dict(row, box=[301, 400, 80, 90])
        reclassed = dict(row, class_id="tobacco800/logo_z")
        assert mods["c"].dedupe_added_marks([row, moved, reclassed]) == [row, moved, reclassed]

    def test_a_box_stored_as_a_tuple_matches_the_same_box_as_a_list(self, mods):
        row = self._rows(mods)[0]
        assert mods["c"].dedupe_added_marks([row, dict(row, box=tuple(row["box"]))]) == [row]

    def test_a_store_written_before_the_dedupe_is_tidied_by_the_next_save(self, mods, tmp_path):
        row = self._rows(mods)[0]
        store = tmp_path / mods["c"].ADDED_MARKS
        store.write_text(json.dumps({"marks": [row, row, row]}), encoding="utf-8")
        assert len(mods["c"].save_added_marks(mods["c"].load_added_marks(store), store)) == 1
