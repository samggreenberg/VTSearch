"""The multi-method completeness pass: merging non-SIFT proposals and dropping what is decided."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield {
            "m": importlib.import_module("completeness_multi"),
            "c": importlib.import_module("completeness"),
            "common": importlib.import_module("sources._common"),
        }
    finally:
        sys.path.remove(str(_FULLMARKS))


def _page(mods, page_id, source="tobacco800", marks=()):
    Mark, Page = mods["common"].Mark, mods["common"].Page
    return Page(page_id=page_id, source=source, path="x.png", width=1000, height=2000, marks=[Mark(*m) for m in marks])


def _corpus(mods):
    pages = {
        p.page_id: p
        for p in (
            _page(mods, "tobacco800/member", marks=[("logo", (100, 100, 200, 200), "tobacco800/logo_a")]),
            _page(mods, "tobacco800/other", marks=[("logo", (500, 500, 100, 100), "tobacco800/logo_z")]),
            _page(mods, "tobacco800/rejected", marks=[("logo", (0, 0, 100, 100), None)]),
            _page(mods, "tobacco800/nobox"),
            _page(mods, "tobacco800/checked"),
            _page(mods, "tobacco800/seen"),
            _page(mods, "ucsf/leak", source="ucsf"),
            _page(mods, "tobacco800/sister", marks=[("logo", (0, 0, 100, 100), "tobacco800/logo_rostered")]),
        )
    }
    classes = {
        "tobacco800/logo_rostered": {"on_roster": True, "page_ids": ["tobacco800/sister"]},
        "tobacco800/logo_a": {
            "on_roster": True,
            "kind": "logo",
            "page_ids": ["tobacco800/member"],
            "audit": {"completeness_checked": {"unboxed_rejected_page_ids": ["tobacco800/checked"]}},
        },
        "tobacco800/logo_off": {"on_roster": False, "page_ids": []},
    }
    return pages, classes


SEPARATION = {
    "left_page_id": "tobacco800/rejected",
    "left_mark_index": 0,
    "right_page_id": "tobacco800/member",
    "right_mark_index": 0,
}
REVIEWED = {
    "class_id": "tobacco800/logo_a",
    "verdict": "none",
    "candidates": [{"page_id": "tobacco800/seen", "mark_index": None}],
}


def _merge(mods, proposals, **kw):
    pages, classes = _corpus(mods)
    m = mods["m"]
    done = m.decided(classes, pages, [SEPARATION], [REVIEWED])
    stats: dict = {}
    entries = m.merge_proposals(classes, pages, proposals, done, stats=stats, **kw)
    (entry,) = [e for e in entries if e.class_id == "tobacco800/logo_a"]
    return entry, stats


class TestMerge:
    def test_decided_pages_and_marks_are_dropped(self, mods):
        rows = [
            ["tobacco800/member", 9.0, [100, 100, 200, 200]],  # member
            ["tobacco800/rejected", 8.0, [10, 10, 50, 50]],  # cannot-linked mark
            ["tobacco800/checked", 7.0, [10, 10, 50, 50]],  # rejected unboxed by the SIFT pass
            ["tobacco800/seen", 6.0, [10, 10, 50, 50]],  # already on a reviewed slate
            ["ucsf/leak", 5.0, [0, 0, 10, 10]],  # not an anchor source
            ["tobacco800/nobox", 4.0, None],  # nothing to show
            ["tobacco800/sister", 3.5, [0, 0, 100, 100]],  # another roster class's instance: a merge question
            ["tobacco800/other", 3.0, [510, 510, 60, 60]],  # new: on a non-roster cluster's mark
        ]
        entry, stats = _merge(mods, {"siglip_tiles": {"tobacco800/logo_a": rows}})
        assert [(c.page_id, c.mark_index, c.mark_class_id) for c in entry.candidates] == [
            ("tobacco800/other", 0, "tobacco800/logo_z")
        ]
        assert stats["siglip_tiles"] == {"proposed": 8, "no_box": 1, "decided": 6, "novel": 1}

    def test_agreement_ranks_first_and_names_the_methods(self, mods):
        proposals = {
            "siglip_tiles": {
                "tobacco800/logo_a": [
                    ["tobacco800/nobox", 1.0, [10, 10, 50, 50]],
                    ["tobacco800/other", 0.5, [510, 510, 60, 60]],
                ]
            },
            "dinov3_patches": {"tobacco800/logo_a": [["tobacco800/other", 0.9, [505, 505, 70, 70]]]},
        }
        entry, _ = _merge(mods, proposals)
        assert [c.page_id for c in entry.candidates] == ["tobacco800/other", "tobacco800/nobox"]
        # both proposed it; DINOv3 localises more tightly than a SigLIP tile, so its box is kept
        assert entry.candidates[0].note == "DINO+SIG" and entry.candidates[0].box == (505, 505, 70, 70)
        assert entry.candidates[1].note == "SIG" and entry.candidates[1].mark_index is None

    def test_the_tighter_localiser_supplies_the_box_whatever_the_rank(self, mods):
        proposals = {
            "siglip_tiles": {"tobacco800/logo_a": [["tobacco800/nobox", 1.0, [0, 0, 400, 400]]]},
            "template_ncc": {
                "tobacco800/logo_a": [
                    ["tobacco800/other", 1.0, [0, 0, 5, 5]],
                    ["tobacco800/nobox", 0.2, [40, 40, 60, 60]],
                ]
            },
        }
        entry, _ = _merge(mods, proposals)
        nobox = next(c for c in entry.candidates if c.page_id == "tobacco800/nobox")
        assert nobox.box == (40, 40, 60, 60) and nobox.note == "SIG+NCC"

    def test_a_coarse_box_containing_a_mark_is_that_mark(self, mods):
        pages, _ = _corpus(mods)
        m = mods["m"]
        assert m.mark_under(pages["tobacco800/other"], (400, 400, 400, 400)) == 0  # tile around the logo
        assert m.mark_under(pages["tobacco800/other"], (510, 510, 60, 60)) == 0  # tight box inside it
        assert m.mark_under(pages["tobacco800/other"], (0, 0, 520, 520)) is None  # clips a corner only

    def test_top_caps(self, mods):
        rows = [["tobacco800/nobox", 1.0, [10, 10, 50, 50]], ["tobacco800/other", 0.5, [510, 510, 60, 60]]]
        entry, _ = _merge(mods, {"ocr_text": {"tobacco800/logo_a": rows}}, top=1)
        assert [c.page_id for c in entry.candidates] == ["tobacco800/nobox"]


class TestInterface:
    def test_proposals_round_trip_and_a_method_twice_is_refused(self, mods, tmp_path):
        m = mods["m"]
        path = m.proposals_path(tmp_path, "ncc")
        m.save_proposals("ncc", {"c/x": [("p/1", 0.7, (1, 2, 3, 4)), ("p/2", 0.1, None)]}, path)
        assert m.load_proposals([path]) == {"ncc": {"c/x": [["p/1", 0.7, [1, 2, 3, 4]], ["p/2", 0.1, None]]}}
        with pytest.raises(ValueError):
            m.load_proposals([path, path])

    def test_verdict_rows_apply_through_the_completeness_applier_without_erasing_the_first_pass(self, mods):
        pages, classes = _corpus(mods)
        m, c = mods["m"], mods["c"]
        proposals = {"siglip_tiles": {"tobacco800/logo_a": [["tobacco800/nobox", 1.0, [10, 10, 50, 50]]]}}
        (entry,) = [
            e
            for e in m.merge_proposals(classes, pages, proposals, m.decided(classes, pages, [], []))
            if e.class_id == "tobacco800/logo_a"
        ]
        row = m.verdict_row(entry)
        assert row["pass"] == "multi" and row["candidates"][0]["methods"] == "SIG"
        row["verdict"] = "none"
        _, problems, merges, _, _ = c.apply_completeness(list(pages.values()), classes, [row])
        assert problems == [] and merges == []
        audit = classes["tobacco800/logo_a"]["audit"]
        assert audit["completeness_checked"]["unboxed_rejected_page_ids"] == ["tobacco800/checked"]
        assert audit["completeness_checked_multi"]["unboxed_rejected_page_ids"] == ["tobacco800/nobox"]
        # and the next merge treats both passes' rejections as decided
        done = m.decided(classes, pages, [], [])
        assert done["tobacco800/logo_a"].rejected_unboxed_pages == {"tobacco800/checked", "tobacco800/nobox"}
