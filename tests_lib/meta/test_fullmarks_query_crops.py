"""Hand-chosen extra query crops: candidate ranking, verdicts, and the durable store."""

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
        yield {"q": importlib.import_module("query_crops"), "common": importlib.import_module("sources._common")}
    finally:
        sys.path.remove(str(_FULLMARKS))


def _page(mods, page_id, box, class_id="t/logo_a", path="x.png", w=400, h=500):
    Mark, Page = mods["common"].Mark, mods["common"].Page
    return Page(
        page_id=page_id, source=page_id.split("/")[0], path=path, width=w, height=h, marks=[Mark("logo", box, class_id)]
    )


def _class(page_ids, query_page):
    return {"page_ids": page_ids, "query_page_id": query_page, "query_crop": "q.png"}


class TestCandidates:
    def test_strongest_match_first_one_per_document_and_never_the_query_page(self, mods):
        pages = {
            p.page_id: p
            for p in (
                _page(mods, "tobacco800/aaa11a00", (0, 0, 50, 50)),  # the query page
                _page(mods, "tobacco800/aaa11a00-page02", (0, 0, 90, 90)),  # same document as the query
                _page(mods, "tobacco800/bbb22b00_1", (0, 0, 40, 40)),
                _page(mods, "tobacco800/bbb22b00_2", (0, 0, 80, 80)),  # same document, stronger
                _page(mods, "tobacco800/ccc33c00", (0, 0, 60, 60)),
            )
        }
        meta = _class(sorted(pages), "tobacco800/aaa11a00")
        inl = {
            "tobacco800/aaa11a00-page02": [99, 0],
            "tobacco800/bbb22b00_1": [10, 0],
            "tobacco800/bbb22b00_2": [30, 0],
            "tobacco800/ccc33c00": [20, 0],
        }
        got = mods["q"].candidates("t/logo_a", meta, pages, inl)
        assert [c.page_id for c in got] == ["tobacco800/bbb22b00_2", "tobacco800/ccc33c00"]

    def test_only_tobacco800_ids_are_grouped_into_documents(self, mods):
        q = mods["q"]
        assert (
            q.document_of("tobacco800/aah97e00-page02_1")
            == q.document_of("tobacco800/aah97e00_2")
            == "tobacco800/aah97e00"
        )
        assert q.document_of("staver/stampds-00213") != q.document_of("staver/stampds-00214")
        assert q.document_of("spods/00003") == "spods/00003"

    def test_top_caps(self, mods):
        pages = {f"spods/{i:05d}": _page(mods, f"spods/{i:05d}", (0, 0, 10 + i, 10)) for i in range(20)}
        meta = _class(sorted(pages), "spods/00000")
        assert len(mods["q"].candidates("t/logo_a", meta, pages, {}, top=5)) == 5


class TestApply:
    ROW = {
        "class_id": "t/logo_a",
        "candidates": [{"page_id": "spods/1", "box": [1, 2, 3, 4]}, {"page_id": "spods/2", "box": [5, 6, 7, 8]}],
    }

    def test_chosen_boxes_replace_the_class_entry_in_order(self, mods):
        store = {"t/logo_a": [{"page_id": "old", "box": [0, 0, 1, 1]}]}
        changes, problems = mods["q"].apply_query_crops(
            {"t/logo_a": {}}, [dict(self.ROW, verdict="1,0")], store, reviewer="sam"
        )
        assert problems == [] and len(changes) == 1
        assert [e["page_id"] for e in store["t/logo_a"]] == ["spods/2", "spods/1"]

    def test_blank_skips_none_clears_and_bad_input_is_a_problem(self, mods):
        store = {"t/logo_a": [{"page_id": "old", "box": [0, 0, 1, 1]}]}
        q = mods["q"]
        assert q.apply_query_crops({"t/logo_a": {}}, [dict(self.ROW, verdict="")], store) == ([], [])
        assert store["t/logo_a"][0]["page_id"] == "old"
        q.apply_query_crops({"t/logo_a": {}}, [dict(self.ROW, verdict="none")], store)
        assert store["t/logo_a"] == []
        for bad in ("2", "0,0", "yes"):
            assert q.apply_query_crops({"t/logo_a": {}}, [dict(self.ROW, verdict=bad)], {})[1]


class TestStoreSurvivesARebuild:
    def test_materialise_cuts_crops_sets_the_list_primary_first_and_warns_on_misfits(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "page.png"
        Image.new("RGB", (400, 500), "white").save(img)
        pages = {"spods/1": _page(mods, "spods/1", (10, 10, 50, 40), path=str(img))}
        classes = {"t/logo_a": {"query_crop": "primary.png"}, "t/logo_b": {}}
        store = {
            "t/logo_a": [
                {"page_id": "spods/1", "box": [10, 10, 50, 40]},
                {"page_id": "spods/gone", "box": [0, 0, 5, 5]},
            ]
        }
        mods["q"].save_store(store, tmp_path / "query_crops.json")
        warnings: list[str] = []
        n = mods["q"].materialise(
            classes, mods["q"].load_store(tmp_path / "query_crops.json"), pages, tmp_path / "queries", warnings
        )
        assert n == 1 and len(warnings) == 1 and "gone" in warnings[0]
        crops = classes["t/logo_a"]["query_crops"]
        assert crops[0] == "primary.png" and crops[1].endswith("t__logo_a__q1.png")
        with Image.open(crops[1]) as im:
            assert im.size == (50, 40)
        assert "query_crops" not in classes["t/logo_b"]  # no primary, no list
