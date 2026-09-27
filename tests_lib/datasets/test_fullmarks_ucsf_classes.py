"""UCSF band classes as roster proposals (#3921, #3922): the parts a verdict passes through.

No matcher runs here; the GRID slate is the measurement.  What is checked is
that a proposal names the band class and roster class it claims to, that the
auto crop lands where the member matches agree, and that a sheet's answer means
what the sheet says it means.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield {
            "u": importlib.import_module("ucsf_classes"),
            "c": importlib.import_module("completeness"),
            "common": importlib.import_module("sources._common"),
            "roster": importlib.import_module("roster"),
            "ev": importlib.import_module("eval_retrieval"),
        }
    finally:
        sys.path.remove(str(_FULLMARKS))


def _write(tmp_path, proposals):
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps({"proposals": proposals}), encoding="utf-8")
    return path


class TestProposals:
    def test_the_committed_proposals_name_on_roster_tobacco800_classes(self, mods):
        raw = json.loads(mods["u"].PROPOSALS.read_text(encoding="utf-8"))
        roster = {p["roster_class"]: {"on_roster": True} for p in raw["proposals"] if p.get("roster_class")}
        proposals = mods["u"].load_proposals(mods["u"].PROPOSALS, roster)
        assert {p.relation for p in proposals} == {"extends", "new"}
        assert all(p.roster_class.startswith("tobacco800/") for p in proposals if p.relation == "extends")
        # the BAT leaf split across two author queries (#3902) is one proposal
        (bat,) = [p for p in proposals if len(p.components) > 1]
        assert {a for a, _ in bat.components} == {"AMERICAN TOBACCO", "BATCO"}

    def test_an_extension_of_an_off_roster_class_is_refused(self, mods, tmp_path):
        path = _write(
            tmp_path, [{"name": "x", "components": [["RJR", 1]], "relation": "extends", "roster_class": "t/gone"}]
        )
        with pytest.raises(ValueError, match="not an on-roster class"):
            mods["u"].load_proposals(path, {"t/gone": {"on_roster": False}})

    def test_a_component_claimed_twice_and_a_bad_relation_are_refused(self, mods, tmp_path):
        path = _write(
            tmp_path,
            [
                {"name": "a", "components": [["RJR", 2]], "relation": "new"},
                {"name": "b", "components": [["RJR", 2]], "relation": "maybe"},
            ],
        )
        with pytest.raises(ValueError) as err:
            mods["u"].load_proposals(path)
        assert "claimed by 2" in str(err.value) and "relation must be" in str(err.value)

    def test_a_new_class_cannot_name_a_roster_class(self, mods, tmp_path):
        path = _write(tmp_path, [{"name": "a", "components": [["RJR", 1]], "relation": "new", "roster_class": "t/x"}])
        with pytest.raises(ValueError, match="cannot name"):
            mods["u"].load_proposals(path)


class TestBandClasses:
    def test_rank_is_by_size_within_the_author(self, mods):
        e = np.eye(3, dtype=np.float32)
        # RJR: three bands on axis 0 and two on axis 1; PM: four on axis 2
        vecs = np.vstack([e[0]] * 3 + [e[1]] * 2 + [e[2]] * 4)
        authors = np.array(["RJR"] * 5 + ["PM"] * 4)
        assert mods["u"].component_members(authors, vecs, "RJR", 1, 0.1).tolist() == [0, 1, 2]
        assert mods["u"].component_members(authors, vecs, "RJR", 2, 0.1).tolist() == [3, 4]
        assert mods["u"].component_members(authors, vecs, "PM", 1, 0.1).tolist() == [5, 6, 7, 8]
        with pytest.raises(ValueError, match="rank 3"):
            mods["u"].component_members(authors, vecs, "RJR", 3, 0.1)

    def test_medoid_is_the_most_central_band(self, mods):
        vecs = np.array([[1, 0], [0.8, 0.6], [0.6, 0.8], [0, 1]], dtype=np.float32)
        assert mods["u"].medoid(vecs) in (1, 2)


class TestConsensusCrop:
    def _logo_matches(self, n, rng):
        # every match puts inliers on a logo at x 0.10-0.20, y 0.2-0.7; each also has its own scattered text hits
        sets = []
        for _ in range(n):
            logo = np.column_stack([rng.uniform(0.10, 0.20, 80), rng.uniform(0.2, 0.7, 80)])
            noise = np.column_stack([rng.uniform(0.3, 1.0, 6), rng.uniform(0, 1, 6)])
            sets.append(np.vstack([logo, noise]))
        return sets

    def test_the_box_is_where_the_matches_agree(self, mods):
        rng = np.random.default_rng(0)
        (x0, y0, x1, y1), support = mods["u"].consensus_box(self._logo_matches(20, rng))
        assert x0 <= 0.10 and x1 >= 0.20 and x1 - x0 < 0.2
        assert y0 <= 0.25 and y1 >= 0.65
        assert support >= 15

    def test_one_dense_match_elsewhere_does_not_move_it(self, mods):
        rng = np.random.default_rng(1)
        sets = self._logo_matches(10, rng)
        sets.append(np.column_stack([rng.uniform(0.6, 0.9, 5000), rng.uniform(0, 1, 5000)]))
        (x0, _, x1, _), _ = mods["u"].consensus_box(sets)
        assert x1 <= 0.25

    def test_nothing_to_vote_on(self, mods):
        assert mods["u"].consensus_box([]) is None
        assert mods["u"].consensus_box([np.zeros((0, 2))]) is None

    def test_weak_or_wide_crops_are_flagged(self, mods):
        u = mods["u"]
        assert u.crop_flags((0.1, 0, 0.3, 1), u.MIN_SUPPORT) == []
        assert len(u.crop_flags((0.0, 0, 0.9, 1), u.MIN_SUPPORT - 1)) == 2

    def test_band_normalised_box_to_page_pixels(self, mods):
        u = mods["u"]
        # a 1240 x 1680 page: the band is 1240 x 420
        assert u.norm_to_page_box((0.5, 0.5, 0.25, 0.0), 1240, 1680) == [310, 0, 310, 210]


class TestVerdicts:
    @pytest.mark.parametrize(
        ("raw", "want"),
        [("all", [0, 1, 2, 3]), ("none", []), ("2, 0", [0, 2]), ("all but 1,3", [0, 2]), ("", None)],
    )
    def test_the_grammar(self, mods, raw, want):
        assert mods["u"].parse_verdict(raw, 4) == (want, None)

    @pytest.mark.parametrize("raw", ["yes", "4", "all but", "all but x"])
    def test_garbage_is_an_error(self, mods, raw):
        accepted, error = mods["u"].parse_verdict(raw, 4)
        assert accepted is None and error


class TestSheets:
    def test_bins_are_half_open_and_drop_the_floor(self, mods):
        scored = [("a", 80), ("b", 79), ("c", 40), ("d", 8), ("e", 7), ("f", 200)]
        got = {mods["u"].bin_label(lo, hi): [p for p, _ in items] for lo, hi, items in mods["u"].bins(scored)}
        assert got == {"80+": ["f", "a"], "40-79": ["b", "c"], "20-39": [], "12-19": [], "8-11": ["d"]}

    def test_a_bin_that_fits_is_shown_whole_and_a_large_one_sampled_in_order(self, mods):
        u = mods["u"]
        items = [(str(i), 100 - i) for i in range(40)]
        assert u.sample_sheet(items[:5], 18, random.Random(0)) == items[:5]
        picked = u.sample_sheet(items, 18, random.Random(0), key=lambda t: -t[1])
        assert len(picked) == 18 and picked == sorted(picked, key=lambda t: -t[1])

    def test_a_merged_class_shows_both_halves(self, mods):
        picked = mods["u"].spread_sample([list(range(440)), list(range(1000, 1103))], 18, random.Random(0))
        assert len(picked) == 18 and sum(1 for p in picked if p >= 1000) == 9
        assert len(mods["u"].spread_sample([[1, 2], [3]], 18, random.Random(0))) == 3

    def test_best_hit_takes_the_stronger_query(self, mods):
        hits = {"crop": {"p": [12, 0, 0, 0.1, 0.1]}, "roster:x": {"p": [30, 0.2, 0.2, 0.4, 0.4]}}
        assert mods["u"].best_hit(hits, ["crop", "roster:x"], "p") == (30, [0.2, 0.2, 0.4, 0.4])
        assert mods["u"].best_hit(hits, ["crop"], "q") == (0, None)

    def test_a_collapsed_fit_does_not_count(self, mods):
        # a 1240 x 1680 page (band 1240 x 420) and a query 300 x 100 px cut from a 1240-wide page
        expect = {"crop": (300 / 1240, 100 / 1240)}
        hits = {"crop": {"p": [93, 0.5, 0.8, 0.514, 0.826], "q": [20, 0.1, 0.2, 0.3, 0.5]}}
        # 17 x 11 px carrying 93 inliers: collapsed
        assert mods["u"].best_hit(hits, ["crop"], "p", expect, (1240, 1680)) == (0, None)
        assert mods["u"].best_hit(hits, ["crop"], "p") == (93, [0.5, 0.8, 0.514, 0.826])
        # 248 x 126 px: a mark
        assert mods["u"].best_hit(hits, ["crop"], "q", expect, (1240, 1680))[0] == 20

    def test_all_never_reaches_past_the_shown_cells(self, mods):
        cells = [{"index": i} for i in range(4)]
        rows = [
            {
                "task": "ucsf_classes",
                "proposal": "p",
                "part": "a_members",
                "sheet": "a",
                "cells": cells,
                "verdict": "all",
            },
            # one sheet of a 40-page bin: 'all but 2' is 3 pages, not 30
            {
                "task": "ucsf_classes",
                "proposal": "p",
                "part": "b_8-11",
                "sheet": "b",
                "cells": cells,
                "n_population": 40,
                "verdict": "all but 2",
            },
            {
                "task": "ucsf_classes",
                "proposal": "p",
                "part": "c_missed",
                "sheet": "c",
                "cells": cells,
                "n_population": 4,
                "verdict": "",
            },
            {"task": "ucsf_classes", "proposal": "p", "part": "b_80+", "sheet": "d", "cells": cells, "verdict": "7"},
            {"task": "ucsf_classes_relation", "proposal": "p", "relation": "new"},
        ]
        counts, problems = mods["u"].tally(rows)
        assert counts["p"] == {"sheets": 4, "answered": 2, "accepted": 7, "rejected": 1, "unreviewed": 4}
        assert len(problems) == 1 and problems[0].startswith("d:")

    def test_every_page_of_a_bin_gets_a_sheet_and_the_first_is_the_old_draw(self, mods):
        u = mods["u"]
        items = [(f"p{i:03d}", 200 - i) for i in range(40)]
        key = lambda t: (-t[1], t[0])  # noqa: E731
        first = u.sample_sheet(items, 18, random.Random(7), key=key)
        sheets = u.sheet_order(items, 18, random.Random(7), key=key)
        assert sheets[0] == first
        assert [len(s) for s in sheets] == [18, 18, 4]
        assert sorted(x for s in sheets for x in s) == sorted(items)
        assert u.sheet_order(items[:3], 18, random.Random(7), key=key) == [items[:3]]

    def test_a_rerender_keeps_answers_only_where_the_pages_are_unchanged(self, mods):
        old = [
            dict(_sheet("a.png", "p", ["x", "y"], "all"), suggestion="all", uncertain=True, note="n"),
            dict(_sheet("b.png", "p", ["x", "y"], "0"), suggestion="0"),
            dict(_relation("p", "new"), uncertain=True, note="r"),
        ]
        new = [_sheet("a.png", "p", ["x", "y"], ""), _sheet("b.png", "p", ["x", "z"], ""), _relation("p", "")]
        for row in new:
            row.update(uncertain=False, note="")
        for row in new[:2]:
            row.update(suggestion="")
        assert mods["u"].carry_answers(old, new) == 2
        assert (new[0]["verdict"], new[0]["suggestion"], new[0]["uncertain"], new[0]["note"]) == (
            "all",
            "all",
            True,
            "n",
        )
        assert new[1]["verdict"] == "" and new[1]["suggestion"] == ""
        assert new[2]["relation"] == "new" and new[2]["note"] == "r"

    def test_render_takes_a_proposal_reference_row_and_title(self, mods, tmp_path):
        from PIL import Image

        Page = mods["common"].Page
        img = tmp_path / "page.png"
        Image.new("RGB", (400, 600), "white").save(img)
        pages = {"ucsf/p#0": Page(page_id="ucsf/p#0", source="ucsf", path=str(img), width=400, height=600, marks=[])}
        c = mods["c"]
        entry = c.ClassCandidates("bat_leaf__a_members", candidates=[c.Candidate("ucsf/p#0", 12, box=(10, 10, 50, 40))])
        (path,) = c.render(
            entry,
            {},
            pages,
            tmp_path,
            refs=[("CROP", Image.new("RGB", (60, 30)))],
            title="bat_leaf (a)",
            unboxed_label="",
        )
        assert path.name == "bat_leaf__a_members_00.png" and path.exists()


def _ucsf_pages(mods):
    Page = mods["common"].Page
    mk = lambda pid, src: Page(page_id=pid, source=src, path="x.png", width=1240, height=1680, marks=[])  # noqa: E731
    return [mk("ucsf/a#0", "ucsf"), mk("ucsf/b#0", "ucsf"), mk("ucsf/c#0", "ucsf"), mk("tobacco800/t", "tobacco800")]


def _sheet(sheet, proposal, page_ids, verdict, located=True):
    cells = [
        {"index": i, "page_id": pid, "inliers": 30, "box": [10, 10, 50, 40], "located": located}
        for i, pid in enumerate(page_ids)
    ]
    return {
        "task": "ucsf_classes",
        "proposal": proposal,
        "part": "b_20-39",
        "sheet": sheet,
        "cells": cells,
        "verdict": verdict,
    }


def _relation(proposal, relation):
    return {
        "task": "ucsf_classes_relation",
        "proposal": proposal,
        "relation": relation,
        "suggested_relation": "new",
        "query_crop": "/crops/x.png",
        "query_page_id": "ucsf/a#0",
    }


class TestApply:
    ROSTER = "tobacco800/logo_x"

    def _classes(self):
        return {
            self.ROSTER: {
                "on_roster": True,
                "source": "tobacco800",
                "kind": "logo",
                "page_ids": ["tobacco800/t"],
                "audit": {},
            }
        }

    def test_an_extension_gains_accepted_pages_and_records_rejected_ones(self, mods):
        pages, classes = _ucsf_pages(mods), self._classes()
        rows = [_relation("lor", f"extends {self.ROSTER}"), _sheet("s0", "lor", ["ucsf/a#0", "ucsf/b#0"], "0")]
        changes, problems, added, negatives, excluded = mods["u"].apply_ucsf_classes(
            pages, classes, rows, reviewer="sam"
        )
        assert problems == []
        meta = classes[self.ROSTER]
        assert meta["page_ids"] == ["tobacco800/t", "ucsf/a#0"] and meta["n_instances"] == 2
        assert meta["reviewed_negative_page_ids"] == ["ucsf/b#0"] and negatives == {self.ROSTER: ["ucsf/b#0"]}
        assert pages[0].marks[0].class_id == self.ROSTER and pages[1].marks == []
        assert added[0]["class_id"] == self.ROSTER and added[0]["provenance"] == "ucsf_classes"

    def test_a_new_class_is_created_and_an_unlocated_accept_is_tagged_band(self, mods):
        pages, classes = _ucsf_pages(mods), self._classes()
        rows = [_relation("bat", "new"), _sheet("s0", "bat", ["ucsf/c#0"], "all", located=False)]
        _, problems, added, _, _ = mods["u"].apply_ucsf_classes(pages, classes, rows)
        assert problems == []
        meta = classes["ucsf/logo_bat"]
        assert meta["source"] == "ucsf" and meta["on_roster"] and meta["page_ids"] == ["ucsf/c#0"]
        # v4.1: UCSF is an eligible source, and the per-page rule keeps its Tobacco industry out.
        assert "ucsf" in meta["eligible_distractor_sources"] and meta["query_crop"] == "/crops/x.png"
        assert added[0]["provenance"] == "ucsf_classes_band"

    def test_suggestions_and_unanswered_sheets_are_never_applied(self, mods):
        pages, classes = _ucsf_pages(mods), self._classes()
        row = dict(_sheet("s0", "lor", ["ucsf/a#0"], ""), suggestion="all")
        rel = dict(_relation("lor", ""), suggested_relation=f"extends {self.ROSTER}")
        assert mods["u"].apply_ucsf_classes(pages, classes, [rel, row]) == ([], [], [], {}, {})
        assert classes == self._classes()

    def test_a_page_accepted_on_one_sheet_is_not_a_negative_from_another(self, mods):
        pages, classes = _ucsf_pages(mods), self._classes()
        rows = [
            _relation("lor", f"extends {self.ROSTER}"),
            _sheet("a", "lor", ["ucsf/a#0"], "all"),
            _sheet("b", "lor", ["ucsf/a#0", "ucsf/b#0"], "none"),
        ]
        _, problems, added, negatives, _ = mods["u"].apply_ucsf_classes(pages, classes, rows)
        assert problems == [] and len(added) == 1
        assert negatives[self.ROSTER] == ["ucsf/b#0"]

    def test_a_blank_relation_with_answered_sheets_and_an_unknown_class_are_problems(self, mods):
        pages, classes = _ucsf_pages(mods), self._classes()
        _, problems, *_ = mods["u"].apply_ucsf_classes(
            pages, classes, [_relation("lor", ""), _sheet("s", "lor", ["ucsf/a#0"], "all")]
        )
        assert problems and "relation is blank" in problems[0]
        _, problems, *_ = mods["u"].apply_ucsf_classes(pages, classes, [_relation("x", "extends tobacco800/nope")])
        assert problems and "not an on-roster class" in problems[0]

    def test_an_accepted_tobacco800_page_for_a_new_class_is_excluded_not_a_negative(self, mods):
        pages, classes = _ucsf_pages(mods), self._classes()
        rows = [_relation("bat", "new"), _sheet("s", "bat", ["tobacco800/t", "ucsf/a#0", "ucsf/b#0"], "0,1")]
        _, problems, added, negatives, excluded = mods["u"].apply_ucsf_classes(pages, classes, rows)
        meta = classes["ucsf/logo_bat"]
        assert problems == [] and [a["page_id"] for a in added] == ["ucsf/a#0"] and pages[3].marks == []
        assert meta["page_ids"] == ["ucsf/a#0"] and meta["excluded_page_ids"] == ["tobacco800/t"]
        assert excluded == {"ucsf/logo_bat": ["tobacco800/t"]} and negatives == {"ucsf/logo_bat": ["ucsf/b#0"]}
        pools = mods["ev"].class_pools(
            meta, {"tobacco800": ["tobacco800/t", "tobacco800/u"], "ucsf": ["ucsf/a#0", "ucsf/b#0"]}, {}
        )
        assert (
            "tobacco800/t" not in pools["own_verified"] | pools["eligible"] and "tobacco800/u" in pools["own_verified"]
        )


class TestPoolsAfterApply:
    """Decisions 3 and 4: only reviewed UCSF pages enter a class's pools."""

    PAGES = {
        "tobacco800": ["tobacco800/t", "tobacco800/u"],
        "ucsf": ["ucsf/acc", "ucsf/rej", "ucsf/unseen", "ucsf/food"],
        "spods": ["spods/x"],
    }
    INDUSTRY = {"ucsf/acc": "Tobacco", "ucsf/rej": "Tobacco", "ucsf/unseen": "Tobacco", "ucsf/food": "Food"}

    def test_a_ucsf_class_keeps_only_reviewed_ucsf_pages_even_in_its_own_source_pool(self, mods):
        meta = {
            "source": "ucsf",
            "page_ids": ["ucsf/acc"],
            "reviewed_negative_page_ids": ["ucsf/rej"],
            "eligible_distractor_sources": ["spods", "staver", "synth", "tobacco800"],
        }
        pools = mods["ev"].class_pools(meta, self.PAGES, self.INDUSTRY)
        assert pools["positives"] == {"ucsf/acc"}
        for name in ("own_verified", "eligible"):
            assert "ucsf/rej" in pools[name]
            assert not {"ucsf/unseen", "ucsf/food"} & pools[name]
        assert {"tobacco800/t", "spods/x"} <= pools["own_verified"]

    def test_a_v41_ucsf_class_takes_unbanded_industries_but_no_unreviewed_tobacco_page(self, mods):
        # The v4.1 rule (2026-09-22 contamination check): UCSF's un-banded
        # industries are presumed negatives; an unreviewed Tobacco page is not.
        meta = {
            "source": "ucsf",
            "page_ids": ["ucsf/acc"],
            "reviewed_negative_page_ids": ["ucsf/rej"],
            "eligible_distractor_sources": ["spods", "staver", "synth", "tobacco800", "ucsf"],
        }
        pools = mods["ev"].class_pools(meta, self.PAGES, self.INDUSTRY)
        for name in ("own_verified", "eligible"):
            assert {"ucsf/rej", "ucsf/food"} <= pools[name]
            assert "ucsf/unseen" not in pools[name]

    def test_a_ucsf_class_with_no_review_record_gets_no_ucsf_negatives(self, mods):
        meta = {"source": "ucsf", "page_ids": ["ucsf/acc"], "eligible_distractor_sources": ["spods", "tobacco800"]}
        own = mods["roster"].eligible_pages(
            meta, self.PAGES, verified_negative_sources=["ucsf"], industry_of=self.INDUSTRY
        )
        assert not [p for p in own["known_negative"] + own["presumed_negative"] if p.startswith("ucsf/")]

    def test_an_extended_tobacco800_class_takes_accepted_and_rejected_ucsf_pages_only(self, mods):
        meta = {
            "source": "tobacco800",
            "page_ids": ["tobacco800/t", "ucsf/acc"],
            "reviewed_negative_page_ids": ["ucsf/rej"],
            "eligible_distractor_sources": ["spods", "staver", "synth", "ucsf"],
        }
        pools = mods["ev"].class_pools(meta, self.PAGES, self.INDUSTRY)
        assert pools["positives"] == {"tobacco800/t", "ucsf/acc"}
        for name in ("own_verified", "eligible"):
            assert "ucsf/rej" in pools[name] and "ucsf/unseen" not in pools[name]
            assert "ucsf/food" in pools[name]  # other industries are unchanged
        assert "tobacco800/u" in pools["own_verified"]

    def test_the_stores_survive_a_rebuild(self, mods, tmp_path):
        r, c = mods["roster"], mods["c"]
        store = tmp_path / r.REVIEWED_NEGATIVES
        r.save_reviewed_negatives(
            {"ucsf/logo_bat": ["ucsf/rej", "ucsf/acc"]}, store, {"ucsf/logo_bat": ["tobacco800/t"]}
        )
        rebuilt = {"ucsf/logo_bat": {"page_ids": ["ucsf/acc"]}, "spods/x": {"page_ids": []}}
        assert (
            r.attach_reviewed_negatives(
                rebuilt, r.load_reviewed_negatives(store), r.load_reviewed_negatives(store, "excluded")
            )
            == 1
        )
        assert rebuilt["ucsf/logo_bat"]["reviewed_negative_page_ids"] == ["ucsf/rej"]
        assert rebuilt["ucsf/logo_bat"]["excluded_page_ids"] == ["tobacco800/t"]
        assert "reviewed_negative_page_ids" not in rebuilt["spods/x"]
        # a UCSF mark keeps its class on replay (nothing clusters UCSF); a clustered source's does not
        pages = _ucsf_pages(mods)
        rows = [
            {"page_id": "ucsf/a#0", "box": [1, 2, 3, 4], "class_id": "ucsf/logo_bat", "provenance": "ucsf_classes"},
            {"page_id": "tobacco800/t", "box": [1, 2, 3, 4], "class_id": "tobacco800/logo_x"},
        ]
        assert c.replay_added_marks(pages, rows) == 2
        assert pages[0].marks[0].class_id == "ucsf/logo_bat" and pages[3].marks[0].class_id is None
