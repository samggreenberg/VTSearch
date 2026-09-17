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

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield {
            "u": importlib.import_module("ucsf_classes"),
            "c": importlib.import_module("completeness"),
            "common": importlib.import_module("sources._common"),
        }
    finally:
        sys.path.remove(str(_DOCMARKS))


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

    def test_tally_estimates_a_sampled_bin_from_its_sheet(self, mods):
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
            # 3 of 4 shown carry the mark, from a bin of 40: an estimate of 30
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
        assert counts["p"] == {"sheets": 4, "answered": 2, "shown": 8, "accepted": 7, "estimated": 30.0, "exact": False}
        assert len(problems) == 1 and problems[0].startswith("d:")

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
