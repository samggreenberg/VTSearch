"""#4088: banded UCSF pages reviewed into known negatives, through the existing UCSF applier."""

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
            "b": importlib.import_module("banded_review"),
            "br": importlib.import_module("binary_review"),
            "u": importlib.import_module("ucsf_classes"),
            "common": importlib.import_module("sources._common"),
        }
    finally:
        sys.path.remove(str(_FULLMARKS))


CLS = "ucsf/logo_bat"
BAND = (0, 0, 1240, 370)


def _pages(mods):
    Page, Mark = mods["common"].Page, mods["common"].Mark

    def page(pid, tier="m", banded=True, marks=()):
        meta = {"tier": tier, "industry": "Tobacco" if banded else "Food"}
        if banded:
            meta["letterhead_author"] = "BATCO"
        ms = [Mark("logo", BAND, None, "candidate")] if banded else []
        return Page(
            page_id=pid, source="ucsf", path="x.png", width=1240, height=1680, marks=ms + list(marks), meta=meta
        )

    return {
        "ucsf/pos#0": page("ucsf/pos#0", marks=[Mark("logo", (10, 10, 50, 40), CLS, "ucsf_classes")]),
        "ucsf/rej#0": page("ucsf/rej#0"),
        "ucsf/q#0": page("ucsf/q#0"),
        "ucsf/new#0": page("ucsf/new#0"),
        "ucsf/neg#0": page("ucsf/neg#0"),
        "ucsf/big#0": page("ucsf/big#0", tier="l"),
        "ucsf/food#0": page("ucsf/food#0", banded=False),
    }


def _meta():
    return {
        "source": "ucsf",
        "page_ids": ["ucsf/pos#0"],
        "n_instances": 1,
        "reviewed_negative_page_ids": ["ucsf/rej#0"],
        "query_page_id": "ucsf/q#0",
        "eligible_distractor_sources": ["spods", "staver", "synth", "tobacco800", "ucsf"],
    }


class TestSizing:
    def test_one_positive_alone_at_the_top_scores_one(self, mods):
        assert mods["b"].shortcut_ap(1, 0) == 1.0
        assert mods["b"].shortcut_ap(5, 0) == 1.0

    def test_more_banded_negatives_lower_the_control(self, mods):
        b = mods["b"]
        assert b.shortcut_ap(20, 0) > b.shortcut_ap(20, 20) > b.shortcut_ap(20, 200)

    def test_the_draw_is_the_fewest_pages_that_reach_the_target(self, mods):
        b = mods["b"]
        n = b.pages_needed(20, 3, target=0.2)
        assert b.shortcut_ap(20, 3 + n) < 0.2 <= b.shortcut_ap(20, 3 + n - 1)

    def test_a_class_already_under_target_draws_nothing_and_a_cap_holds(self, mods):
        b = mods["b"]
        assert b.pages_needed(2, 500, target=0.2) == 0
        assert b.pages_needed(50, 0, target=0.1, cap=7) == 7


class TestFrame:
    def test_only_unruled_banded_pages_in_the_tier_are_drawn(self, mods):
        frame = mods["b"].banded_frame(CLS, _meta(), _pages(mods), {"s", "m"})
        # not the positive, the reviewed negative, the query page, a tier-l page or an un-banded page
        assert frame == ["ucsf/neg#0", "ucsf/new#0"]


class TestVotesReachTheApplier:
    """A Good/Bad vote on a banded question lands as a positive or a known negative."""

    def _run(self, mods, votes):
        b, br = mods["b"], mods["br"]
        pages = _pages(mods)
        classes = {CLS: _meta()}
        items = [("ucsf/new#0", "uniform"), ("ucsf/neg#0", "uniform"), ("ucsf/pos#0", "control")]
        qs = [b.question(CLS, i, pid, arm, pages[pid], refs=[]) for i, (pid, arm) in enumerate(items)]
        rows = [
            {
                "task": "ucsf_classes_relation",
                "proposal": "bat",
                "relation": "new",
                "query_crop": "/c.png",
                "query_page_id": "ucsf/q#0",
            },
        ] + [b.slate_row(CLS, q, i, pages[q.page_id]) for i, q in enumerate(qs)]
        questions = {q.filename: {"task": q.task, "key": q.key} for q in qs}
        by_fn = {q.page_id: q.filename for q in qs}
        filled, unanswered = br.translate_ucsf(rows, questions, {by_fn[p]: v for p, v in votes.items()})
        changes, problems, added, negatives, _ = mods["u"].apply_ucsf_classes(
            list(pages.values()), classes, filled, reviewer="sam"
        )
        return classes[CLS], pages, added, negatives, problems, unanswered

    def test_good_is_a_band_located_positive_and_bad_a_reviewed_negative(self, mods):
        meta, pages, added, negatives, problems, unanswered = self._run(
            mods, {"ucsf/new#0": "good", "ucsf/neg#0": "bad", "ucsf/pos#0": "good"}
        )
        assert problems == [] and unanswered == []
        assert meta["page_ids"] == ["ucsf/new#0", "ucsf/pos#0"]
        assert meta["reviewed_negative_page_ids"] == ["ucsf/neg#0", "ucsf/rej#0"]
        assert negatives == {CLS: ["ucsf/neg#0"]}
        # the new positive is boxed by its letterhead band and says so
        assert [(a["page_id"], tuple(a["box"]), a["provenance"]) for a in added] == [
            ("ucsf/new#0", BAND, "ucsf_classes_band")
        ]

    def test_a_missed_control_does_not_turn_a_positive_into_a_negative(self, mods):
        meta, _pages_, added, _neg, problems, _ = self._run(
            mods, {"ucsf/new#0": "bad", "ucsf/neg#0": "bad", "ucsf/pos#0": "bad"}
        )
        assert problems == [] and added == []
        assert "ucsf/pos#0" in meta["page_ids"] and "ucsf/pos#0" not in meta["reviewed_negative_page_ids"]

    def test_an_unanswered_question_leaves_its_page_unreviewed(self, mods):
        meta, _pages_, _added, _neg, problems, unanswered = self._run(mods, {"ucsf/neg#0": "bad"})
        assert problems == [] and len(unanswered) == 2
        assert "ucsf/new#0" not in meta["reviewed_negative_page_ids"] + meta["page_ids"]

    def test_the_task_routes_to_its_own_slate(self, mods):
        src, translator = mods["br"].TRANSLATORS["ucsf_banded"]
        assert src(Path("/c")) == Path("/c/audit/ucsf_banded/verdicts.jsonl")
        assert translator is mods["br"].translate_ucsf
