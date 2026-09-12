"""Applying a recheck pass to `corrections.json` (#3778).

A recheck exists because a ruling moved a class boundary under rows that were
already cast. The planter ruling (#3784) did that to 80 of them, and 21 came
back Bad -- so this pass is the first thing in the pile that *retires* a human
positive at scale, against a file where 865 of 872 rows say `present: true`.

The properties worth holding it to are all about what it refuses. A recheck
re-answers rows that already exist, so a verdict with no row means the slate and
the file have drifted and the safe reading is not "add it". And the verdicts are
answers to a *named rule* -- the only wording the reviewer ever saw (#3612) -- so
applying them under a different one would apply the old ruling in the new one's
name, which is the exact confusion this issue was filed about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

RULE = "vase not planters"


@pytest.fixture(scope="module")
def ar():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import apply_recheck

    return apply_recheck


def _verdict(iid: int) -> dict:
    return {"filename": f"{iid}.jpg", "label": "good", "md5": "x"}


def _labelset(cls: str = "vase", good: list[int] | None = None, bad: list[int] | None = None, rule: str = RULE) -> dict:
    return {
        "class": cls,
        "kind": "recheck",
        "rule": rule,
        "question": "does this photo contain one under the revised rule?",
        "good": [_verdict(i) for i in good or []],
        "bad": [_verdict(i) for i in bad or []],
    }


def _row(iid: int, cls: str = "vase", present: bool = True) -> dict:
    box = [[0.1, 0.1, 0.2, 0.2]] if present else []
    return {
        "image_id": iid,
        "class": cls,
        "present": present,
        "boxes": box,
        "box_space": "normalised",
        "source": "human_review",
    }


class TestPlan:
    def test_a_bad_verdict_retires_the_row(self, ar):
        rows, stats, problems = ar.plan({"vase": _labelset(bad=[7])}, [_row(7)], {"vase": RULE})

        assert problems == []
        assert stats["rows changed"] == 1
        (row,) = rows
        assert row["present"] is False
        assert row["source"] == ar.SOURCE_RECHECK

    def test_a_retired_row_carries_no_box(self, ar):
        """`apply_corrections` pops the class on a boxless absent; a box would band it."""
        rows, _, _ = ar.plan({"vase": _labelset(bad=[7])}, [_row(7)], {"vase": RULE})

        assert rows[0]["boxes"] == []

    def test_a_retired_row_says_why(self, ar):
        """A ruling is not an annotator error, and an audit has to be able to tell."""
        rows, _, _ = ar.plan({"vase": _labelset(bad=[7])}, [_row(7)], {"vase": RULE})

        assert RULE in rows[0]["note"] and "3778" in rows[0]["note"]

    def test_a_good_verdict_leaves_its_row_exactly_as_it_was(self, ar):
        """Re-confirmation changes nothing: the row and its box were right all along."""
        before = _row(7)
        rows, stats, problems = ar.plan({"vase": _labelset(good=[7])}, [before], {"vase": RULE})

        assert problems == [] and rows == [before]
        assert stats["vase: re-confirmed"] == 1 and not stats["rows changed"]

    def test_it_touches_no_other_class(self, ar):
        """`corrections.json` is shared by every build of this family (#3588)."""
        other = _row(7, "bowl")
        rows, _, problems = ar.plan({"vase": _labelset(bad=[7])}, [_row(7), other], {"vase": RULE})

        assert problems == []
        assert other in rows
        assert [r for r in rows if r["class"] == "vase"][0]["present"] is False

    def test_applying_twice_changes_nothing(self, ar):
        once, _, _ = ar.plan({"vase": _labelset(bad=[7])}, [_row(7)], {"vase": RULE})
        twice, stats, problems = ar.plan({"vase": _labelset(bad=[7])}, once, {"vase": RULE})

        assert problems == [] and twice == once
        assert stats["already retired"] == 1 and not stats["rows changed"]

    def test_the_row_count_never_moves(self, ar):
        rows, _, _ = ar.plan({"vase": _labelset(good=[1, 2], bad=[3])}, [_row(i) for i in (1, 2, 3)], {"vase": RULE})

        assert len(rows) == 3


class TestRefusals:
    """Each of these means the pass and the file disagree about what was reviewed."""

    def test_a_superseded_rule_is_refused(self, ar):
        """The verdicts answer the question the reviewer was shown, and no other."""
        rows, _, problems = ar.plan({"vase": _labelset(bad=[7])}, [_row(7)], {"vase": "vase incl pots and planters"})

        assert len(problems) == 1 and "superseded" in problems[0]
        assert rows == [_row(7)], "a refusal writes nothing"

    def test_a_verdict_with_no_row_is_refused(self, ar):
        """A recheck re-answers existing rows; a new one would invent membership."""
        _, _, problems = ar.plan({"vase": _labelset(bad=[7])}, [], {"vase": RULE})

        assert len(problems) == 1 and "no row to re-answer" in problems[0]

    def test_one_image_answered_both_ways_is_refused(self, ar):
        _, _, problems = ar.plan({"vase": _labelset(good=[7], bad=[7])}, [_row(7)], {"vase": RULE})

        assert any("both good and bad" in p for p in problems)

    def test_a_re_confirmation_of_an_absent_row_is_refused(self, ar):
        """Somebody else already retired it: two passes disagree and neither wins here."""
        _, _, problems = ar.plan({"vase": _labelset(good=[7])}, [_row(7, present=False)], {"vase": RULE})

        assert any("already says absent" in p for p in problems)

    def test_a_labelset_of_another_kind_is_refused(self, ar):
        """A slate asks "is this box one?" -- a different question with the same shape."""
        ls = _labelset(bad=[7])
        ls["kind"] = "slate"

        _, _, problems = ar.plan({"vase": ls}, [_row(7)], {"vase": RULE})

        assert any("not 'recheck'" in p for p in problems)
