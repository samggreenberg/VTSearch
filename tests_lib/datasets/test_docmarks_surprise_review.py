"""#4089: a run's top-ranked presumed negatives become review items, not false positives."""

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
        yield {
            "s": importlib.import_module("surprise_review"),
            "ev": importlib.import_module("eval_retrieval"),
            "br": importlib.import_module("binary_review"),
        }
    finally:
        sys.path.remove(str(_DOCMARKS))


class TestTopPresumed:
    def test_only_presumed_pages_are_kept_with_their_real_rank(self, mods):
        ranked = ["pos", "known", "p1", "pos2", "p2", "p3"]
        got = mods["ev"].top_presumed(ranked, {"p1", "p2", "p3"}, k=2)
        assert got == [{"page_id": "p1", "rank": 3}, {"page_id": "p2", "rank": 5}]

    def test_the_pools_expose_the_presumed_negatives_of_the_headline_pool(self, mods):
        meta = {
            "source": "tobacco800",
            "page_ids": ["tobacco800/a"],
            "eligible_distractor_sources": ["ucsf"],
        }
        pools = mods["ev"].class_pools(
            meta,
            {"tobacco800": ["tobacco800/a", "tobacco800/b"], "ucsf": ["ucsf/food", "ucsf/tob"]},
            {"ucsf/food": "Food", "ucsf/tob": "Tobacco"},
        )
        # tobacco800/b is a known negative (its own, exhaustively checked source);
        # UCSF Tobacco is out; only UCSF Food is presumed.
        assert pools["presumed"] == {"ucsf/food"}
        assert pools["presumed"] <= pools["own_verified"]


class TestPagesToReview:
    HITS = {
        "s": {"c/x": {"siglip": [{"page_id": "u/1", "rank": 4}, {"page_id": "u/2", "rank": 9}]}},
        "m": {
            "c/x": {
                "vlad": [{"page_id": "u/1", "rank": 2}, {"page_id": "u/done", "rank": 1}],
            },
            "c/gone": {"siglip": [{"page_id": "u/9", "rank": 1}]},
        },
    }

    def test_one_question_per_page_best_rank_first_and_ruled_pages_dropped(self, mods):
        classes = {"c/x": {"page_ids": [], "reviewed_negative_page_ids": ["u/done"]}}
        todo = mods["s"].pages_to_review(self.HITS, classes)
        assert list(todo) == ["c/x"]  # a class no longer in the corpus is skipped
        assert [(h["page_id"], h["best_rank"]) for h in todo["c/x"]] == [("u/1", 2), ("u/2", 9)]
        assert sorted(todo["c/x"][0]["found_by"]) == ["m:vlad@2", "s:siglip@4"]

    def test_a_method_at_chance_can_be_left_out(self, mods):
        classes = {"c/x": {"page_ids": []}}
        todo = mods["s"].pages_to_review(self.HITS, classes, methods=["siglip"])
        assert [(h["page_id"], h["best_rank"]) for h in todo["c/x"]] == [("u/1", 4), ("u/2", 9)]


class TestApply:
    def _rows(self, verdicts):
        return [
            {"task": "surprise", "class_id": "c/x", "page_id": p, "arm": arm, "filename": f"{p}.jpg", "verdict": v}
            for p, arm, v in verdicts
        ]

    def test_bad_is_a_known_negative_good_is_excluded_and_controls_only_score(self, mods):
        classes = {"c/x": {"page_ids": ["u/pos"], "excluded_page_ids": ["u/old"]}}
        rows = self._rows(
            [("u/1", "hit", "bad"), ("u/2", "hit", "good"), ("u/pos", "control", "bad"), ("u/3", "hit", "")]
        )
        changes, problems, negatives, exclusions = mods["s"].apply_surprise(classes, rows, reviewer="sam")
        meta = classes["c/x"]
        assert negatives == {"c/x": ["u/1"]} and exclusions == {"c/x": ["u/2"]}
        assert meta["reviewed_negative_page_ids"] == ["u/1"]
        assert meta["excluded_page_ids"] == ["u/2", "u/old"]  # an earlier exclusion survives
        assert meta["page_ids"] == ["u/pos"]  # a missed control never demotes a positive
        assert problems == ["c/x: 1 of 1 planted control(s) missed"]
        assert "box them" in changes[0]

    def test_votes_route_through_the_bank_by_filename(self, mods):
        rows = self._rows([("u/1", "hit", ""), ("u/2", "hit", "")])
        questions = {"u/1.jpg": {"task": "surprise"}, "u/2.jpg": {"task": "surprise"}}
        filled, unanswered = mods["s"].translate_surprise(rows, questions, {"u/1.jpg": "bad"})
        assert [r["verdict"] for r in filled] == ["bad", ""] and unanswered == ["c/x: u/2 unanswered"]
        src, translator = mods["br"].TRANSLATORS["surprise"]
        assert src(Path("/c")) == Path("/c/audit/surprise/verdicts.jsonl")
        assert translator(rows, questions, {"u/2.jpg": "good"})[0][1]["verdict"] == "good"
