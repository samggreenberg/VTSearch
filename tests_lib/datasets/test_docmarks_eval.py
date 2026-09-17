"""The DocMarks roster evaluation (#3904): what gets scored, and how it is scored.

The embedders themselves are not exercised here -- a GRID run does that.  What a
laptop can check is the part that decides whether the numbers mean anything:
which pages a class is ranked against, and the arithmetic over the ranking.
"""

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
            "roster": importlib.import_module("roster"),
            "eval": importlib.import_module("eval_retrieval"),
        }
    finally:
        sys.path.remove(str(_DOCMARKS))


class TestTheContaminationRuleIsAppliedPerPage:
    """A Tobacco800 class may meet UCSF Food, never UCSF Tobacco."""

    PAGES = {
        "tobacco800": ["tobacco800/a", "tobacco800/b"],
        "ucsf": ["ucsf/food1", "ucsf/tob1", "ucsf/unknown"],
        "spods": ["spods/x"],
    }
    INDUSTRY = {"ucsf/food1": "Food", "ucsf/tob1": "Tobacco", "ucsf/unknown": None}

    def _meta(self):
        # What build_corpus actually writes for a Tobacco800 class: the source
        # list can only say `ucsf`, which is the whole of the problem.
        return {
            "source": "tobacco800",
            "page_ids": ["tobacco800/a"],
            "eligible_distractor_sources": ["spods", "staver", "synth", "ucsf"],
        }

    def test_ucsf_tobacco_pages_are_never_scored_for_a_tobacco800_class(self, mods):
        split = mods["roster"].eligible_pages(self._meta(), self.PAGES, industry_of=self.INDUSTRY)
        assert "ucsf/tob1" not in split["presumed_negative"]
        assert "ucsf/food1" in split["presumed_negative"]

    def test_same_source_pages_stay_out_unless_verified(self, mods):
        split = mods["roster"].eligible_pages(self._meta(), self.PAGES, industry_of=self.INDUSTRY)
        assert "tobacco800/b" not in split["presumed_negative"] + split["known_negative"]
        assert split["presumed_negative"] == ["spods/x", "ucsf/food1", "ucsf/unknown"]

    def test_a_spods_class_keeps_all_of_ucsf(self, mods):
        meta = {"source": "spods", "page_ids": ["spods/x"], "eligible_distractor_sources": ["tobacco800", "ucsf"]}
        split = mods["roster"].eligible_pages(meta, self.PAGES, industry_of=self.INDUSTRY)
        assert set(split["presumed_negative"]) == {
            "tobacco800/a",
            "tobacco800/b",
            "ucsf/food1",
            "ucsf/tob1",
            "ucsf/unknown",
        }

    def test_the_eval_pools_use_the_per_page_rule_and_drop_the_query_page(self, mods):
        meta = dict(self._meta(), page_ids=["tobacco800/a", "tobacco800/b"], query_page_id="tobacco800/a")
        pools = mods["eval"].class_pools(meta, self.PAGES, self.INDUSTRY)
        assert pools["positives"] == {"tobacco800/b"}
        assert "tobacco800/a" not in pools["eligible"] | pools["naive"]
        assert "ucsf/tob1" not in pools["eligible"]
        assert "ucsf/tob1" in pools["naive"]


class TestMetrics:
    def test_average_precision_by_hand(self, mods):
        # hits at ranks 1 and 3: (1/1 + 2/3) / 2
        assert mods["eval"].average_precision(["p", "n", "q", "n2"], {"p", "q"}) == pytest.approx(5 / 6)

    def test_a_positive_never_ranked_counts_against_ap(self, mods):
        assert mods["eval"].average_precision(["p", "n"], {"p", "absent"}) == pytest.approx(0.5)

    def test_recall_and_first_hit(self, mods):
        ranked = ["n1", "n2", "p1", "n3", "p2"]
        assert mods["eval"].recall_at(ranked, {"p1", "p2"}, 3) == 0.5
        assert mods["eval"].first_hit(ranked, {"p1", "p2"}) == 3
        assert mods["eval"].first_hit(ranked, {"zz"}) is None

    def test_ranking_is_deterministic_on_ties(self, mods):
        assert mods["eval"].rank_pool({"b": 1.0, "a": 1.0, "c": 2.0}, ["a", "b", "c"]) == ["c", "a", "b"]


class TestRerankGoesThroughTheAppPath:
    def test_a_verified_candidate_overtakes_a_stronger_stage1_score(self, mods):
        from vtscore.media.structural import MatchStats, StructuralFeatures

        import numpy as np

        feats = StructuralFeatures(keypoints=np.zeros((5, 2), np.float32), descriptors=np.zeros((5, 128), np.float32))

        class Matcher:
            def verify(self, template, candidate):
                good = candidate is feats_match
                return MatchStats(model_ok=good, inlier_count=40 if good else 0, inlier_ratio=0.9 if good else 0.0)

        feats_match = StructuralFeatures(
            keypoints=np.ones((5, 2), np.float32), descriptors=np.ones((5, 128), np.float32)
        )
        ranked = ["decoy", "real", "tail"]
        scores = {"decoy": 0.9, "real": 0.5, "tail": 0.1}
        out = mods["eval"].rerank(ranked, scores, {"decoy": feats, "real": feats_match}, feats, Matcher(), k=2)
        assert out == ["real", "decoy", "tail"]

    def test_no_template_leaves_stage1_alone(self, mods):
        assert mods["eval"].rerank(["a", "b"], {"a": 1.0, "b": 0.0}, {}, None, object(), k=2) == ["a", "b"]
