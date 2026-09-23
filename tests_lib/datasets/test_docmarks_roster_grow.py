"""#4143: growing the roster on a staging corpus, and reviewing the new classes."""

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
        yield {"g": importlib.import_module("roster_grow"), "r": importlib.import_module("roster_review")}
    finally:
        sys.path.remove(str(_DOCMARKS))


class TestStaging:
    def test_stage_copies_the_stores_and_refuses_to_overwrite(self, mods, tmp_path):
        src = tmp_path / "live"
        (src / "queries").mkdir(parents=True)
        (src / "queries" / "a.png").write_bytes(b"x")
        for name in ("corpus.jsonl", "classes.json", "roster.json"):
            (src / name).write_text("{}", encoding="utf-8")
        out = tmp_path / "staging"
        copied = mods["g"].stage(src, out)
        assert set(copied) == {"corpus.jsonl", "classes.json", "roster.json"}
        assert (out / "queries" / "a.png").exists() and (out / "STAGING.md").exists()
        with pytest.raises(SystemExit):
            mods["g"].stage(src, out)

    def test_admit_refuses_the_live_corpus(self, mods, tmp_path):
        with pytest.raises(SystemExit, match="staging"):
            mods["g"].main(["admit", "--corpus", str(tmp_path), "--classes", "x/y"])


class TestIdentityPairs:
    CLASSES = {
        "s/old1": {"source": "s", "on_roster": True},
        "s/old2": {"source": "s", "on_roster": True},
        "s/old3": {"source": "s", "on_roster": True},
        "s/new1": {"source": "s", "on_roster": True},
        "s/new2": {"source": "s", "on_roster": True},
        "t/old": {"source": "t", "on_roster": True},
    }

    def test_nearest_existing_classes_of_the_same_source_and_every_new_pair_once(self, mods):
        sim = {("s/new1", "s/old2"): 0.9, ("s/new1", "s/old1"): 0.5, ("s/new2", "s/old3"): 0.8}
        pairs = mods["r"].identity_pairs(["s/new1", "s/new2"], self.CLASSES, sim, k=1)
        assert pairs == [("s/new1", "s/new2"), ("s/new1", "s/old2"), ("s/new2", "s/old3")]
        assert not any("t/old" in p for p in pairs)


class TestTranslators:
    def test_identity_good_is_same_bad_is_different(self, mods):
        rows = [
            {"left_class_id": "a", "right_class_id": "b", "sheet": "identity__000", "verdict": ""},
            {"left_class_id": "a", "right_class_id": "c", "sheet": "identity__001", "verdict": ""},
            {"left_class_id": "b", "right_class_id": "c", "sheet": "identity__002", "verdict": ""},
        ]
        questions = {
            f"identity__00{i}.jpg": {"task": "roster_identity", "key": {"sheet": f"identity__00{i}"}} for i in range(3)
        }
        out, unanswered = mods["r"].translate_identity(
            rows, questions, {"identity__000.jpg": "good", "identity__001.jpg": "bad"}
        )
        assert [r["verdict"] for r in out] == ["same", "different", ""]
        assert unanswered == ["b vs c: unanswered"]

    def test_membership_is_ok_or_the_bad_indices_and_blank_until_complete(self, mods):
        rows = [{"class_id": "c", "page_ids": ["p0", "p1", "p2"], "verdict": ""}]
        questions = {f"m{i}.jpg": {"task": "roster_membership", "key": {"class_id": "c", "index": i}} for i in range(3)}
        tr = mods["r"].translate_membership
        assert tr(rows, questions, {"m0.jpg": "good", "m1.jpg": "good", "m2.jpg": "good"})[0][0]["verdict"] == "ok"
        assert tr(rows, questions, {"m0.jpg": "good", "m1.jpg": "bad", "m2.jpg": "bad"})[0][0]["verdict"] == "1,2"
        out, unanswered = tr(rows, questions, {"m0.jpg": "good"})
        assert out[0]["verdict"] == "" and unanswered == ["c: 2 of 3 unanswered"]

    def test_completeness_accepts_the_good_candidates_and_skips_classes_not_asked(self, mods):
        rows = [
            {"class_id": "c", "candidates": [{"index": 0}, {"index": 1}, {"index": 2}], "verdict": ""},
            {"class_id": "other", "candidates": [{"index": 0}], "verdict": ""},
        ]
        questions = {
            f"k{i}.jpg": {"task": "roster_completeness", "key": {"class_id": "c", "index": i}} for i in range(3)
        }
        tr = mods["r"].translate_completeness
        out, unanswered = tr(rows, questions, {"k0.jpg": "bad", "k1.jpg": "good", "k2.jpg": "good"})
        assert out[0]["verdict"] == "1,2" and out[1]["verdict"] == "" and unanswered == []
        assert tr(rows, questions, {f"k{i}.jpg": "bad" for i in range(3)})[0][0]["verdict"] == "none"
