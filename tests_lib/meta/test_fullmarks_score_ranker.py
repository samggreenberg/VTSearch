"""#4108: bring-your-own-ranker scoring for FullMarks, and the frozen version manifest."""

from __future__ import annotations

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
        yield {
            "s": importlib.import_module("score_ranker"),
            "ev": importlib.import_module("eval_retrieval"),
            "splg": importlib.import_module("eval_splg_rank"),
        }
    finally:
        sys.path.remove(str(_FULLMARKS))


class TestScoresFile:
    def test_round_trip_through_gzip(self, mods, tmp_path):
        s = mods["s"]
        scores = {"c/a": {"p1": 0.5, "p2": -1.25}, "c/b": {"p1": 3.0}}
        path = tmp_path / "x.csv.gz"
        assert s.write_scores(scores, path) == 3
        assert s.read_scores(path) == scores
        assert s.read_scores(path, classes=["c/b"]) == {"c/b": {"p1": 3.0}}

    @pytest.mark.parametrize(
        "body, match",
        [
            ("class_id,page_id\nc,p\n", "header lacks"),
            ("class_id,page_id,score\nc,p,1\nc,p,2\n", "scored twice"),
            ("class_id,page_id,score\nc,p,nan\n", "not finite"),
        ],
    )
    def test_a_file_that_would_make_rankings_order_dependent_is_refused(self, mods, tmp_path, body, match):
        path = tmp_path / "bad.csv"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            mods["s"].read_scores(path)


class TestSiftScores:
    def test_the_scalar_ranks_exactly_as_eval_sift_rank_does(self, mods):
        inliers = {"c": {"a": [12, 40], "b": [12, 55], "c": [30, 1], "d": [0, 9]}}
        pool = {"a", "b", "c", "d", "e", "f"}
        expected = mods["splg"].rank_splg({p: tuple(v) for p, v in inliers["c"].items()}, pool)
        got = mods["ev"].rank_pool(mods["s"].sift_scores(inliers)["c"], pool)
        assert got == expected == ["c", "b", "a", "d", "e", "f"]


class TestProvenancePrior:
    def test_same_source_and_industry_first_then_same_source(self, mods):
        source_of = {"u/pos": "ucsf", "u/tob": "ucsf", "u/food": "ucsf", "t/x": "tobacco800"}
        industry_of = {"u/pos": "Tobacco", "u/tob": "Tobacco", "u/food": "Food"}
        s = mods["s"].provenance_prior_scores({"u/pos"}, source_of, industry_of)
        assert s["u/tob"] > s["u/food"] > s["t/x"]
        assert int(s["u/tob"]) == 2 and int(s["u/food"]) == 1 and int(s["t/x"]) == 0


class TestVersionStamp:
    def _corpus(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"crop")
        (tmp_path / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "classes.json").write_text(
            json.dumps({"c": {"on_roster": True, "query_crop": str(crop)}}), encoding="utf-8"
        )
        return tmp_path, crop

    def test_a_matching_corpus_is_stamped_as_matching_and_a_changed_crop_is_named(self, mods, tmp_path):
        s = mods["s"]
        corpus, crop = self._corpus(tmp_path)
        versions = tmp_path / "versions"
        versions.mkdir()
        (versions / "v9.9.json").write_text(json.dumps({"files": s.fingerprint(corpus)}), encoding="utf-8")
        assert s.stamp(corpus, "v9.9", versions) == {"corpus_version": "v9.9", "matches_frozen": True, "differs": []}
        crop.write_bytes(b"re-cut")
        st = s.stamp(corpus, "v9.9", versions)
        assert st["matches_frozen"] is False and st["differs"] == ["query_crop:c"]

    def test_an_unfrozen_version_never_claims_to_match(self, mods, tmp_path):
        corpus, _ = self._corpus(tmp_path)
        assert mods["s"].stamp(corpus, "v0.1", tmp_path)["matches_frozen"] is False


class TestFrozenManifest:
    def test_the_current_version_has_a_committed_manifest(self, mods):
        # A version bump without `score_ranker.py freeze` would stamp every new
        # result "no frozen manifest", so a version is not released until it is frozen.
        s = mods["s"]
        path = s.VERSIONS / f"{s.cfg.CORPUS_VERSION}.json"
        assert path.exists(), f"run `score_ranker.py freeze` for {s.cfg.CORPUS_VERSION}"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["version"] == s.cfg.CORPUS_VERSION
        assert set(s.VERSION_FILES) <= set(manifest["files"])
        assert any(k.startswith("query_crop:") for k in manifest["files"])
