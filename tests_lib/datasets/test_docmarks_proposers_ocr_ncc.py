"""Non-SIFT completeness proposers: text scoring and NCC geometry (no OCR or GPU here)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def m():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield importlib.import_module("proposers_ocr_ncc")
    finally:
        sys.path.remove(str(_DOCMARKS))


class TestText:
    def test_normalise_and_tokens_ignore_case_punctuation_accents_and_single_letters(self, m):
        assert m.normalise("DY.Secretary") == "dysecretary"
        assert m.tokens("Vicè-Presidènt  E-Mail: mmm@ccc.com a") == ["vice", "president", "mail", "mmm", "ccc", "com"]

    def test_similarity_is_normalised_levenshtein(self, m):
        assert m.similarity("outward", "outward") == 1.0
        assert m.similarity("outward", "0utward") == pytest.approx(6 / 7)
        assert m.similarity("", "x") == 0.0

    def test_words_from_ocr_split_an_item_into_tokens_sharing_its_box(self, m):
        words = m.words_from_ocr([[[[10, 20], [110, 20], [110, 40], [10, 40]], "Under Secretary", 0.9]])
        assert [w.text for w in words] == ["under", "secretary"]
        assert all(w.box == (10, 20, 100, 20) for w in words)

    def test_a_word_cut_in_two_by_the_detector_is_rejoined(self, m):
        items = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], "DY. Secre", 0.9],
            [[[55, 2], [90, 2], [90, 22], [55, 22]], "tary", 0.9],
            [[[0, 500], [40, 500], [40, 520], [0, 520]], "1111222", 0.9],
        ]
        words = m.words_from_ocr(items)
        assert [w.text for w in words] == ["dy", "secre", "tary", "secretary", "1111222"]
        assert words[3].box == (0, 0, 90, 22)  # the far-away third item joins nothing

    def test_tokens_must_cluster_near_one_anchor(self, m):
        W = m.Word
        near = [W("philip", (100, 100, 60, 20)), W("morris", (170, 100, 60, 20))]
        far = [W("philip", (100, 100, 60, 20)), W("morris", (2000, 3000, 60, 20))]
        s_near, box = m.score_page_text(["philip", "morris"], near, radius=200)
        s_far, _ = m.score_page_text(["philip", "morris"], far, radius=200)
        assert s_near == 1.0 and box == (100, 100, 130, 20)
        assert s_far == 0.5  # one token alone scores half

    def test_noisy_ocr_still_matches_and_unrelated_text_does_not(self, m):
        W = m.Word
        assert m.score_page_text(["outward"], [W("0utwarb", (0, 0, 10, 10))], radius=50)[0] == pytest.approx(5 / 7)
        assert m.score_page_text(["outward"], [W("invoice", (0, 0, 10, 10))], radius=50) == (0.0, None)


class TestNccGeometry:
    def test_scales_span_the_members_with_padding(self, m):
        ladder = m.template_scales([0.1, 0.2])
        assert ladder[0] == pytest.approx(0.085) and ladder[-1] == pytest.approx(0.23)
        assert all(a < b for a, b in zip(ladder, ladder[1:]))
        assert len(m.template_scales([0.1, 0.1])) == 4  # the 15% padding alone spans a few rungs
        assert len(m.template_scales([0.01, 1.0])) == 8  # capped

    def test_peak_maps_back_to_page_pixels(self, m):
        # canvas 1024 over a 3072-px page: factor 1/3
        assert m.peak_to_box(100, 50, 30, 20, 1 / 3) == [300, 150, 90, 60]


class TestOutputs:
    def test_recall_at_and_proposals_file_shape(self, m, tmp_path):
        ranked = [["p1", 0.9, [0, 0, 1, 1]], ["x", 0.8, None], ["p2", 0.7, None]]
        assert m.recall_at(ranked, {"p1", "p2"}, 2) == 0.5
        assert m.recall_at(ranked, {"p1", "p2"}, 3) == 1.0
        path = m.write_proposals("ocr_text", {"c": ranked}, out=tmp_path)
        assert json.loads(path.read_text()) == {"method": "ocr_text", "classes": {"c": ranked}}

    def test_class_text_spec_is_well_formed(self, m):
        spec = json.loads(m.CLASS_TEXT.read_text(encoding="utf-8"))
        assert spec["note"]
        for class_id, entry in spec["classes"].items():
            assert "/" in class_id and set(entry) <= {"text", "flag"}
            texts = entry["text"]
            assert texts is None or all(m.tokens(t) for t in ([texts] if isinstance(texts, str) else texts))
