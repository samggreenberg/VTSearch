"""A text or example sort's display line, redrawn at another balance (#4760).

The ranking of a cosine sort does not depend on the balance, but its display
line does: a typed query's is the count line at beta 1 or below (#4603), the
Goods' centroid's is the count line at every balance (#4732), and a tiled
structural example's rises at the precision end (#4479).  So the sort keeps a
:class:`~vtscore.training.query_sort.SortLine` with its cuts, and the sort cache
redraws the line from it.  These pin that the redraw draws exactly the line the
sort itself would have drawn at that balance, and that it really does move.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.state.core import get_active_context
from vtscore.training.query_sort import SortLine, cosine_sort_cuts, example_sort_cuts_from_paths, text_sort_active
from vtscore.training.structural_similarity import STRUCTURAL_DECISION_THRESHOLD, _example_line
from vtscore.training.thresholds import TextSortCuts, text_sort_cuts

from tests_lib.sorting.test_query_sort import _fill_active_medias
from tests_lib.sorting.test_text_sort_threshold import _shoulder

BALANCES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _fill_shoulder_medias(n_bulk: int = 300, n_match: int = 15, dim: int = 256) -> np.ndarray:
    """An active dataset whose cosines to the returned query are a broad bulk plus a thin shoulder.

    Enough media, and a tight enough bulk, for the count line to count something
    (it needs 50 scores, and matches standing four sigmas out), so the line
    moves with the balance rather than falling back to the guarded line.
    """
    rng = np.random.default_rng(42)
    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0
    medias = get_active_context().medias
    medias.clear()
    for i in range(n_bulk + n_match):
        vec = rng.normal(0.0, 1.0, dim).astype(np.float32)
        if i < n_match:
            vec = vec / np.linalg.norm(vec) * 0.4 + query
        medias[i + 1] = {
            "id": i + 1,
            "embeddings": {"test_emb": (vec / np.linalg.norm(vec)).astype(np.float32)},
            "embedder": "test_emb",
            "media_type": "image",
        }
    return query


class TestSortLine:
    def test_a_text_line_is_the_text_rule_at_each_balance(self):
        scores = _shoulder()[0]
        line = SortLine("text", scores)
        for beta in BALANCES:
            assert line.threshold_at(beta) == round(text_sort_cuts(scores.tolist(), beta=beta).threshold, 4)

    def test_a_text_line_moves_with_the_balance(self):
        line = SortLine("text", _shoulder()[0])
        # The count line keeps beta ** 0.708 times the matches: fewer at the precision end.
        assert line.threshold_at(0.25) > line.threshold_at(1.0)

    def test_a_centroid_line_is_the_centroid_rule_at_each_balance(self):
        from vtscore.detectors.centroid_head import centroid_cut

        scores = _shoulder()[0]
        line = SortLine("centroid", scores)
        for beta in BALANCES:
            assert line.threshold_at(beta) == round(centroid_cut(scores, beta=beta), 4)

    def test_a_centroid_line_moves_at_every_balance(self):
        line = SortLine("centroid", _shoulder()[0])
        assert line.threshold_at(0.25) > line.threshold_at(1.0) > line.threshold_at(4.0)

    @pytest.mark.parametrize("tiled", [False, True])
    def test_a_structural_line_is_the_inlier_gate(self, tiled):
        line = SortLine("structural", tiled=tiled)
        for beta in BALANCES:
            assert line.threshold_at(beta) == _example_line(tiled, beta)

    def test_only_a_tiled_structural_line_rises_at_the_precision_end(self):
        assert SortLine("structural", tiled=False).threshold_at(0.25) == STRUCTURAL_DECISION_THRESHOLD
        assert SortLine("structural", tiled=True).threshold_at(0.25) > STRUCTURAL_DECISION_THRESHOLD
        assert SortLine("structural", tiled=True).threshold_at(1.0) == STRUCTURAL_DECISION_THRESHOLD

    @pytest.mark.parametrize("rule", ["text", "centroid"])
    def test_a_cosine_line_needs_its_scores(self, rule):
        with pytest.raises(ValueError, match="scores"):
            SortLine(rule).threshold_at(1.0)

    def test_cuts_compare_without_their_line(self):
        with_line = TextSortCuts(0.5, 0.25, "count", SortLine("text", np.array([0.1, 0.2])))
        assert with_line == TextSortCuts(0.5, 0.25, "count")


class TestASortRedrawsItsOwnLine:
    """At the balance a sort was drawn at, its line redraws the sort's own number, bit for bit."""

    @pytest.mark.parametrize("beta", BALANCES)
    def test_text_sort(self, beta):
        query = _fill_shoulder_medias()
        _results, cuts = text_sort_active(query, beta=beta)
        assert cuts.line is not None and cuts.line.rule == "text"
        assert cuts.line.threshold_at(beta) == cuts.threshold

    @pytest.mark.parametrize("beta", BALANCES)
    def test_example_sort(self, beta):
        query = _fill_shoulder_medias()
        _results, cuts = cosine_sort_cuts(query, beta=beta)
        assert cuts.line is not None and cuts.line.rule == "centroid"
        assert cuts.line.threshold_at(beta) == cuts.threshold

    def test_a_redraw_is_the_sort_at_the_new_balance(self):
        query = _fill_shoulder_medias()
        _r, at_one = cosine_sort_cuts(query, beta=1.0)
        _r, at_quarter = cosine_sort_cuts(query, beta=0.25)
        assert at_one.line is not None
        assert at_one.line.threshold_at(0.25) == at_quarter.threshold
        # The fixture is not vacuous: the line really moved, and the acquisition cut did not.
        assert at_quarter.threshold != at_one.threshold
        assert at_quarter.acq_threshold == at_one.acq_threshold

    def test_a_text_redraw_is_the_sort_at_the_new_balance(self):
        query = _fill_shoulder_medias()
        _r, at_one = text_sort_active(query, beta=1.0)
        _r, at_quarter = text_sort_active(query, beta=0.25)
        assert at_one.line is not None
        assert at_one.line.threshold_at(0.25) == at_quarter.threshold
        assert at_quarter.threshold != at_one.threshold
        assert at_quarter.acq_threshold == at_one.acq_threshold


class TestAStructuralExampleSortsLine:
    """The verified ranking's line is the inlier gate; a rerank that could not run keeps the centroid's."""

    class _StructuralEmb:
        supports_geometric_verification = True

        def __init__(self, query: np.ndarray):
            self._query = query

        def embed_media(self, media):
            return self._query

        def local_features_forward(self, media):
            return "features"

    @pytest.fixture
    def structural(self, monkeypatch):
        query = _fill_active_medias()
        monkeypatch.setattr(
            "vtscore.training.query_sort.score_embedder_for_active",
            lambda snap: (self._StructuralEmb(query), "sift_vlad"),
        )
        monkeypatch.setattr("vtscore.media.embedder.media_from_path", lambda p: {"path": p.name})

        def use(rerank):
            monkeypatch.setattr("vtscore.training.structural_similarity.maybe_structural_rerank_example", rerank)

        return use

    @pytest.mark.parametrize("tiled", [False, True])
    def test_a_verified_ranking_redraws_the_inlier_gate(self, structural, monkeypatch, tmp_path, tiled):
        structural(lambda results, threshold, *a, **k: (list(results), 0.5))
        monkeypatch.setattr("vtscore.training.structural_stage1.snapshot_has_tiles", lambda snap: tiled)
        _results, cuts = example_sort_cuts_from_paths([tmp_path / "crop.png"])
        assert cuts.line is not None and cuts.line.rule == "structural"
        assert cuts.line.tiled is tiled

    def test_a_rerank_that_could_not_run_keeps_the_centroid_line(self, structural, tmp_path):
        structural(lambda results, threshold, *a, **k: (results, threshold))
        _results, cuts = example_sort_cuts_from_paths([tmp_path / "crop.png"])
        assert cuts.line is not None and cuts.line.rule == "centroid"
