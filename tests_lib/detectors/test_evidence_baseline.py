"""#5089: the Evidence verdict's training baseline, stamped onto the detector and refused on any mismatch.

The baseline is two distances in the embedder's geometry, recorded where the
detector is tested on its training dataset and read back wherever it travels.
Pinned here: what the stamp covers (any vote, the embedder, its width, ``k`` and
``alpha``), that a reader refuses a stale stamp, and that recording one is a
read-modify-write that keeps every other key of the detector file.  And the
draw a learned sort leaves (#5120): that it is the evidence route's draw over
the same inputs, that it refuses once the detector has moved on, and that it
lands once the sorts go quiet, or at once when a reader flushes it.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.evidence_baseline import (
    EVIDENCE_BASELINE_KEY,
    PendingDraw,
    baseline_stamp,
    defer_baseline_draw,
    discard_baseline_draw,
    draw_baseline,
    flush_baseline_draw,
    labelset_evidence,
    record_baseline,
    scored_queries,
    stored_baseline,
    wait_for_baseline_draws,
)
from vtscore.detectors.evidence_coverage import reference_cutoffs
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.detectors.store import _read_detector, _write_detector
from vtscore.state.core import DatasetContext, DetectorContext


def _labels(*marks: str) -> LabelSet:
    return LabelSet([LabeledElement(md5=f"m{i}", label=label) for i, label in enumerate(marks)])


def _stamp(labels: LabelSet, embedder: str = "siglip", dim: int = 768) -> dict:
    stamp = baseline_stamp(labels, embedder, dim)
    assert stamp is not None
    return stamp


def _baseline(stamp: dict) -> dict:
    return {**stamp, "dataset_id": "ds", "n_pos": 30, "n_neg": 500, "cut_pos": 0.41, "cut_neg": 0.52}


class TestTheStamp:
    def test_the_same_labels_in_the_same_embedder_stamp_alike(self):
        assert baseline_stamp(_labels("good", "bad"), "siglip", 768) == baseline_stamp(
            _labels("good", "bad"), "siglip", 768
        )

    def test_a_vote_another_embedder_or_width_restamps(self):
        stamp = baseline_stamp(_labels("good", "bad"), "siglip", 768)
        assert baseline_stamp(_labels("good", "bad", "bad"), "siglip", 768) != stamp
        assert baseline_stamp(_labels("good", "good"), "siglip", 768) != stamp
        assert baseline_stamp(_labels("good", "bad"), "clip", 768) != stamp
        assert baseline_stamp(_labels("good", "bad"), "siglip", 1152) != stamp
        assert baseline_stamp(_labels("good", "bad"), "siglip", 768, alpha=0.01) != stamp

    def test_no_embedder_name_no_stamp(self):
        assert baseline_stamp(_labels("good", "bad"), "", 768) is None


class TestReadingItBack:
    def test_a_matching_stamp_reads_the_baseline(self):
        stamp = _stamp(_labels("good", "bad"))
        assert stored_baseline({EVIDENCE_BASELINE_KEY: _baseline(stamp)}, stamp) == _baseline(stamp)

    def test_a_stale_stamp_reads_nothing(self):
        old = _stamp(_labels("good", "bad"))
        new = _stamp(_labels("good", "bad", "good"))
        assert stored_baseline({EVIDENCE_BASELINE_KEY: _baseline(old)}, new) is None

    def test_no_baseline_or_no_stamp_reads_nothing(self):
        stamp = _stamp(_labels("good", "bad"))
        assert stored_baseline({}, stamp) is None
        assert stored_baseline(None, stamp) is None
        assert stored_baseline({EVIDENCE_BASELINE_KEY: "junk"}, stamp) is None
        assert stored_baseline({EVIDENCE_BASELINE_KEY: _baseline(stamp)}, None) is None


class TestRecordingIt:
    def test_it_keeps_every_other_key_and_writes_once(self, tmp_path):
        path = tmp_path / "birds.json"
        _write_detector(path, {"name": "birds", "labelset": {"labels": []}, "text_query": "a bird"})
        baseline = _baseline(_stamp(_labels("good", "bad")))
        assert record_baseline(path, baseline) is True
        data = _read_detector(path)
        assert data == {
            "name": "birds",
            "labelset": {"labels": []},
            "text_query": "a bird",
            EVIDENCE_BASELINE_KEY: baseline,
        }
        assert record_baseline(path, baseline) is False

    def test_a_new_baseline_replaces_the_old(self, tmp_path):
        path = tmp_path / "birds.json"
        _write_detector(path, {"name": "birds"})
        record_baseline(path, _baseline(_stamp(_labels("good", "bad"))))
        newer = _baseline(_stamp(_labels("good", "bad", "bad")))
        assert record_baseline(path, newer) is True
        data = _read_detector(path)
        assert data is not None and data[EVIDENCE_BASELINE_KEY] == newer

    def test_no_detector_file_writes_nothing(self, tmp_path):
        assert record_baseline(tmp_path / "gone.json", {"cut_pos": 0.1}) is False
        assert not (tmp_path / "gone.json").exists()


_EMBEDDER = "emb"
_DIM = 8


def _media(cid: int, vector: np.ndarray) -> dict:
    name = f"m{cid}.wav"
    return {
        "id": cid,
        "media_type": "audio",
        "embedder": _EMBEDDER,
        "md5": f"md5-{cid}",
        "embeddings": {_EMBEDDER: vector},
        "filename": name,
        "origin": {"importer": "test", "params": {}},
        "origin_name": name,
    }


def _element(media: dict, label: str) -> LabeledElement:
    return LabeledElement(md5=media["md5"], label=label, origin=media["origin"], origin_name=media["origin_name"])


def _world(tmp_path, *, n: int = 60) -> PendingDraw:
    """A training dataset of *n* media, a detector with two Goods and two Bads on it, and the draw its sort leaves."""
    rng = np.random.default_rng(42)
    dataset = DatasetContext("ds-a")
    for cid in range(n):
        vector = rng.standard_normal(_DIM).astype(np.float32)
        dataset.medias[cid] = _media(cid, vector / np.linalg.norm(vector))
    marks = {0: "good", 1: "good", 2: "bad", 3: "bad"}
    labelset = LabelSet([_element(dataset.medias[cid], label) for cid, label in marks.items()])
    det = DetectorContext("det-a")
    det.cached_labelset = labelset
    det.embedder = _EMBEDDER
    for cid, elem in zip(marks, labelset.elements, strict=True):
        det.label_embeddings[stable_element_id(elem)] = dataset.medias[cid]["embeddings"][_EMBEDDER]
    path = tmp_path / "birds.json"
    _write_detector(path, {"name": "birds", "labelset": labelset.to_dict()})
    stamp = baseline_stamp(labelset, _EMBEDDER, _DIM)
    assert stamp is not None
    return PendingDraw(
        detector_id="det-a",
        path=path,
        stamp=stamp,
        dataset_id="ds-a",
        dataset_name="A",
        det_ctx=det,
        dataset=dataset,
        labelset=labelset,
        medias=dict(dataset.medias),
        results=[{"id": cid, "score": float(rng.random())} for cid in dataset.medias],
        threshold=0.5,
        embedder=_EMBEDDER,
    )


def _recorded(pending: PendingDraw) -> dict | None:
    return (_read_detector(pending.path) or {}).get(EVIDENCE_BASELINE_KEY)


class TestDrawingIt:
    def test_the_scored_media_are_read_in_the_labels_space_and_called_at_the_line(self):
        medias = {1: _media(1, np.ones(_DIM, dtype=np.float32)), 2: {"id": 2, "embeddings": {}}}
        ids, matrix, predicted = scored_queries([(1, 0.7), (2, 0.9), (3, 0.9)], medias, _EMBEDDER, 0.5)
        assert ids == [1]
        assert matrix is not None and matrix.shape == (1, _DIM)
        assert predicted.tolist() == [True]
        assert scored_queries([(2, 0.9)], medias, _EMBEDDER, 0.5)[1] is None

    def test_the_labelled_media_are_left_out_of_the_reference(self, tmp_path):
        pending = _world(tmp_path)
        evidence = labelset_evidence(pending.det_ctx, pending.labelset)
        assert evidence is not None
        pos, neg = evidence
        assert (pos.shape[0], neg.shape[0]) == (2, 2)
        ids, matrix, predicted = scored_queries(
            ((r["id"], r["score"]) for r in pending.results), pending.medias, _EMBEDDER, pending.threshold
        )
        assert matrix is not None
        baseline = draw_baseline(
            pending.stamp,
            dataset_id="ds-a",
            dataset_name="A",
            pos=pos,
            neg=neg,
            query_ids=ids,
            query_matrix=matrix,
            predicted_positive=predicted,
            labelled={0, 1, 2, 3},
        )
        expected = reference_cutoffs(pos, neg, matrix[4:], predicted[4:])
        assert baseline == {**pending.stamp, "dataset_id": "ds-a", "dataset_name": "A", **expected}
        assert baseline["n_pos"] + baseline["n_neg"] == len(ids) - 4


class TestAPendingDraw:
    def test_it_draws_what_the_route_draws_over_the_same_inputs(self, tmp_path):
        pending = _world(tmp_path)
        baseline = pending.draw()
        assert baseline is not None
        assert baseline["dataset_id"] == "ds-a" and baseline["dataset_name"] == "A"
        assert stored_baseline({EVIDENCE_BASELINE_KEY: baseline}, pending.stamp) == baseline
        assert baseline["n_pos"] + baseline["n_neg"] == len(pending.medias) - 4

    def test_a_vote_since_draws_nothing(self, tmp_path):
        pending = _world(tmp_path)
        pending.det_ctx.cached_labelset = LabelSet([*pending.labelset.elements, _element(pending.medias[4], "good")])
        assert pending.draw() is None

    def test_labels_read_in_another_space_since_draw_nothing(self, tmp_path):
        pending = _world(tmp_path)
        pending.det_ctx.embedder = "other"
        assert pending.draw() is None


class TestDeferringIt:
    def test_a_flush_lands_it_once(self, tmp_path):
        pending = _world(tmp_path)
        defer_baseline_draw(pending, quiet=3600.0)
        assert _recorded(pending) is None
        assert flush_baseline_draw("det-a") is True
        assert _recorded(pending) == pending.draw()
        assert flush_baseline_draw("det-a") is False
        assert wait_for_baseline_draws(timeout=0)

    def test_a_newer_sort_replaces_the_waiting_draw(self, tmp_path):
        older = _world(tmp_path)
        newer = dataclasses.replace(older, dataset_name="A, renamed")
        defer_baseline_draw(older, quiet=3600.0)
        defer_baseline_draw(newer, quiet=3600.0)
        assert flush_baseline_draw("det-a") is True
        recorded = _recorded(newer)
        assert recorded is not None and recorded["dataset_name"] == "A, renamed"

    def test_a_discarded_draw_never_lands(self, tmp_path):
        pending = _world(tmp_path)
        defer_baseline_draw(pending, quiet=3600.0)
        discard_baseline_draw("det-a")
        assert wait_for_baseline_draws(timeout=0)
        assert flush_baseline_draw("det-a") is False
        assert _recorded(pending) is None

    def test_once_quiet_it_lands_on_its_own(self, tmp_path):
        pending = _world(tmp_path)
        defer_baseline_draw(pending, quiet=0.0)
        assert wait_for_baseline_draws(timeout=30.0)
        assert _recorded(pending) == pending.draw()

    def test_recording_repoints_the_detectors_cached_labelset(self, tmp_path):
        """So the write does not read as an outside edit and rehydrate the session in memory."""
        pending = _world(tmp_path)
        before = pending.det_ctx.cached_labelset
        defer_baseline_draw(pending, quiet=3600.0)
        assert flush_baseline_draw("det-a") is True
        assert pending.det_ctx.cached_labelset is not before
        assert pending.det_ctx.cached_labelset_mtime == pending.path.stat().st_mtime

    def test_a_failing_draw_is_logged_not_raised(self, tmp_path, monkeypatch, caplog):
        pending = _world(tmp_path)

        def _boom(_self):
            raise RuntimeError("no vectors")

        monkeypatch.setattr(PendingDraw, "draw", _boom)
        defer_baseline_draw(pending, quiet=3600.0)
        with caplog.at_level(logging.ERROR, logger="vtscore.detectors.evidence_baseline"):
            assert flush_baseline_draw("det-a") is False
        assert "det-a" in caplog.text
        assert wait_for_baseline_draws(timeout=0)

    def test_nothing_waits_for_a_detector_with_no_draw(self):
        assert flush_baseline_draw("det-none") is False
        discard_baseline_draw("det-none")
        assert wait_for_baseline_draws(timeout=0)
