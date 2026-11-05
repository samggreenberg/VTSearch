"""#5567: the training-domain summary a detector records, and how a reader resolves it.

The overlap check needs the training dataset's vectors, which do not travel
with a detector.  The summary does: anchor items named by origin and MD5, and
calibration scores stamped with the embedder.  Pinned here: what the record
holds (origins and scalars, never a vector), that a reader refuses a malformed
one, that the build is reused per dataset, that anchors are read from the
active dataset before any fetch, that a missing anchor refuses rather than
skews the check, and that the record rides with the training-dataset pointer.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import pytest

import vtscore.coverage.domain_summary as summary_algo
import vtscore.detectors.domain_summary as summary_mod
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.domain_summary import (
    DOMAIN_SUMMARY_KEY,
    SummaryUnavailable,
    anchor_vectors,
    build_domain_summary,
    domain_summary_for,
    stored_domain_summary,
)
from vtscore.detectors.evidence_baseline import (
    PendingDraw,
    baseline_stamp,
    defer_baseline_draw,
    flush_baseline_draw,
    record_training_dataset,
)
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.detectors.store import _read_detector, _write_detector
from vtscore.detectors.training_domain import TRAINING_DATASET_KEY, training_dataset_record
from vtscore.state.core import DatasetContext, DetectorContext

_EMBEDDER = "emb"
_DIM = 16


def _media(cid: int, vector: np.ndarray, *, dataset: str = "a") -> dict:
    name = f"{dataset}_{cid}.wav"
    return {
        "id": cid,
        "media_type": "audio",
        "embedder": _EMBEDDER,
        "md5": f"md5-{dataset}-{cid}",
        "embeddings": {_EMBEDDER: vector},
        "filename": name,
        "origin": {"importer": "test", "params": {"dataset": dataset}},
        "origin_name": name,
    }


def _medias(n: int = 200, *, dataset: str = "a", seed: int = 42) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    out = {}
    for cid in range(n):
        vector = rng.standard_normal(_DIM).astype(np.float32)
        out[cid] = _media(cid, vector / np.linalg.norm(vector), dataset=dataset)
    return out


def _summary(medias: dict[int, dict], dataset_id: str = "ds-a") -> dict:
    summary = build_domain_summary(
        medias, dataset_id=dataset_id, dataset_name="A", embedder=_EMBEDDER, media_type="audio"
    )
    assert summary is not None
    return summary


def _registered(medias: dict[int, dict], name: str = "A") -> dict:
    from vtscore.datasets.registry import register_dataset

    return register_dataset(
        name=name, media_type="audio", num_items=len(medias), pkl_path=f"/tmp/{name}.pkl", embedder=_EMBEDDER
    )


class TestTheRecord:
    def test_it_names_its_anchors_by_origin_and_holds_no_vector(self):
        medias = _medias()
        summary = _summary(medias)
        assert summary["format"] == 1
        assert summary["dataset_id"] == "ds-a" and summary["dataset_name"] == "A"
        assert summary["embedder"] == _EMBEDDER and summary["dim"] == _DIM and summary["media_type"] == "audio"
        assert summary["n_reference"] == 200
        # Each anchor is a media of the dataset, by its identity fields only.
        by_md5 = {m["md5"]: m for m in medias.values()}
        for anchor in summary["anchors"]:
            assert set(anchor) == {"md5", "origin", "origin_name", "filename"}
            media = by_md5[anchor["md5"]]
            assert anchor["origin"] == media["origin"] and anchor["origin_name"] == media["origin_name"]
        assert len(summary["calibration"]) == 100
        assert all(isinstance(x, float) for x in summary["calibration"])
        # It is plain JSON: nothing in it is an array.
        assert json.loads(json.dumps(summary)) == summary

    def test_no_embedder_name_no_vectors_or_too_few_no_summary(self):
        medias = _medias()
        assert build_domain_summary(medias, dataset_id="d", dataset_name="", embedder="", media_type="audio") is None
        assert build_domain_summary({}, dataset_id="d", dataset_name="", embedder=_EMBEDDER, media_type="audio") is None
        few = _medias(39)
        assert build_domain_summary(few, dataset_id="d", dataset_name="", embedder=_EMBEDDER, media_type="") is None

    def test_media_without_a_vector_are_left_out(self):
        medias = _medias()
        for cid in range(50):
            medias[cid]["embeddings"] = {}
        assert _summary(medias)["n_reference"] == 150

    def test_it_reads_back_as_written(self):
        summary = _summary(_medias())
        assert stored_domain_summary({DOMAIN_SUMMARY_KEY: json.loads(json.dumps(summary))}) == summary

    @pytest.mark.parametrize(
        "change",
        [
            {"format": 2},
            {"anchors": []},
            {"anchors": ["not-a-dict"]},
            {"calibration": [0.5] * 19},
            {"calibration": ["x"] * 40},
            {"calibration": [True] * 40},
            {"dataset_id": ""},
            {"embedder": ""},
            {"dim": 0},
            {"dim": "wide"},
        ],
    )
    def test_a_malformed_record_reads_as_none(self, change):
        summary = _summary(_medias())
        assert stored_domain_summary({DOMAIN_SUMMARY_KEY: {**summary, **change}}) is None

    def test_no_record_reads_as_none(self):
        assert stored_domain_summary(None) is None
        assert stored_domain_summary({}) is None
        assert stored_domain_summary({DOMAIN_SUMMARY_KEY: "x"}) is None


class TestBuildingItForADataset:
    def test_it_reads_the_registry_and_reuses_the_build(self, monkeypatch):
        medias = _medias()
        entry = _registered(medias)
        calls = []
        real = summary_algo.summarize_reference

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(summary_algo, "summarize_reference", counting)
        first = domain_summary_for(entry["id"], medias)
        assert first is not None
        assert first["dataset_id"] == entry["id"] and first["dataset_name"] == "A"
        assert first["embedder"] == _EMBEDDER and first["media_type"] == "audio"
        assert domain_summary_for(entry["id"], dict(medias)) == first
        assert len(calls) == 1
        # New media: built again.
        medias[500] = _media(500, medias[0]["embeddings"][_EMBEDDER])
        assert domain_summary_for(entry["id"], medias)["n_reference"] == 201
        assert len(calls) == 2

    def test_a_rename_relabels_the_reused_build(self):
        from vtscore.datasets.registry import rename_dataset

        medias = _medias()
        entry = _registered(medias)
        domain_summary_for(entry["id"], medias)
        rename_dataset(entry["id"], "B")
        assert domain_summary_for(entry["id"], medias)["dataset_name"] == "B"

    def test_an_unregistered_dataset_has_no_embedder_to_stamp(self):
        assert domain_summary_for("no-such-dataset", _medias()) is None

    def test_a_failing_build_is_logged_not_raised(self, monkeypatch, caplog):
        medias = _medias()
        entry = _registered(medias)

        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(summary_algo, "summarize_reference", boom)
        with caplog.at_level(logging.ERROR):
            assert domain_summary_for(entry["id"], medias) is None
        assert "Could not summarise" in caplog.text


class TestResolvingTheAnchors:
    @staticmethod
    def _fetching(monkeypatch, source: dict[int, dict]) -> list[str]:
        """Stand in for fetching an anchor's file: the vector the other instance holds, by origin name."""
        by_name = {m["origin_name"]: m["embeddings"][_EMBEDDER] for m in source.values()}
        fetched: list[str] = []

        def fetch(anchor, *, media_type, embedder):
            assert embedder == _EMBEDDER and media_type == "audio"
            fetched.append(anchor["origin_name"])
            return by_name.get(anchor["origin_name"])

        monkeypatch.setattr(summary_mod, "_embed_anchor", fetch)
        return fetched

    def test_anchors_the_active_dataset_holds_are_read_from_it(self, monkeypatch):
        source = _medias()
        summary = _summary(source)
        fetched = self._fetching(monkeypatch, {})
        ctx = DatasetContext("ds-b")
        ctx.medias.update(source)
        vectors = anchor_vectors(summary, ctx, embedder=_EMBEDDER)
        assert vectors.shape == (len(summary["anchors"]), _DIM)
        assert fetched == []

    def test_the_rest_are_fetched_in_the_summarys_order(self, monkeypatch):
        source = _medias()
        summary = _summary(source)
        fetched = self._fetching(monkeypatch, source)
        vectors = anchor_vectors(summary, DatasetContext("ds-b"), embedder=_EMBEDDER)
        assert fetched == [a["origin_name"] for a in summary["anchors"]]
        by_name = {m["origin_name"]: m["embeddings"][_EMBEDDER] for m in source.values()}
        expected = np.stack([by_name[a["origin_name"]] for a in summary["anchors"]])
        assert np.array_equal(vectors, expected)
        # Fetched once per process.
        anchor_vectors(summary, DatasetContext("ds-c"), embedder=_EMBEDDER)
        assert len(fetched) == len(summary["anchors"])

    def test_one_that_cannot_be_fetched_refuses(self, monkeypatch):
        source = _medias()
        summary = _summary(source)
        missing = summary["anchors"][3]["origin_name"]
        fetched = self._fetching(monkeypatch, {k: m for k, m in source.items() if m["origin_name"] != missing})
        with pytest.raises(SummaryUnavailable, match=missing):
            anchor_vectors(summary, DatasetContext("ds-b"), embedder=_EMBEDDER)
        assert len(fetched) == 4
        # A failed fetch is not retried at once; the ones that worked are kept.
        with pytest.raises(SummaryUnavailable):
            anchor_vectors(summary, DatasetContext("ds-b"), embedder=_EMBEDDER)
        assert len(fetched) == 4
        monkeypatch.setattr(summary_mod, "_RETRY_FAILED_SECONDS", 0.0)
        with pytest.raises(SummaryUnavailable):
            anchor_vectors(summary, DatasetContext("ds-b"), embedder=_EMBEDDER)
        assert fetched[4:] == [missing]

    def test_an_anchor_without_an_origin_is_never_fetched(self):
        summary = _summary(_medias())
        summary["anchors"][0].pop("origin")
        with pytest.raises(SummaryUnavailable):
            anchor_vectors(summary, DatasetContext("ds-b"), embedder=_EMBEDDER)


def _world(tmp_path, medias: dict[int, dict], dataset_id: str) -> PendingDraw:
    """A detector with four labels on *medias*, and the draw its learned sort leaves there."""
    marks = {0: "good", 1: "good", 2: "bad", 3: "bad"}
    dataset = DatasetContext(dataset_id)
    dataset.medias.update(medias)
    labelset = LabelSet(
        [
            LabeledElement(
                md5=medias[c]["md5"], label=lab, origin=medias[c]["origin"], origin_name=medias[c]["origin_name"]
            )
            for c, lab in marks.items()
        ]
    )
    det = DetectorContext("det-a")
    det.cached_labelset = labelset
    det.embedder = _EMBEDDER
    for cid, elem in zip(marks, labelset.elements, strict=True):
        det.label_embeddings[stable_element_id(elem)] = medias[cid]["embeddings"][_EMBEDDER]
    path = tmp_path / "birds.json"
    _write_detector(path, {"name": "birds", "labelset": labelset.to_dict()})
    stamp = baseline_stamp(labelset, _EMBEDDER, _DIM)
    assert stamp is not None
    rng = np.random.default_rng(7)
    return PendingDraw(
        detector_id="det-a",
        path=path,
        stamp=stamp,
        dataset_id=dataset_id,
        dataset_name="A",
        det_ctx=det,
        dataset=dataset,
        labelset=labelset,
        medias=dict(medias),
        results=[{"id": cid, "score": float(rng.random())} for cid in medias],
        threshold=0.5,
        embedder=_EMBEDDER,
        labels_held=4,
    )


class TestItRidesWithThePointer:
    def test_a_landed_draw_records_the_summary(self, tmp_path):
        medias = _medias()
        entry = _registered(medias)
        pending = _world(tmp_path, medias, entry["id"])
        defer_baseline_draw(pending, quiet=3600.0)
        assert flush_baseline_draw("det-a") is True
        data = _read_detector(pending.path) or {}
        assert data[TRAINING_DATASET_KEY]["dataset_id"] == entry["id"]
        assert stored_domain_summary(data) == domain_summary_for(entry["id"], medias)

    def test_a_pointer_only_draw_records_it_too_and_writes_once(self, tmp_path):
        medias = _medias()
        entry = _registered(medias)
        pending = dataclasses.replace(_world(tmp_path, medias, entry["id"]), draws_baseline=False)
        defer_baseline_draw(pending, quiet=3600.0)
        assert flush_baseline_draw("det-a") is True
        assert stored_domain_summary(_read_detector(pending.path)) is not None
        defer_baseline_draw(pending, quiet=3600.0)
        assert flush_baseline_draw("det-a") is False

    def test_a_summary_of_another_dataset_is_not_recorded(self, tmp_path):
        path = tmp_path / "birds.json"
        _write_detector(path, {"name": "birds"})
        other = _summary(_medias(), dataset_id="ds-other")
        assert record_training_dataset(path, training_dataset_record("ds-a", "A", 3), domain_summary=other) is True
        data = _read_detector(path) or {}
        assert TRAINING_DATASET_KEY in data and DOMAIN_SUMMARY_KEY not in data
