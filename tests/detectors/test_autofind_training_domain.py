"""The training-domain warning in Find and AutoFind runs, and its reference read from disk (#5501).

Tune's warning (``GET /api/find/training-domain``, #4961) used to be the only
place the overlap check ran, and only against a *loaded* training dataset.
Pinned here:

- every Find / AutoFind result carries the check, per detector, so the Find
  Results dialog, the run's notice and the exported results can warn;
- a detector records its training dataset, and the check follows that record
  to the dataset's saved file when it is not loaded, so a run after a restart
  or on a fresh import still has something to compare against;
- the checks of one run read a shared reference once.
"""

from __future__ import annotations

import pickle

import numpy as np

import vtsearch.domain_overlap as overlap_mod
import vtsearch.runtime.autofind as autofind_mod
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    register_context,
    register_detector_context,
    set_thread_dataset_context,
)
from vtscore.utils.hashing import content_md5
from vtsearch import settings

_EMBEDDER = "clap"


def _media(cid: int, dataset: str, *, shifted: bool) -> dict:
    """A synthetic media: in-domain vectors live in the first half of the coordinates, shifted ones in the second."""
    rng = np.random.default_rng(cid + (2000 if shifted else 0))
    emb = np.zeros(64, dtype=np.float32)
    if shifted:
        emb[32:] = rng.standard_normal(32).astype(np.float32)
        emb[32] += 4.0
    else:
        emb[:32] = rng.standard_normal(32).astype(np.float32)
        emb[cid % 2] += 4.0
    emb /= np.linalg.norm(emb)
    name = f"{dataset}_{cid}.wav"
    return {
        "id": cid,
        "media_type": "audio",
        "embedder": _EMBEDDER,
        "md5": content_md5(f"{dataset}/{cid}".encode()),
        "embeddings": {_EMBEDDER: emb},
        "filename": name,
        "origin": {"importer": "test", "params": {"dataset": dataset}},
        "origin_name": name,
    }


def _medias(dataset: str, n: int, *, shifted: bool = False) -> dict[int, dict]:
    return {cid: _media(cid, dataset, shifted=shifted) for cid in range(n)}


def _loaded(name: str, n: int, *, shifted: bool = False) -> tuple[dict, DatasetContext]:
    from vtscore.datasets.registry import add_loaded_id, register_dataset

    entry = register_dataset(
        name=name, media_type="audio", num_items=n, pkl_path=f"/tmp/{name}.pkl", embedder=_EMBEDDER
    )
    ctx = DatasetContext(entry["id"])
    for cid, media in _medias(name, n, shifted=shifted).items():
        ctx.medias[cid] = media
    ctx.dataset_display_name = name
    register_context(ctx)
    add_loaded_id(entry["id"])
    return entry, ctx


def _saved_not_loaded(tmp_path, name: str, n: int) -> tuple[dict, dict[int, dict]]:
    """A dataset registered with a real saved container, as after a restart: nothing is loaded."""
    from vtscore.datasets.container import write_container
    from vtscore.datasets.registry import register_dataset

    medias = _medias(name, n)
    path = tmp_path / f"ds_{name}.pkl"
    write_container(path, pickle.dumps({"medias": medias}, protocol=5), {})
    entry = register_dataset(name=name, media_type="audio", num_items=n, pkl_path=str(path), embedder=_EMBEDDER)
    return entry, medias


def _labels(medias: dict[int, dict], ids) -> LabelSet:
    return LabelSet(
        [
            LabeledElement(
                md5=medias[cid]["md5"],
                label="good" if i % 2 == 0 else "bad",
                origin=medias[cid]["origin"],
                origin_name=medias[cid]["origin_name"],
            )
            for i, cid in enumerate(ids)
        ]
    )


def _detector_on_disk(name: str, labelset: LabelSet, *, training: dict | None = None) -> str:
    """A detector JSON (with the training dataset recorded, when given) registered and on the AutoFind list."""
    from vtscore.detectors.registry import register_detector
    from vtscore.detectors.store import _detector_path, _write_detector
    from vtscore.detectors.training_domain import TRAINING_DATASET_KEY

    data = {"name": name, "text_query": "", "media_type": "audio", "examples": [], "labelset": labelset.to_dict()}
    if training is not None:
        data[TRAINING_DATASET_KEY] = training
    _write_detector(_detector_path(name), data)
    entry = register_detector(name=name, media_type="audio", num_training=len(labelset))
    return entry["id"]


def _stub_scoring(monkeypatch) -> None:
    """Score without training: the check is what is under test, not the head."""

    def fake_score(name, *_args, **_kwargs):
        return name, {"detector_name": name, "threshold": 0.5, "total_hits": 0, "hits": [], "negative_hits": []}

    monkeypatch.setattr(autofind_mod, "score_detector", fake_score)


def _run(target_ctx: DatasetContext, names: list[str]) -> dict:
    settings.set_autofind_detectors(names)
    set_thread_dataset_context(target_ctx)
    snap = dict(target_ctx.medias)
    plan = autofind_mod.plan_autofind(snap)
    return autofind_mod.score_autofind(plan, snap)


class TestFindAndAutoFindCarryTheCheck:
    def test_each_result_says_how_unlike_the_training_dataset_this_is(self, monkeypatch):
        _stub_scoring(monkeypatch)
        ref_entry, ref_ctx = _loaded("photos", 120)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk("mammal", _labels(ref_ctx.medias, range(10)))

        response = _run(target_ctx, ["mammal"])

        check = response["results"]["mammal"]["training_domain"]
        assert check["reference_dataset_id"] == ref_entry["id"]
        assert check["reference_dataset_name"] == "photos"
        assert check["labels_in_reference"] == check["labels_total"] == 10
        assert check["domain_shift"]["shifted"] is True
        assert autofind_mod.domain_shifts(response) == [
            {
                "detector_name": "mammal",
                "reference_dataset_name": "photos",
                "frac_atypical": check["domain_shift"]["frac_atypical"],
            }
        ]

    def test_in_domain_is_reported_and_not_listed_as_a_shift(self, monkeypatch):
        _stub_scoring(monkeypatch)
        _ref_entry, ref_ctx = _loaded("photos", 120)
        _target_entry, target_ctx = _loaded("more-photos", 40)
        _detector_on_disk("mammal", _labels(ref_ctx.medias, range(10)))

        response = _run(target_ctx, ["mammal"])

        assert response["results"]["mammal"]["training_domain"]["domain_shift"]["shifted"] is False
        assert autofind_mod.domain_shifts(response) == []

    def test_on_the_training_dataset_there_is_nothing_to_compare(self, monkeypatch):
        _stub_scoring(monkeypatch)
        _ref_entry, ref_ctx = _loaded("photos", 120)
        _detector_on_disk("mammal", _labels(ref_ctx.medias, range(10)))

        response = _run(ref_ctx, ["mammal"])

        check = response["results"]["mammal"]["training_domain"]
        assert check["domain_shift"] is None
        assert "from this dataset" in check["reason"]

    def test_a_failing_check_does_not_sink_the_run(self, monkeypatch):
        _stub_scoring(monkeypatch)
        _ref_entry, ref_ctx = _loaded("photos", 120)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk("mammal", _labels(ref_ctx.medias, range(10)))

        def boom(*_a, **_k):
            raise RuntimeError("no")

        monkeypatch.setattr(overlap_mod, "training_domain_for", boom)
        response = _run(target_ctx, ["mammal"])

        assert response["detectors_run"] == 1
        assert response["results"]["mammal"]["training_domain"] is None


class TestTheRecordedTrainingDataset:
    def test_an_unloaded_training_dataset_is_read_from_its_saved_file(self, monkeypatch, tmp_path):
        from vtscore.detectors.training_domain import training_dataset_record

        _stub_scoring(monkeypatch)
        ref_entry, ref_medias = _saved_not_loaded(tmp_path, "photos", 120)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk(
            "mammal",
            _labels(ref_medias, range(10)),
            training=training_dataset_record(ref_entry["id"], "photos", 10),
        )

        response = _run(target_ctx, ["mammal"])

        check = response["results"]["mammal"]["training_domain"]
        assert check["reference_dataset_id"] == ref_entry["id"]
        assert check["reference_dataset_name"] == "photos"
        assert check["labels_in_reference"] == 10
        assert check["domain_shift"]["shifted"] is True

    def test_without_the_record_an_unloaded_training_dataset_is_out_of_reach(self, monkeypatch, tmp_path):
        _stub_scoring(monkeypatch)
        _ref_entry, ref_medias = _saved_not_loaded(tmp_path, "photos", 120)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk("mammal", _labels(ref_medias, range(10)))

        check = _run(target_ctx, ["mammal"])["results"]["mammal"]["training_domain"]

        assert check["domain_shift"] is None
        assert "none of the loaded datasets" in check["reason"].lower()

    def test_a_loaded_dataset_holding_more_of_the_labels_beats_the_record(self, monkeypatch, tmp_path):
        from vtscore.detectors.training_domain import training_dataset_record

        _stub_scoring(monkeypatch)
        stale_entry, _stale_medias = _saved_not_loaded(tmp_path, "old", 120)
        live_entry, live_ctx = _loaded("photos", 120)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk(
            "mammal",
            _labels(live_ctx.medias, range(10)),
            training=training_dataset_record(stale_entry["id"], "old", 3),
        )

        check = _run(target_ctx, ["mammal"])["results"]["mammal"]["training_domain"]

        assert check["reference_dataset_id"] == live_entry["id"]

    def test_a_deleted_training_dataset_is_ignored(self, monkeypatch):
        from vtscore.detectors.training_domain import training_dataset_record

        _stub_scoring(monkeypatch)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk(
            "mammal",
            _labels(_medias("gone", 10), range(10)),
            training=training_dataset_record("no-such-dataset", "gone", 10),
        )

        check = _run(target_ctx, ["mammal"])["results"]["mammal"]["training_domain"]

        assert check["domain_shift"] is None
        assert check["reference_dataset_id"] == ""

    def test_detectors_sharing_a_training_dataset_read_it_once(self, monkeypatch, tmp_path):
        from vtscore.detectors.training_domain import training_dataset_record

        _stub_scoring(monkeypatch)
        ref_entry, ref_medias = _saved_not_loaded(tmp_path, "photos", 120)
        _target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        for name in ("mammal", "bird"):
            _detector_on_disk(
                name, _labels(ref_medias, range(10)), training=training_dataset_record(ref_entry["id"], "photos", 10)
            )
        reads: list[str] = []
        real = overlap_mod._stored_vectors

        def counting(dataset_id):
            reads.append(dataset_id)
            return real(dataset_id)

        monkeypatch.setattr(overlap_mod, "_stored_vectors", counting)
        response = _run(target_ctx, ["mammal", "bird"])

        assert reads == [ref_entry["id"]]
        assert all(r["training_domain"]["domain_shift"]["shifted"] for r in response["results"].values())

    def test_tunes_warning_follows_the_record_too(self, client, tmp_path):
        """The route Tune's warning calls reads the record from the active detector's file."""
        from vtscore.detectors.training_domain import training_dataset_record

        ref_entry, ref_medias = _saved_not_loaded(tmp_path, "photos", 120)
        target_entry, target_ctx = _loaded("screens", 40, shifted=True)
        labelset = _labels(ref_medias, range(10))
        detector_id = _detector_on_disk(
            "mammal", labelset, training=training_dataset_record(ref_entry["id"], "photos", 10)
        )
        det = DetectorContext(detector_id, name="mammal")
        det.cached_labelset = labelset
        register_detector_context(det)
        set_thread_dataset_context(target_ctx)

        resp = client.get(
            "/api/find/training-domain", headers={"X-Dataset-Id": target_entry["id"], "X-Detector-Id": detector_id}
        )

        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["reference_dataset_id"] == ref_entry["id"]
        assert data["domain_shift"]["shifted"] is True


class TestTheBackgroundRunNamesTheShift:
    def test_the_task_row_lists_shifted_detectors(self, client, monkeypatch):
        from tests import wait_for_loading_task

        _stub_scoring(monkeypatch)
        _ref_entry, ref_ctx = _loaded("photos", 120)
        target_entry, _target_ctx = _loaded("screens", 40, shifted=True)
        _detector_on_disk("mammal", _labels(ref_ctx.medias, range(10)))
        settings.set_autofind_detectors(["mammal"])

        resp = client.post(f"/api/datasets/registry/{target_entry['id']}/autofind")
        assert resp.status_code == 200, resp.get_json()
        final = wait_for_loading_task(resp.get_json()["task_id"])

        assert final["error"] is None, final
        shifts = final["autofind"]["domain_shifts"]
        assert [s["detector_name"] for s in shifts] == ["mammal"]
        assert shifts[0]["reference_dataset_name"] == "photos"
        assert shifts[0]["frac_atypical"] > 0.5

        run = client.get(f"/api/autofind/runs/{resp.get_json()['task_id']}").get_json()
        assert run["results"]["mammal"]["training_domain"]["domain_shift"]["shifted"] is True
