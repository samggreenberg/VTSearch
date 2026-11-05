"""The training-domain check after a handoff: the detector's training dataset is not here at all (#5567).

Until #5567 the overlap check needed the training dataset on this instance,
loaded or registered (#5501).  A detector handed to another user arrives
without it, and the check went blank.  The detector now carries a summary of
it: anchor items named by origin, and calibration scores stamped with the
embedder.  Pinned here, through AutoFind (which reads the detector's file) and
through Tune's route:

- with no trace of the training dataset here, the check reads the summary,
  flags a dataset unlike it and stays quiet on one like it;
- anchors the active dataset holds are read from it, the rest fetched;
- an anchor that cannot be fetched, another embedder, or a summary of the
  active dataset itself refuses with a reason rather than guess;
- the training dataset, when it *is* here, still beats its summary, and the
  summary steps in when its saved file is gone;
- the summary travels in an exported labelset without the pointer (#5571):
  Export → JSON file → New detector › Trained gives the receiver the same
  report the sender had, and a summary drawn in another embedder refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import vtscore.detectors.domain_summary as summary_mod
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


def _media(cid: int, dataset: str, *, shifted: bool, seed_offset: int) -> dict:
    """In-domain vectors live in the first half of the coordinates, shifted ones in the second."""
    rng = np.random.default_rng(cid + seed_offset + (50_000 if shifted else 0))
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
        "origin": {"importer": "test", "params": {"dataset": dataset, "media_type": "audio"}},
        "origin_name": name,
    }


def _medias(dataset: str, n: int, *, shifted: bool = False, seed_offset: int = 0) -> dict[int, dict]:
    return {cid: _media(cid, dataset, shifted=shifted, seed_offset=seed_offset) for cid in range(n)}


def _loaded(name: str, medias: dict[int, dict], *, embedder: str = _EMBEDDER) -> tuple[dict, DatasetContext]:
    from vtscore.datasets.registry import add_loaded_id, register_dataset

    entry = register_dataset(
        name=name, media_type="audio", num_items=len(medias), pkl_path=f"/tmp/{name}.pkl", embedder=embedder
    )
    ctx = DatasetContext(entry["id"])
    ctx.medias.update(medias)
    ctx.dataset_display_name = name
    register_context(ctx)
    add_loaded_id(entry["id"])
    return entry, ctx


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


def _from_another_instance(medias: dict[int, dict], dataset_id: str = "elsewhere-photos") -> tuple[dict, dict]:
    """The pointer and the summary a detector trained on *medias* on another instance carries."""
    from vtscore.detectors.domain_summary import build_domain_summary
    from vtscore.detectors.training_domain import training_dataset_record

    summary = build_domain_summary(
        medias, dataset_id=dataset_id, dataset_name="photos", embedder=_EMBEDDER, media_type="audio"
    )
    assert summary is not None
    return training_dataset_record(dataset_id, "photos", 10), summary


def _detector_on_disk(name: str, labelset: LabelSet, *, training: dict | None, summary: dict | None) -> str:
    from vtscore.detectors.domain_summary import DOMAIN_SUMMARY_KEY
    from vtscore.detectors.registry import register_detector
    from vtscore.detectors.store import _detector_path, _write_detector
    from vtscore.detectors.training_domain import TRAINING_DATASET_KEY

    data = {"name": name, "text_query": "", "media_type": "audio", "examples": [], "labelset": labelset.to_dict()}
    if training is not None:
        data[TRAINING_DATASET_KEY] = training
    if summary is not None:
        data[DOMAIN_SUMMARY_KEY] = summary
    _write_detector(_detector_path(name), data)
    return register_detector(name=name, media_type="audio", num_training=len(labelset))["id"]


def _fetchable(monkeypatch, medias: dict[int, dict]) -> list[str]:
    """Stand in for fetching an anchor from its origin: what the other instance's file embeds to."""
    by_name = {m["origin_name"]: m["embeddings"][_EMBEDDER] for m in medias.values()}
    fetched: list[str] = []

    def fetch(anchor, *, media_type, embedder):
        fetched.append(anchor["origin_name"])
        return by_name.get(anchor["origin_name"])

    monkeypatch.setattr(summary_mod, "_embed_anchor", fetch)
    return fetched


def _stub_scoring(monkeypatch) -> None:
    def fake_score(name, *_args, **_kwargs):
        return name, {"detector_name": name, "threshold": 0.5, "total_hits": 0, "hits": [], "negative_hits": []}

    monkeypatch.setattr(autofind_mod, "score_detector", fake_score)


def _autofind_check(target_ctx: DatasetContext, name: str) -> dict:
    settings.set_autofind_detectors([name])
    set_thread_dataset_context(target_ctx)
    snap = dict(target_ctx.medias)
    return autofind_mod.score_autofind(autofind_mod.plan_autofind(snap), snap)["results"][name]["training_domain"]


class TestTheHandoff:
    def test_a_dataset_unlike_the_training_one_is_flagged(self, monkeypatch):
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        fetched = _fetchable(monkeypatch, photos)
        _detector_on_disk("mammal", _labels(photos, range(10)), training=training, summary=summary)
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))

        check = _autofind_check(target_ctx, "mammal")

        assert check["reason"] == ""
        assert check["reference_dataset_id"] == "elsewhere-photos"
        assert check["reference_dataset_name"] == "photos"
        assert check["labels_total"] == 10 and check["labels_in_reference"] == 10 and check["labels_in_active"] == 0
        report = check["domain_shift"]
        assert report["reference_dataset_id"] == "elsewhere-photos"
        assert report["n_items"] == 80
        assert report["shifted"] is True
        assert report["frac_atypical"] > 0.5
        assert fetched == [a["origin_name"] for a in summary["anchors"]]

    def test_a_dataset_like_the_training_one_is_quiet(self, monkeypatch):
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        _fetchable(monkeypatch, photos)
        _detector_on_disk("mammal", _labels(photos, range(10)), training=training, summary=summary)
        _entry, target_ctx = _loaded("more-photos", _medias("more-photos", 400, seed_offset=10_000))

        report = _autofind_check(target_ctx, "mammal")["domain_shift"]

        assert report["shifted"] is False
        assert report["frac_atypical"] < 0.15

    def test_anchors_the_active_dataset_holds_are_not_fetched(self, monkeypatch):
        """The receiving dataset overlaps the training one: those anchors are read from it, as labels are."""
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        fetched = _fetchable(monkeypatch, photos)
        _detector_on_disk("mammal", _labels(photos, range(10)), training=training, summary=summary)
        # The labels' media stay behind, so the training dataset still holds the most of them.
        _entry, target_ctx = _loaded("photos-copy", {cid: m for cid, m in photos.items() if cid >= 10})

        check = _autofind_check(target_ctx, "mammal")

        assert check["reference_dataset_id"] == "elsewhere-photos"
        assert check["domain_shift"]["shifted"] is False
        held_back = {photos[cid]["origin_name"] for cid in range(10)}
        assert set(fetched) == {a["origin_name"] for a in summary["anchors"]} & held_back

    def test_an_anchor_that_cannot_be_fetched_refuses(self, monkeypatch):
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        lost = summary["anchors"][0]["origin_name"]
        _fetchable(monkeypatch, {cid: m for cid, m in photos.items() if m["origin_name"] != lost})
        _detector_on_disk("mammal", _labels(photos, range(10)), training=training, summary=summary)
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))

        check = _autofind_check(target_ctx, "mammal")

        assert check["domain_shift"] is None
        assert check["reference_dataset_id"] == "elsewhere-photos"
        assert lost in check["reason"] and "could not be fetched" in check["reason"]

    def test_another_embedder_refuses(self, monkeypatch):
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        _fetchable(monkeypatch, photos)
        _detector_on_disk("mammal", _labels(photos, range(10)), training=training, summary=summary)
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True), embedder="beats")

        check = _autofind_check(target_ctx, "mammal")

        assert check["domain_shift"] is None
        assert "Embedder mismatch" in check["reason"]

    def test_without_the_pointer_the_summary_stands_in_for_every_label(self, monkeypatch):
        """A summary that came alone, in an exported labelset (#5571), still names the training dataset."""
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        _training, summary = _from_another_instance(photos)
        _fetchable(monkeypatch, photos)
        _detector_on_disk("mammal", _labels(photos, range(10)), training=None, summary=summary)
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))

        check = _autofind_check(target_ctx, "mammal")

        assert check["reason"] == ""
        assert check["reference_dataset_id"] == "elsewhere-photos"
        assert check["reference_dataset_name"] == "photos"
        assert check["labels_in_reference"] == 10
        assert check["domain_shift"]["shifted"] is True

    def test_a_pointer_to_another_dataset_beats_a_summary_without_one(self, monkeypatch):
        """The pointer says where the labels were drawn; a summary of somewhere else does not overrule it."""
        from vtscore.detectors.training_domain import training_dataset_record

        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        _training, summary = _from_another_instance(photos)
        _fetchable(monkeypatch, photos)
        _detector_on_disk(
            "mammal",
            _labels(photos, range(10)),
            training=training_dataset_record("somewhere-else", "other", 10),
            summary=summary,
        )
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))

        check = _autofind_check(target_ctx, "mammal")

        assert check["domain_shift"] is None
        assert "none of the loaded datasets" in check["reason"].lower()

    def test_tunes_route_reads_the_summary_from_the_detector_file(self, client, monkeypatch):
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        _fetchable(monkeypatch, photos)
        labelset = _labels(photos, range(10))
        detector_id = _detector_on_disk("mammal", labelset, training=training, summary=summary)
        det = DetectorContext(detector_id, name="mammal")
        det.cached_labelset = labelset
        register_detector_context(det)
        target_entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))
        set_thread_dataset_context(target_ctx)

        resp = client.get(
            "/api/find/training-domain", headers={"X-Dataset-Id": target_entry["id"], "X-Detector-Id": detector_id}
        )

        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["reference_dataset_name"] == "photos"
        assert data["domain_shift"]["shifted"] is True


class TestTheDatasetItselfStillComesFirst:
    def test_a_loaded_training_dataset_is_compared_in_full(self, monkeypatch):
        _stub_scoring(monkeypatch)
        photos_entry, photos_ctx = _loaded("photos", _medias("photos", 200))
        from vtscore.detectors.domain_summary import build_domain_summary
        from vtscore.detectors.training_domain import training_dataset_record

        summary = build_domain_summary(
            photos_ctx.medias,
            dataset_id=photos_entry["id"],
            dataset_name="photos",
            embedder=_EMBEDDER,
            media_type="audio",
        )
        fetched = _fetchable(monkeypatch, {})
        _detector_on_disk(
            "mammal",
            _labels(photos_ctx.medias, range(10)),
            training=training_dataset_record(photos_entry["id"], "photos", 10),
            summary=summary,
        )
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))
        reads = []
        monkeypatch.setattr(overlap_mod, "cached_summary_shift", lambda *a, **k: reads.append(1) or {})

        check = _autofind_check(target_ctx, "mammal")

        assert check["domain_shift"]["shifted"] is True
        assert reads == [] and fetched == []

    def test_a_registered_training_dataset_whose_file_is_gone_falls_back_to_its_summary(self, monkeypatch, tmp_path):
        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.training_domain import training_dataset_record

        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        entry = register_dataset(
            name="photos", media_type="audio", num_items=200, pkl_path=str(tmp_path / "gone.pkl"), embedder=_EMBEDDER
        )
        _training, summary = _from_another_instance(photos, dataset_id=entry["id"])
        _fetchable(monkeypatch, photos)
        _detector_on_disk(
            "mammal",
            _labels(photos, range(10)),
            training=training_dataset_record(entry["id"], "photos", 10),
            summary=summary,
        )
        _target_entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))

        check = _autofind_check(target_ctx, "mammal")

        assert check["reference_dataset_id"] == entry["id"]
        assert check["reason"] == ""
        assert check["domain_shift"]["shifted"] is True

    def test_a_summary_of_the_active_dataset_is_nothing_to_compare(self, monkeypatch):
        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        target_entry, target_ctx = _loaded("photos", photos)
        _training, summary = _from_another_instance(photos, dataset_id=target_entry["id"])
        _detector_on_disk("mammal", _labels(photos, range(10)), training=_training, summary=summary)

        check = _autofind_check(target_ctx, "mammal")

        assert check["domain_shift"] is None
        assert "from this dataset" in check["reason"]


class TestTheHandoffByLabelset:
    """#5571: the summary rides in an exported labelset, and the detector imported from it keeps it."""

    @staticmethod
    def _export(client, tmp_path, detector_id: str, labelset: LabelSet) -> Path:
        """What the Dashboard's Export does: the detector's labels, posted to the Server JSON File exporter."""
        out = tmp_path / "shared.labels.json"
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(out)},
                "payload_kind": "labelset",
                "detector_id": detector_id,
                "results": {"labels": labelset.to_dict()["labels"]},
            },
        )
        assert res.status_code == 200, res.get_json()
        return out

    @staticmethod
    def _import_as_new(client, path: Path, name: str) -> dict:
        """What New detector › Trained does with that file, on the receiving instance."""
        from vtscore.detectors.store import _detector_path, _read_detector

        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file", json={"name": name, "filepath": str(path)}
        )
        assert res.status_code == 201, res.get_json()
        return _read_detector(_detector_path(name)) or {}

    def test_the_receiver_gets_the_senders_report(self, client, monkeypatch, tmp_path):
        from vtscore.detectors.domain_summary import DOMAIN_SUMMARY_KEY, stored_domain_summary
        from vtscore.detectors.training_domain import TRAINING_DATASET_KEY

        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        training, summary = _from_another_instance(photos)
        _fetchable(monkeypatch, photos)
        labelset = _labels(photos, range(10))
        source_id = _detector_on_disk("mammal", labelset, training=training, summary=summary)
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))
        on_the_sender = _autofind_check(target_ctx, "mammal")

        shipped_path = self._export(client, tmp_path, source_id, labelset)
        shipped = json.loads(shipped_path.read_text())
        assert shipped["detector_meta"] == {DOMAIN_SUMMARY_KEY: summary}
        assert TRAINING_DATASET_KEY not in shipped and TRAINING_DATASET_KEY not in shipped["detector_meta"]

        received = self._import_as_new(client, shipped_path, "mammal-copy")
        assert stored_domain_summary(received) == summary
        assert TRAINING_DATASET_KEY not in received

        on_the_receiver = _autofind_check(target_ctx, "mammal-copy")
        assert on_the_sender["reason"] == "" and on_the_receiver["reason"] == ""
        assert on_the_receiver["domain_shift"]["shifted"] is True
        assert on_the_receiver["domain_shift"] == on_the_sender["domain_shift"]
        for key in ("reference_dataset_id", "reference_dataset_name", "labels_total", "labels_in_reference"):
            assert on_the_receiver[key] == on_the_sender[key], key

    def test_a_summary_in_another_embedder_says_why_it_cannot_be_read(self, client, monkeypatch, tmp_path):
        from vtscore.detectors.domain_summary import build_domain_summary, stored_domain_summary

        _stub_scoring(monkeypatch)
        photos = _medias("photos", 200)
        summary = build_domain_summary(
            photos, dataset_id="elsewhere-photos", dataset_name="photos", embedder="beats", media_type="audio"
        )
        assert summary is not None
        _fetchable(monkeypatch, photos)
        labelset = _labels(photos, range(10))
        source_id = _detector_on_disk("mammal", labelset, training=None, summary=summary)
        _entry, target_ctx = _loaded("screens", _medias("screens", 80, shifted=True))

        received = self._import_as_new(client, self._export(client, tmp_path, source_id, labelset), "mammal-copy")
        assert stored_domain_summary(received) == summary

        check = _autofind_check(target_ctx, "mammal-copy")
        assert check["domain_shift"] is None
        assert check["reference_dataset_id"] == "elsewhere-photos"
        assert "Embedder mismatch" in check["reason"] and "'beats'" in check["reason"]

    def test_a_detector_with_no_summary_exports_labels_alone(self, client, tmp_path):
        photos = _medias("photos", 200)
        labelset = _labels(photos, range(10))
        source_id = _detector_on_disk("mammal", labelset, training=None, summary=None)

        shipped = json.loads(self._export(client, tmp_path, source_id, labelset).read_text())

        assert "detector_meta" not in shipped

    def test_importing_into_an_existing_detector_keeps_it_only_when_it_has_none(self, client, monkeypatch, tmp_path):
        from vtscore.detectors.domain_summary import stored_domain_summary
        from vtscore.detectors.store import _detector_path, _read_detector

        photos = _medias("photos", 200)
        _training, summary = _from_another_instance(photos)
        labelset = _labels(photos, range(10))
        source_id = _detector_on_disk("mammal", labelset, training=None, summary=summary)
        path = self._export(client, tmp_path, source_id, labelset)
        _detector_on_disk("blank", LabelSet(), training=None, summary=None)
        own = _from_another_instance(_medias("birds", 200), dataset_id="elsewhere-birds")[1]
        _detector_on_disk("birder", LabelSet(), training=None, summary=own)

        for name in ("blank", "birder"):
            res = client.post(f"/api/detectors/{name}/import-labels/server_json_file", json={"filepath": str(path)})
            assert res.status_code == 200, res.get_json()

        assert stored_domain_summary(_read_detector(_detector_path("blank"))) == summary
        assert stored_domain_summary(_read_detector(_detector_path("birder"))) == own

    def test_the_cli_import_keeps_it(self, client, tmp_path):
        from vtscore.cli.labels import import_labels_into_detector
        from vtscore.detectors.domain_summary import stored_domain_summary
        from vtscore.detectors.store import _detector_path, _read_detector

        photos = _medias("photos", 200)
        _training, summary = _from_another_instance(photos)
        labelset = _labels(photos, range(10))
        source_id = _detector_on_disk("mammal", labelset, training=None, summary=summary)
        path = self._export(client, tmp_path, source_id, labelset)

        applied, _skipped = import_labels_into_detector(
            "mammal-cli", "server_json_file", {"filepath": str(path)}, create_media_type="audio"
        )

        assert applied == 10
        assert stored_domain_summary(_read_detector(_detector_path("mammal-cli"))) == summary

    def test_importing_into_the_active_detector_keeps_it(self, client, tmp_path):
        from vtscore.detectors.domain_summary import stored_domain_summary
        from vtscore.detectors.store import _detector_path, _read_detector

        photos = _medias("photos", 200)
        _training, summary = _from_another_instance(photos)
        labelset = _labels(photos, range(10))
        source_id = _detector_on_disk("mammal", labelset, training=None, summary=summary)
        path = self._export(client, tmp_path, source_id, labelset)
        blank_id = _detector_on_disk("blank", LabelSet(), training=None, summary=None)
        register_detector_context(DetectorContext(blank_id, name="blank"))
        here_entry, here_ctx = _loaded("photos-here", photos)
        set_thread_dataset_context(here_ctx)

        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(path)},
            headers={"X-Dataset-Id": here_entry["id"], "X-Detector-Id": blank_id},
        )

        assert res.status_code == 200, res.get_json()
        assert res.get_json()["applied"] == 10
        assert stored_domain_summary(_read_detector(_detector_path("blank"))) == summary
