"""#5571: a detector's training-domain summary travels in an exported labelset.

A detector shared as a labelset, rather than as its file, used to arrive
without the summary its overlap check reads (#5567).  The summary now rides in
the labelset's ``detector_meta``, as the sending instance recorded it, and the
receiving detector keeps it.  Pinned here, at the library tier:

- the export block carries it, and the #5131 strip lets it through;
- the import keeps it only when the detector has no training dataset of its
  own, or its own is the same dataset, and never merges two;
- the sync push sends it and the pull keeps it;
- label importers hand the block over (``run_labelset``), the JSON importer
  and exporter included, and an importer that returns labels alone still works.
"""

from __future__ import annotations

import json

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet, labelset_for_export
from vtscore.detectors.domain_summary import (
    DOMAIN_SUMMARY_KEY,
    adopt_domain_summary,
    adopt_domain_summary_on_file,
    build_domain_summary,
    stored_domain_summary,
)
from vtscore.detectors.input_spec import apply_detector_meta, build_detector_meta
from vtscore.detectors.store import _read_detector, _write_detector
from vtscore.detectors.training_domain import TRAINING_DATASET_KEY, training_dataset_record
from vtscore.labels.importers.base import LabelImporter, PluginField

_EMBEDDER = "emb"


def _medias(n: int = 200, *, dataset: str = "a") -> dict[int, dict]:
    rng = np.random.default_rng(42)
    out = {}
    for cid in range(n):
        vector = rng.standard_normal(16).astype(np.float32)
        name = f"{dataset}_{cid}.wav"
        out[cid] = {
            "id": cid,
            "media_type": "audio",
            "md5": f"md5-{dataset}-{cid}",
            "embeddings": {_EMBEDDER: vector / np.linalg.norm(vector)},
            "filename": name,
            "origin": {"importer": "test", "params": {"dataset": dataset}},
            "origin_name": name,
        }
    return out


def _summary(dataset_id: str = "ds-a", *, dataset: str = "a") -> dict:
    summary = build_domain_summary(
        _medias(dataset=dataset), dataset_id=dataset_id, dataset_name="A", embedder=_EMBEDDER, media_type="audio"
    )
    assert summary is not None
    return summary


def _labels() -> list[dict]:
    origin = {"importer": "test", "params": {"dataset": "a"}}
    return [
        LabeledElement(md5="md5-a-0", label="good", origin=origin, origin_name="a_0.wav").to_dict(),
        LabeledElement(md5="md5-a-1", label="bad", origin=origin, origin_name="a_1.wav").to_dict(),
    ]


class TestTheExportCarriesIt:
    def test_the_detector_meta_block_holds_the_summary(self):
        summary = _summary()
        meta = build_detector_meta({"media_type": "audio", DOMAIN_SUMMARY_KEY: summary})
        assert meta == {"media_type": "audio", DOMAIN_SUMMARY_KEY: summary}

    def test_a_detector_without_one_or_with_a_malformed_one_sends_none(self):
        assert build_detector_meta({"media_type": "audio"}) == {"media_type": "audio"}
        broken = {**_summary(), "format": 99}
        assert build_detector_meta({"media_type": "audio", DOMAIN_SUMMARY_KEY: broken}) == {"media_type": "audio"}

    def test_the_pointer_stays_behind(self):
        data = {
            "media_type": "audio",
            TRAINING_DATASET_KEY: training_dataset_record("ds-a", "A", 2),
            DOMAIN_SUMMARY_KEY: _summary(),
        }
        assert TRAINING_DATASET_KEY not in build_detector_meta(data)

    def test_the_5131_strip_lets_it_through_and_still_drops_the_threshold(self):
        summary = _summary()
        payload = {"labels": _labels(), "detector_meta": {"threshold": 0.4, DOMAIN_SUMMARY_KEY: summary}}
        out = labelset_for_export(payload)
        assert out["detector_meta"] == {DOMAIN_SUMMARY_KEY: summary}
        assert json.loads(json.dumps(out)) == out

    def test_a_labelset_round_trips_it_and_a_merge_drops_it(self):
        summary = _summary()
        labelset = LabelSet.from_dict({"labels": _labels(), "detector_meta": {DOMAIN_SUMMARY_KEY: summary}})
        assert LabelSet.from_dict(labelset.to_dict()).detector_meta == {DOMAIN_SUMMARY_KEY: summary}
        # A merge is a new detector: one sender's training domain is not its own.
        assert labelset.merge(LabelSet()).detector_meta is None


class TestTheImportKeepsIt:
    def test_a_detector_with_no_training_dataset_of_its_own_takes_it(self):
        summary = _summary()
        data: dict = {"name": "copy"}
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: summary}) is True
        assert stored_domain_summary(data) == summary
        assert TRAINING_DATASET_KEY not in data

    def test_the_same_summary_again_changes_nothing(self):
        summary = _summary()
        data = {DOMAIN_SUMMARY_KEY: summary}
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: json.loads(json.dumps(summary))}) is False

    def test_a_later_draw_of_the_same_dataset_replaces_it(self):
        older = _summary()
        newer = {**older, "n_reference": older["n_reference"] + 1}
        data = {DOMAIN_SUMMARY_KEY: older}
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: newer}) is True
        assert stored_domain_summary(data) == newer

    def test_a_summary_of_another_dataset_never_overwrites_the_detectors_own(self):
        own = _summary("ds-own", dataset="own")
        data = {DOMAIN_SUMMARY_KEY: own}
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: _summary("ds-a")}) is False
        assert stored_domain_summary(data) == own

    def test_a_pointer_to_another_dataset_counts_as_its_own(self):
        """Trained here on another dataset, with no summary of it: the import does not overrule that."""
        data = {TRAINING_DATASET_KEY: training_dataset_record("ds-own", "Own", 5)}
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: _summary("ds-a")}) is False
        assert DOMAIN_SUMMARY_KEY not in data

    def test_a_pointer_to_the_same_dataset_takes_its_summary(self):
        data = {TRAINING_DATASET_KEY: training_dataset_record("ds-a", "A", 5)}
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: _summary("ds-a")}) is True

    def test_nothing_missing_or_malformed_is_stored(self):
        data: dict = {}
        assert adopt_domain_summary(data, None) is False
        assert adopt_domain_summary(data, {}) is False
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: "x"}) is False
        assert adopt_domain_summary(data, {DOMAIN_SUMMARY_KEY: {**_summary(), "calibration": []}}) is False
        assert data == {}

    def test_apply_detector_meta_keeps_it_beside_the_input_spec(self):
        summary = _summary()
        data: dict = {"media_type": "audio"}
        meta = {"input_spec": {"clipper": "sound_tiling"}, DOMAIN_SUMMARY_KEY: summary}
        assert apply_detector_meta(data, meta) is True
        assert data["input_spec"] == {"clipper": "sound_tiling"}
        assert stored_domain_summary(data) == summary
        assert apply_detector_meta(data, meta) is False

    def test_on_file_it_writes_once(self, tmp_path):
        path = tmp_path / "copy.json"
        _write_detector(path, {"name": "copy", "labelset": {"labels": _labels()}})
        summary = _summary()
        assert adopt_domain_summary_on_file(path, {DOMAIN_SUMMARY_KEY: summary}) is True
        on_disk = _read_detector(path) or {}
        assert stored_domain_summary(on_disk) == summary
        assert on_disk["labelset"] == {"labels": _labels()}
        assert adopt_domain_summary_on_file(path, {DOMAIN_SUMMARY_KEY: summary}) is False
        assert adopt_domain_summary_on_file(tmp_path / "gone.json", {DOMAIN_SUMMARY_KEY: summary}) is False


class TestTheSyncCarriesIt:
    def test_a_push_sends_it(self):
        from vtscore.labels.sync import _push_payload

        summary = _summary()
        data = {
            "media_type": "audio",
            "labelset": {"labels": _labels()},
            TRAINING_DATASET_KEY: training_dataset_record("ds-a", "A", 2),
            DOMAIN_SUMMARY_KEY: summary,
        }
        pushed = _push_payload(data)
        assert pushed.detector_meta == {"media_type": "audio", DOMAIN_SUMMARY_KEY: summary}


class _LabelsOnly(LabelImporter):
    name = "_stub_labels_only"
    display_name = "Labels only"
    description = "Returns labels and nothing else."
    fields = [PluginField("filepath", "Path", "text")]

    def run(self, field_values):
        return _labels()


class TestImportersHandTheBlockOver:
    def test_an_importer_that_returns_labels_alone_still_works(self):
        importer = _LabelsOnly()
        assert importer.run_labelset({"filepath": "x"}) == {"labels": _labels()}
        assert importer.run_labelset_cli({"filepath": "x"}) == {"labels": _labels()}

    def test_the_json_exporter_writes_it_and_the_json_importer_reads_it_back(self, tmp_path):
        from vtscore.exporters.server_json_file import EXPORTER
        from vtscore.labels.importers.server_json_file import LABEL_IMPORTER

        summary = _summary()
        path = tmp_path / "shared.labels.json"
        payload = labelset_for_export({"labels": _labels(), "detector_meta": {DOMAIN_SUMMARY_KEY: summary}})
        EXPORTER.export_labelset({**payload, "selected_columns": ["md5", "label", "origin"]}, {"filepath": str(path)})

        assert json.loads(path.read_text())["detector_meta"] == {DOMAIN_SUMMARY_KEY: summary}
        read = LABEL_IMPORTER.run_labelset({"filepath": str(path)})
        assert read["detector_meta"] == {DOMAIN_SUMMARY_KEY: summary}
        assert [e["md5"] for e in read["labels"]] == ["md5-a-0", "md5-a-1"]
        assert LABEL_IMPORTER.run_labelset_cli({"filepath": str(path)}) == read
        # ``run`` keeps returning the labels alone, for callers of the old contract.
        assert LABEL_IMPORTER.run({"filepath": str(path)}) == read["labels"]

    def test_a_file_without_the_block_imports_labels_alone(self, tmp_path):
        from vtscore.labels.importers.server_json_file import LABEL_IMPORTER

        path = tmp_path / "plain.labels.json"
        path.write_text(json.dumps({"labels": _labels()}))
        assert LABEL_IMPORTER.run_labelset({"filepath": str(path)}) == {"labels": _labels()}
