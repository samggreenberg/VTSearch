"""End-to-end tests for the ``--stream-results`` CLI autodetect path.

Streaming scores each chunk and writes its hits straight to the exporter
with no global accumulation, so a media source larger than RAM can be
scanned.  These tests pin: (1) NDJSON streaming export shape, (2) negatives
dropped by default but kept with ``keep_negatives``, and (3) a clear error
when the chosen exporter can't stream.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import pytest

from vtsearch.settings import get_detectors_dir


def _unique_bytes(media_id: int) -> bytes:
    return media_id.to_bytes(4, "little") + b"\x00" * 96


def _make_audio_media(media_id: int) -> dict:
    raw = _unique_bytes(media_id)
    # 2-dim *unit-norm* embedding whose dim 0 strictly decreases with id
    # (0.9, 0.8, 0.7, 0.6, 0.5 for ids 1..5).  Embeddings are L2-normalized at
    # ingest, so we store a unit vector to make that normalization a no-op and
    # keep dim 0 the clean, id-ordered signal the stubbed MLP thresholds on.
    e0 = 1.0 - 0.1 * media_id
    e1 = (1.0 - e0 * e0) ** 0.5
    return {
        "id": media_id,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "embedding": [e0, e1],
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
        "filename": f"clip_{media_id:03d}.wav",
        "category": "test",
        "origin": {"importer": "stub_ds", "params": {}},
        "origin_name": f"clip_{media_id:03d}.wav",
    }


def _write_pickle_dataset(path: Path, medias: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"medias": medias}, f)


def _settings_file_with_detector(tmp_path: Path, detector_name: str) -> Path:
    settings = {"autorun_detectors": [detector_name], "detectors_dir": str(get_detectors_dir())}
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return p


def _write_pretrained_detector(name: str) -> None:
    from vtscore.detectors.store import _detector_path, _write_detector

    _write_detector(
        _detector_path(name),
        {
            "name": name,
            "media_type": "audio",
            "labelset": {
                "labels": [
                    {"md5": "a" * 32, "label": "good", "origin": {"importer": "s", "params": {}}, "origin_name": "a"},
                    {"md5": "b" * 32, "label": "bad", "origin": {"importer": "s", "params": {}}, "origin_name": "b"},
                ]
            },
        },
    )


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


@pytest.fixture
def _stub_split_training(monkeypatch):
    """Train a deterministic MLP whose logit is ``100 * (embedding[0] - 0.65)``.

    With the unit-norm embeddings from ``_make_audio_media`` (dim 0 = 0.9, 0.8,
    0.7, 0.6, 0.5 for ids 1..5) and threshold 0.5 (sigmoid), the 0.65 boundary
    puts ids 1/2/3 above threshold (good) and 4/5 below (bad), giving a fixed
    positive/negative split to assert against.  The wide ``100`` scale keeps the
    sigmoid margins clear of float-precision wobble.
    """
    import torch
    from torch import nn

    import vtscore.cli as cli_mod

    def _fake_load_and_train(detector_names, media_type, first_chunk_medias):
        linear = nn.Linear(2, 1)
        with torch.no_grad():
            linear.weight.data = torch.tensor([[100.0, 0.0]])
            linear.bias.data = torch.tensor([-65.0])
        mlp = nn.Sequential(linear)
        mlp.eval()
        return {name: {"mlp": mlp, "threshold": 0.5} for name in detector_names}

    monkeypatch.setattr(cli_mod, "_load_and_train_detectors", _fake_load_and_train)


def _read_ndjson(path: Path) -> tuple[dict, list[dict]]:
    """Return ``(meta, hit_records)`` from an NDJSON export file."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    objs = [json.loads(ln) for ln in lines]
    meta = objs[0]["_meta"]
    return meta, objs[1:]


class TestStreamingNdjsonExport:
    def test_default_streams_only_positive_hits(self, client, tmp_path, _stub_split_training):
        _write_pretrained_detector("stream-tm")
        settings_path = _settings_file_with_detector(tmp_path, "stream-tm")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})
        out = tmp_path / "hits.ndjson"

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            dataset_path=str(ds_path),
            chunk_size=2,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
        )

        meta, hits = _read_ndjson(out)
        assert meta["format"] == "vtsearch-hits-ndjson/v1"
        assert meta["keep_negatives"] is False
        assert {d["detector_name"] for d in meta["detectors"]} == {"stream-tm"}
        # ids 1,2,3 are above threshold; negatives dropped by default.
        assert all(h["label"] == "good" for h in hits)
        assert sorted(h["id"] for h in hits) == [1, 2, 3]
        assert all(h["detector"] == "stream-tm" for h in hits)
        # The atomic temp file must not be left behind.
        assert not out.with_name(out.name + ".tmp").exists()

    def test_keep_negatives_streams_both(self, client, tmp_path, _stub_split_training):
        _write_pretrained_detector("stream-tm2")
        settings_path = _settings_file_with_detector(tmp_path, "stream-tm2")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})
        out = tmp_path / "hits.ndjson"

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            dataset_path=str(ds_path),
            chunk_size=2,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
            keep_negatives=True,
        )

        meta, hits = _read_ndjson(out)
        assert meta["keep_negatives"] is True
        good = sorted(h["id"] for h in hits if h["label"] == "good")
        bad = sorted(h["id"] for h in hits if h["label"] == "bad")
        assert good == [1, 2, 3]
        assert bad == [4, 5]


class TestStreamingExporterGuard:
    def test_non_streaming_exporter_raises(self):
        from vtscore.cli import _run_streaming_pipeline

        with pytest.raises(ValueError, match="does not support --stream-results"):
            _run_streaming_pipeline(
                iter([{1: _make_audio_media(1)}]),
                exporter_name="webhook",
                exporter_field_values={},
                override_detectors=None,
                autorun_detectors=[],
                keep_negatives=False,
                empty_error="none",
            )
