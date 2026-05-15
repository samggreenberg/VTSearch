"""Tests for the detector CLI autodetect path.

Exercises the new ``autorun_detectors`` settings key, the
``--import-labels-into`` one-shot import flow, and the clear-error path
when a labelset's origin files can't be resolved from the CLI environment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import app as app_module
from helpers import make_dataset_file as _make_dataset_file
from vtsearch.settings import get_detectors_dir
from vtsearch.media.audio.audio_generator import generate_wav


@pytest.fixture(autouse=True)
def _clean_tm_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _write_trainable_model(name: str, labelset: dict) -> Path:
    from vtsearch.detectors.store import _detector_path, _write_detector

    path = _detector_path(name)
    _write_detector(
        path,
        {
            "name": name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "labelset": labelset,
        },
    )
    return path


def _stub_resolve(monkeypatch, file_map: dict[str, Path]) -> None:
    """Patch ``resolve_file_context`` to look up *file_map* by origin name."""
    from contextlib import contextmanager

    import vtsearch.detectors.resolver as resolver_mod

    @contextmanager
    def _fake_ctx(origin, origin_name="", filename=""):
        yield file_map.get(origin_name) or file_map.get(filename)

    monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)


def _make_audio_files(tmp_path: Path, names: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for i, name in enumerate(names):
        path = tmp_path / name
        path.write_bytes(generate_wav(220 + 110 * i, 0.1))
        out[name] = path
    return out


def _settings_file_with_detectors(tmp_path: Path, tm_names: list[str]) -> Path:
    """Settings JSON that activates *tm_names* but defines no autorun processors.

    Includes ``detectors_dir`` so ``set_settings_path`` doesn't reset
    the directory to the production default after conftest redirected it.
    """
    settings = {
        "autorun_detectors": list(tm_names),
        "detectors_dir": str(get_detectors_dir()),
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings))
    return settings_path


# ---------------------------------------------------------------------------
# autorun_detectors in settings drives CLI autodetect
# ---------------------------------------------------------------------------


class TestAutorunDetectorsCLI:
    def test_settings_drives_trainable_model_scoring(self, client, tmp_path, monkeypatch):
        """A settings file with autorun_detectors scores the dataset
        with each named detector.  No autorun_processors needed."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)

        labelset = {
            "labels": [
                {
                    "md5": "a" * 32,
                    "label": "good",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "alpha.wav",
                },
                {
                    "md5": "b" * 32,
                    "label": "good",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "beta.wav",
                },
                {
                    "md5": "c" * 32,
                    "label": "bad",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "gamma.wav",
                },
            ]
        }
        _write_trainable_model("ds-a-detector", labelset)

        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["ds-a-detector"])
        out_path = tmp_path / "hits.json"

        from vtsearch.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
        )

        body = json.loads(out_path.read_text())
        results = body.get("results", {})
        assert "ds-a-detector" in results, f"Expected detector in results, got {list(results)}"
        det = results["ds-a-detector"]
        assert isinstance(det.get("hits"), list)
        assert det["detector_name"] == "ds-a-detector"

    def test_clear_error_when_origins_unresolvable(self, client, tmp_path, monkeypatch):
        """No origins resolve → ValueError with a CLI-friendly explanation."""
        # Stub yields None for everything (simulates labels from local_folder).
        from contextlib import contextmanager

        import vtsearch.detectors.resolver as resolver_mod

        @contextmanager
        def _fake_ctx(*_a, **_kw):
            yield None

        monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)

        labelset = {
            "labels": [
                {
                    "md5": "a" * 32,
                    "label": "good",
                    "origin": {"importer": "local_folder", "params": {}},
                    "origin_name": "uploaded1",
                },
                {
                    "md5": "b" * 32,
                    "label": "bad",
                    "origin": {"importer": "local_folder", "params": {}},
                    "origin_name": "uploaded2",
                },
            ]
        }
        _write_trainable_model("unreachable-tm", labelset)

        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["unreachable-tm"])

        from vtsearch.cli import _run_pipeline, _load_pickle_whole

        with pytest.raises(ValueError) as exc:
            _run_pipeline(_load_pickle_whole(str(dataset_path)), settings_path=str(settings_path))
        msg = str(exc.value)
        assert "unreachable-tm" in msg
        assert "could not train" in msg.lower() or "resolve" in msg.lower()


# ---------------------------------------------------------------------------
# Label-import-then-score one-shot flow
# ---------------------------------------------------------------------------


class TestImportLabelsIntoDetectorCLI:
    def test_merges_external_labels_into_trainable_model(self, tmp_path):
        """Calling import_labels_into_detector_from_file with a
        server_json_file label file appends new entries to the on-disk
        labelset and dedupes by (md5, label)."""
        # Seed model with one existing entry — to verify dedup.
        existing = {
            "labels": [
                {
                    "md5": "a" * 32,
                    "label": "good",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "alpha.wav",
                },
            ]
        }
        _write_trainable_model("import-tm", existing)

        # External label file in the canonical {"labels": [...]} shape that
        # server_json_file label importer reads.
        new_labels = {
            "labels": [
                {"md5": "a" * 32, "label": "good"},  # duplicate — should skip
                {"md5": "b" * 32, "label": "good"},
                {"md5": "c" * 32, "label": "bad"},
            ]
        }
        labels_path = tmp_path / "new_labels.json"
        labels_path.write_text(json.dumps(new_labels))

        from vtsearch.cli import import_labels_into_detector_from_file
        from vtsearch.datasets.labelset import LabelSet
        from vtsearch.detectors.store import _detector_path, _read_detector

        applied, skipped = import_labels_into_detector_from_file(
            "import-tm",
            "server_json_file",
            str(labels_path),
        )
        assert applied == 2
        assert skipped == 1

        saved = _read_detector(_detector_path("import-tm"))
        ls = LabelSet.from_dict(saved["labelset"])
        md5s = sorted(el.md5 for el in ls.elements)
        assert md5s == sorted(["a" * 32, "b" * 32, "c" * 32])

    def test_unknown_trainable_model_raises_value_error(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps({"labels": []}))

        from vtsearch.cli import import_labels_into_detector_from_file

        with pytest.raises(ValueError) as exc:
            import_labels_into_detector_from_file("no-such-model", "server_json_file", str(labels_path))
        assert "no-such-model" in str(exc.value)
