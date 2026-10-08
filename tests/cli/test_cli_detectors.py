"""Tests for the detector CLI autodetect path.

Exercises the new ``autofind_detectors`` settings key, the
``--import-labels-into`` one-shot import flow, and the clear-error path
when a labelset's origin files can't be resolved from the CLI environment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.helpers import make_dataset_file as _make_dataset_file
from vtsearch.settings import get_detectors_dir
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.state import medias


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
    from vtscore.detectors.store import _detector_path, _write_detector

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

    import vtscore.detectors.resolver as resolver_mod

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
        "autofind_detectors": list(tm_names),
        "detectors_dir": str(get_detectors_dir()),
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings))
    return settings_path


# ---------------------------------------------------------------------------
# autofind_detectors in settings drives CLI autodetect
# ---------------------------------------------------------------------------


class TestAutofindDetectorsCLI:
    def test_settings_drives_trainable_model_scoring(self, client, tmp_path, monkeypatch):
        """A settings file with autofind_detectors scores the dataset
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

        dataset_path = _make_dataset_file(tmp_path, medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["ds-a-detector"])
        out_path = tmp_path / "hits.json"

        from vtscore.cli import autodetect_main

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

    def test_detector_with_mismatched_input_spec_is_reclipped(self, client, tmp_path, monkeypatch):
        """A detector whose input_spec.clipper doesn't match the dataset is now
        kept and re-clipped at scoring time (not skipped): its scoring info
        carries the clipper to re-apply."""
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
                    "md5": "c" * 32,
                    "label": "bad",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "gamma.wav",
                },
            ]
        }

        # Write the detector by hand so input_spec is present on disk.
        from vtscore.detectors.store import _detector_path, _write_detector

        _write_detector(
            _detector_path("spec-mismatch"),
            {
                "name": "spec-mismatch",
                "media_type": "audio",
                "labelset": labelset,
                "input_spec": {
                    "clipper": "sound_tiling",
                    "clipper_params": {"duration": "2.0"},
                },
            },
        )

        # The test fixture's medias have no clipper, so the detector's
        # sound_tiling(2.0) input_spec mismatches. It is kept (not skipped) and
        # its scoring info records the clipper to re-apply.
        from vtscore.cli import _load_and_train_detectors

        snap = dict(medias)
        trained = _load_and_train_detectors(["spec-mismatch"], "audio", snap)

        assert "spec-mismatch" in trained
        assert trained["spec-mismatch"]["clipper"] == "sound_tiling"
        assert trained["spec-mismatch"]["clipper_params"].get("duration") == "2.0"

    def test_detector_with_matching_input_spec_runs(self, client, tmp_path, monkeypatch):
        """When the dataset's clipper matches the detector's input_spec, the detector runs."""
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
                    "md5": "c" * 32,
                    "label": "bad",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "gamma.wav",
                },
            ]
        }

        from vtscore.detectors.store import _detector_path, _write_detector

        _write_detector(
            _detector_path("spec-matched"),
            {
                "name": "spec-matched",
                "media_type": "audio",
                "labelset": labelset,
                "input_spec": {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}},
            },
        )

        # Stamp the matching clipper config onto every test media's origin
        # so the CLI sees a dataset that was loaded with sound_tiling(2.0).
        # The fixture medias use the default importer with empty params; we
        # only mutate origin.params here, not the medias themselves.
        original_origins = {}
        try:
            for mid, media in medias.items():
                original_origins[mid] = media.get("origin")
                media["origin"] = {
                    "importer": "test",
                    "params": {
                        "clipper": "sound_tiling",
                        "clipper_duration": "2.0",
                    },
                }

            dataset_path = _make_dataset_file(tmp_path, medias)
            settings_path = _settings_file_with_detectors(tmp_path, ["spec-matched"])
            out_path = tmp_path / "hits.json"

            from vtscore.cli import autodetect_main

            autodetect_main(
                str(dataset_path),
                settings_path=str(settings_path),
                exporter_name="server_json_file",
                exporter_field_values={"filepath": str(out_path)},
            )
            body = json.loads(out_path.read_text())
            assert "spec-matched" in body.get("results", {})
        finally:
            for mid, origin in original_origins.items():
                if mid in medias:
                    medias[mid]["origin"] = origin

    def test_clear_error_when_origins_unresolvable(self, client, tmp_path, monkeypatch):
        """No origins resolve → ValueError with a CLI-friendly explanation."""
        # Stub yields None for everything (simulates labels from local_folder).
        from contextlib import contextmanager

        import vtscore.detectors.resolver as resolver_mod

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

        dataset_path = _make_dataset_file(tmp_path, medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["unreachable-tm"])

        from vtscore.cli import _run_pipeline, _load_pickle_whole

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
        # Seed model with one existing entry; to verify dedup.
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
                {"md5": "a" * 32, "label": "good"},  # duplicate; should skip
                {"md5": "b" * 32, "label": "good"},
                {"md5": "c" * 32, "label": "bad"},
            ]
        }
        labels_path = tmp_path / "new_labels.json"
        labels_path.write_text(json.dumps(new_labels))

        from vtscore.cli import import_labels_into_detector_from_file
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.store import _detector_path, _read_detector

        applied, skipped = import_labels_into_detector_from_file(
            "import-tm",
            "server_json_file",
            str(labels_path),
        )
        assert applied == 2
        assert skipped == 1

        saved = _read_detector(_detector_path("import-tm"))
        assert saved is not None
        ls = LabelSet.from_dict(saved["labelset"])
        md5s = sorted(el.md5 for el in ls.elements)
        assert md5s == sorted(["a" * 32, "b" * 32, "c" * 32])

    def test_unknown_trainable_model_raises_value_error(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps({"labels": []}))

        from vtscore.cli import import_labels_into_detector_from_file

        with pytest.raises(ValueError) as exc:
            import_labels_into_detector_from_file("no-such-model", "server_json_file", str(labels_path))
        assert "no-such-model" in str(exc.value)


# ---------------------------------------------------------------------------
# --import-labels-into picks the detector the run scores with (#4235)
# ---------------------------------------------------------------------------


class TestImportLabelsIntoRunsThatDetector:
    def test_flag_run_scores_imported_detector_not_autofind_list(self, client, tmp_path, monkeypatch):
        """``--autodetect --import-labels-into NAME`` scores with NAME alone:
        the settings file's AutoFind list names only a detector that does not
        exist, so the run fails unless that list is bypassed - and the
        detector is not on it, so it would never run if it were read."""
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
                    "md5": "c" * 32,
                    "label": "bad",
                    "origin": {"importer": "ds_a", "params": {}},
                    "origin_name": "gamma.wav",
                },
            ]
        }
        _write_trainable_model("Not On AutoFind", labelset)
        labels_path = tmp_path / "new_labels.json"
        labels_path.write_text(json.dumps({"labels": [{"md5": "a" * 32, "label": "good"}]}))

        dataset_path = _make_dataset_file(tmp_path, medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["nonexistent-detector"])
        out_path = tmp_path / "hits.json"

        from vtsearch import cli_main

        monkeypatch.setattr(
            "sys.argv",
            [
                "app.py",
                "--autodetect",
                "--tempimport",
                "--dataset",
                str(dataset_path),
                "--settings",
                str(settings_path),
                "--import-labels-into",
                "Not On AutoFind",
                "--label-importer-file",
                str(labels_path),
                "--exporter",
                "server_json_file",
                "--filepath",
                str(out_path),
            ],
        )
        cli_main.main(None, None)

        results = json.loads(out_path.read_text()).get("results", {})
        assert list(results) == ["Not On AutoFind"]
        # The settings file is read, never rewritten.
        on_disk = json.loads(settings_path.read_text())
        assert on_disk["autofind_detectors"] == ["nonexistent-detector"]


# ---------------------------------------------------------------------------
# Creating a missing detector from imported labels (#4238)
# ---------------------------------------------------------------------------


def _labels_file(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": entries}))
    return path


class TestCreateDetectorOnImport:
    def test_creates_and_registers_missing_detector(self, tmp_path):
        """With create_media_type a missing detector is written the way the
        Dashboard's New Detector writes one, and registered for its creator."""
        from vtscore.cli import import_labels_into_detector
        from vtscore.detectors.registry import find_by_name
        from vtscore.detectors.store import _detector_path, _read_detector

        labels = _labels_file(
            tmp_path,
            [
                {"md5": "a" * 32, "label": "good"},
                {"md5": "b" * 32, "label": "bad"},
                {"md5": "c" * 32, "label": "maybe"},
            ],
        )
        applied, skipped = import_labels_into_detector(
            "Fresh Det", "server_json_file", {"filepath": str(labels)}, create_media_type="audio"
        )
        assert (applied, skipped) == (2, 1)

        saved = _read_detector(_detector_path("Fresh Det"))
        assert saved is not None
        assert saved["name"] == "Fresh Det"
        assert saved["media_type"] == "audio"
        assert saved["examples"] == []
        assert saved["created_at"] > 0
        assert len(saved["labelset"]["labels"]) == 2
        # Origins and labels only: no vectors or weights ride along.
        assert "embeddings" not in json.dumps(saved) and "weights" not in saved

        entry = find_by_name("Fresh Det")
        assert entry is not None
        assert entry["media_type"] == "audio"
        assert entry["num_training"] == 2
        assert entry["created_by"] == "default"

    def test_created_detector_belongs_to_the_run_user(self, tmp_path):
        from vtscore.cli import import_labels_into_detector
        from vtscore.detectors.registry import find_by_name
        from vtscore.state.current_user import thread_user

        labels = _labels_file(tmp_path, [{"md5": "a" * 32, "label": "good"}])
        with thread_user("alice"):
            import_labels_into_detector(
                "Alice Det", "server_json_file", {"filepath": str(labels)}, create_media_type="image"
            )
        entry = find_by_name("Alice Det")
        assert entry is not None
        assert entry["created_by"] == "alice"

    def test_missing_detector_without_create_raises_not_found(self, tmp_path):
        from vtscore.cli import DetectorNotFoundError, import_labels_into_detector
        from vtscore.detectors.store import _detector_path, _read_detector

        labels = _labels_file(tmp_path, [{"md5": "a" * 32, "label": "good"}])
        with pytest.raises(DetectorNotFoundError) as exc:
            import_labels_into_detector("Ghost", "server_json_file", {"filepath": str(labels)})
        assert isinstance(exc.value, ValueError)
        assert exc.value.det_name == "Ghost"
        assert _read_detector(_detector_path("Ghost")) is None

    def test_nothing_created_without_good_or_bad_labels(self, tmp_path):
        from vtscore.cli import import_labels_into_detector
        from vtscore.detectors.registry import find_by_name
        from vtscore.detectors.store import _detector_path, _read_detector

        labels = _labels_file(tmp_path, [{"md5": "a" * 32, "label": "maybe"}])
        with pytest.raises(ValueError, match="no good/bad labels"):
            import_labels_into_detector(
                "Empty Det", "server_json_file", {"filepath": str(labels)}, create_media_type="audio"
            )
        assert _read_detector(_detector_path("Empty Det")) is None
        assert find_by_name("Empty Det") is None

    def test_existing_detector_is_merged_not_recreated(self, tmp_path):
        """create_media_type is only for a missing detector: an existing one
        keeps its media type and gets no new registry entry."""
        from vtscore.cli import import_labels_into_detector
        from vtscore.detectors.registry import find_by_name
        from vtscore.detectors.store import _detector_path, _read_detector

        _write_trainable_model("Existing", {"labels": [{"md5": "a" * 32, "label": "good"}]})
        labels = _labels_file(tmp_path, [{"md5": "b" * 32, "label": "bad"}])
        applied, _ = import_labels_into_detector(
            "Existing", "server_json_file", {"filepath": str(labels)}, create_media_type="image"
        )
        assert applied == 1
        saved = _read_detector(_detector_path("Existing"))
        assert saved is not None
        assert saved["media_type"] == "audio"
        assert len(saved["labelset"]["labels"]) == 2
        assert find_by_name("Existing") is None

    def test_orphaned_registry_entry_is_reused(self, tmp_path):
        """A registry entry whose labelset file is gone still owns the name, so
        creating the file must not register a second entry sharing it."""
        from vtscore.cli import import_labels_into_detector
        from vtscore.detectors.registry import list_detectors, register_detector

        register_detector(name="Orphan", media_type="audio")
        labels = _labels_file(tmp_path, [{"md5": "a" * 32, "label": "good"}])
        import_labels_into_detector("Orphan", "server_json_file", {"filepath": str(labels)}, create_media_type="audio")
        assert [e["name"] for e in list_detectors()].count("Orphan") == 1

    def test_unknown_create_media_type_rejected(self, tmp_path):
        from vtscore.cli import import_labels_into_detector

        labels = _labels_file(tmp_path, [{"md5": "a" * 32, "label": "good"}])
        with pytest.raises(ValueError, match="Unknown media type"):
            import_labels_into_detector(
                "Typo Det", "server_json_file", {"filepath": str(labels)}, create_media_type="audoi"
            )


class TestLabelImportMediaType:
    """Where a detector --create-detector makes gets its media type."""

    @staticmethod
    def _resolve(det_name, spec, **kw):
        from vtscore.cli import _label_import_media_type

        kw.setdefault("create", True)
        return _label_import_media_type(det_name, spec, media_type_option="--detector-media-type", **kw)

    def test_pickle_metadata_supplies_it(self, client, tmp_path):
        from vtscore.cli import _SourceSpec
        from vtscore.datasets.loader import export_dataset_to_file

        path = tmp_path / "typed.pkl"
        path.write_bytes(export_dataset_to_file(dict(medias), media_type="audio"))
        assert self._resolve("New", _SourceSpec(kind="pickle", dataset_path=str(path))) == "audio"

    def test_pickle_without_it_asks_for_the_flag(self, client, tmp_path):
        from vtscore.cli import _SourceSpec

        path = _make_dataset_file(tmp_path, medias)  # meta.json carries no media_type
        with pytest.raises(ValueError, match="--detector-media-type"):
            self._resolve("New", _SourceSpec(kind="pickle", dataset_path=str(path)))

    def test_missing_pickle_is_reported_as_missing(self, tmp_path):
        from vtscore.cli import _SourceSpec

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            self._resolve("New", _SourceSpec(kind="pickle", dataset_path=str(tmp_path / "nope.pkl")))

    def test_importer_field_supplies_it(self):
        from vtscore.cli import _SourceSpec

        spec = _SourceSpec(
            kind="importer", importer_name="server_folder", field_values={"path": "/x", "media_type": "video"}
        )
        assert self._resolve("New", spec) == "video"

    def test_importer_field_default_applies(self):
        from vtscore.cli import _SourceSpec

        spec = _SourceSpec(kind="importer", importer_name="server_folder", field_values={"path": "/x"})
        assert self._resolve("New", spec) == "audio"

    def test_importer_without_the_field_asks_for_the_flag(self):
        from vtscore.cli import _SourceSpec

        spec = _SourceSpec(kind="importer", importer_name="local_folder", field_values={})
        with pytest.raises(ValueError, match="--detector-media-type"):
            self._resolve("New", spec)

    def test_explicit_media_type_wins(self):
        from vtscore.cli import _SourceSpec

        spec = _SourceSpec(
            kind="importer", importer_name="server_folder", field_values={"path": "/x", "media_type": "video"}
        )
        assert self._resolve("New", spec, media_type="image") == "image"

    def test_existing_detector_needs_none(self):
        from vtscore.cli import _SourceSpec

        _write_trainable_model("Existing", {"labels": []})
        spec = _SourceSpec(kind="importer", importer_name="local_folder", field_values={})
        assert self._resolve("Existing", spec) == ""

    def test_create_off_needs_none(self):
        from vtscore.cli import _SourceSpec

        spec = _SourceSpec(kind="importer", importer_name="local_folder", field_values={})
        assert self._resolve("Missing", spec, create=False) == ""

    def test_unknown_explicit_media_type_rejected_even_for_existing_detector(self):
        from vtscore.cli import _SourceSpec

        _write_trainable_model("Existing", {"labels": []})
        spec = _SourceSpec(kind="importer", importer_name="server_folder", field_values={"path": "/x"})
        with pytest.raises(ValueError, match="Unknown media type for --detector-media-type"):
            self._resolve("Existing", spec, media_type="imgae")


class TestCreateDetectorEndToEnd:
    def test_flag_run_creates_and_scores_new_detector(self, client, tmp_path, monkeypatch):
        """The whole #4235 hope: labels + a dataset in, hits out, with the
        detector made along the way and never touched in the UI."""
        from vtsearch import cli_main
        from vtscore.datasets.loader import export_dataset_to_file
        from vtscore.detectors.registry import find_by_name
        from vtscore.detectors.store import _detector_path, _read_detector

        snapshot = dict(medias)
        ids = sorted(snapshot)
        labels = _labels_file(
            tmp_path,
            [
                {"md5": snapshot[ids[0]]["md5"], "label": "good"},
                {"md5": snapshot[ids[-1]]["md5"], "label": "bad"},
            ],
        )
        dataset_path = tmp_path / "typed.pkl"
        dataset_path.write_bytes(export_dataset_to_file(snapshot, media_type="audio"))
        settings_path = _settings_file_with_detectors(tmp_path, ["nonexistent-detector"])
        out_path = tmp_path / "hits.json"

        argv = [
            "app.py",
            "--autodetect",
            "--tempimport",
            "--dataset",
            str(dataset_path),
            "--settings",
            str(settings_path),
            "--import-labels-into",
            "Brand New",
            "--create-detector",
            "--label-importer-file",
            str(labels),
            "--exporter",
            "server_json_file",
            "--filepath",
            str(out_path),
        ]
        monkeypatch.setattr("sys.argv", argv)
        cli_main.main(None, None)

        results = json.loads(out_path.read_text()).get("results", {})
        assert list(results) == ["Brand New"]
        saved = _read_detector(_detector_path("Brand New"))
        assert saved is not None
        assert saved["media_type"] == "audio"
        entry = find_by_name("Brand New")
        assert entry is not None
        assert entry["created_by"] == "default"
