"""Tests for the ``portable_detector`` results exporter (issue #2391).

The exporter is the CLI / CI counterpart to the GUI's portable-export modal:
run ``--autodetect --exporter portable_detector`` and, instead of the scored
hits, it writes one standalone ONNX scoring bundle per trained detector.  These
tests drive both the end-to-end CLI pipeline path and the exporter's own
:meth:`export_cli_detectors` (skip rules, multi-detector disambiguation).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from contextlib import contextmanager
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


def _labelset(good: list[str], bad: list[str]) -> dict:
    labels = []
    for i, n in enumerate(good):
        labels.append(
            {"md5": f"{i:032x}", "label": "good", "origin": {"importer": "ds_a", "params": {}}, "origin_name": n}
        )
    for i, n in enumerate(bad):
        labels.append(
            {"md5": f"{i + 100:032x}", "label": "bad", "origin": {"importer": "ds_a", "params": {}}, "origin_name": n}
        )
    return {"labels": labels}


def _settings_file(tmp_path: Path, names: list[str]) -> Path:
    settings = {"autofind_detectors": list(names), "detectors_dir": str(get_detectors_dir())}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return path


def _assert_valid_bundle(zip_path: Path, *, detector_name: str) -> dict:
    """Assert *zip_path* is a well-formed portable-detector bundle; return the manifest."""
    from vtscore.detectors import portable_bundle as pb

    assert zip_path.exists(), f"expected bundle at {zip_path}"
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["README.md", "detector.onnx", "manifest.json"]
        manifest = json.loads(zf.read("manifest.json"))
        readme = zf.read("README.md").decode()
        onnx_bytes = zf.read("detector.onnx")
    assert manifest["format"] == pb.BUNDLE_FORMAT
    assert manifest["detector_name"] == detector_name
    assert manifest["contains_media_data"] is False
    # The bundle carries the classifier only: no embeddings smuggled anywhere.
    assert "no raw media" in readme.lower()
    assert onnx_bytes[:4] != b"PK\x03\x04"  # sanity: it's the onnx, not a nested zip
    return manifest


# ---------------------------------------------------------------------------
# End-to-end: CLI autodetect writes a bundle per detector
# ---------------------------------------------------------------------------


class TestPortableDetectorCLI:
    def test_writes_one_bundle_per_detector(self, client, tmp_path, monkeypatch):
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)
        _write_trainable_model("cats", _labelset(good=["alpha.wav", "beta.wav"], bad=["gamma.wav"]))

        dataset_path = _make_dataset_file(tmp_path, medias)
        settings_path = _settings_file(tmp_path, ["cats"])
        out_template = str(tmp_path / "{detector_name}-detector.zip")

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            settings_path=str(settings_path),
            exporter_name="portable_detector",
            exporter_field_values={"filepath": out_template},
        )

        manifest = _assert_valid_bundle(tmp_path / "cats-detector.zip", detector_name="cats")
        assert manifest["media_type"] == "audio"
        assert manifest["training_labels"] == {"good": 2, "bad": 1}
        # The embedder the labels trained in is recorded so a recipient can reproduce it.
        assert manifest["embedder"]["name"]
        assert manifest["embedder"]["embedding_dim"] > 0

    def test_multi_detector_disambiguates_when_no_placeholder(self, client, tmp_path, monkeypatch):
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)
        ls = _labelset(good=["alpha.wav", "beta.wav"], bad=["gamma.wav"])
        _write_trainable_model("cats", ls)
        _write_trainable_model("dogs", ls)

        dataset_path = _make_dataset_file(tmp_path, medias)
        settings_path = _settings_file(tmp_path, ["cats", "dogs"])
        # No {detector_name} in the path: the exporter must insert the slug so
        # the two bundles don't overwrite each other.
        out_path = str(tmp_path / "bundle.zip")

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            settings_path=str(settings_path),
            exporter_name="portable_detector",
            exporter_field_values={"filepath": out_path},
        )

        _assert_valid_bundle(tmp_path / "bundle-cats.zip", detector_name="cats")
        _assert_valid_bundle(tmp_path / "bundle-dogs.zip", detector_name="dogs")
        assert not (tmp_path / "bundle.zip").exists()


# ---------------------------------------------------------------------------
# Exporter unit behaviour
# ---------------------------------------------------------------------------


def _real_weights(input_dim: int = 512, hidden: int = 8) -> dict:
    import torch  # noqa: PLC0415
    from vtscore.detectors.training import serialize_weights  # noqa: PLC0415
    from vtscore.training.mlp import build_model  # noqa: PLC0415

    gen = torch.Generator().manual_seed(0)
    model = build_model(input_dim, hidden_dim=hidden, dropout=0.5, generator=gen)
    model.eval()
    return serialize_weights(model)


class TestExportCliDetectors:
    def test_export_results_path_is_unsupported(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("portable_detector")
        assert exp.needs_trained_detectors is True
        with pytest.raises(NotImplementedError, match="trained classifier"):
            exp.export({"results": {}}, {"filepath": "x.zip"})

    def test_writes_bundle_and_reports(self, tmp_path):
        from vtscore.exporters import get_exporter

        exp = get_exporter("portable_detector")
        descriptor = {
            "detector_name": "cats",
            "media_type": "image",
            "weights": _real_weights(768),
            "threshold": 0.6,
            "embedder": "siglip",
            "embedder_type": "semantic",
            "good_count": 5,
            "bad_count": 3,
        }
        out = tmp_path / "{detector_name}.zip"
        result = exp.export_cli_detectors([descriptor], {"filepath": str(out)})

        assert "Wrote 1" in result["message"]
        manifest = _assert_valid_bundle(tmp_path / "cats.zip", detector_name="cats")
        assert manifest["embedder"]["embedding_dim"] == 768
        assert manifest["training_labels"] == {"good": 5, "bad": 3}

    def test_skips_malformed_weights(self, tmp_path):
        from vtscore.exporters import get_exporter

        exp = get_exporter("portable_detector")
        # The fixed ONNX graph models the linear head (1 Linear) and the 2-layer
        # MLP; a 3-layer stack can't be modelled, so the exporter must skip it
        # rather than abort the whole export.  (A single-layer dict is the valid
        # production linear head, not malformed - so we can't use that here.)
        bad_descriptor = {
            "detector_name": "malformed",
            "media_type": "image",
            "weights": {
                "0.weight": [[1.0, 2.0]],
                "0.bias": [0.0],
                "3.weight": [[1.0]],
                "3.bias": [0.0],
                "6.weight": [[1.0]],
                "6.bias": [0.0],
            },
            "threshold": 0.5,
            "embedder": "siglip",
            "embedder_type": "semantic",
            "good_count": 2,
            "bad_count": 2,
        }
        out = tmp_path / "{detector_name}.zip"
        result = exp.export_cli_detectors([bad_descriptor], {"filepath": str(out)})

        assert "No portable detector bundles written" in result["message"]
        assert "skipped" in result["message"].lower()
        assert not (tmp_path / "malformed.zip").exists()

    def test_skips_structural_detector(self, tmp_path):
        """A real (well-formed) structural detector is blocked, not silently mis-exported.

        Structural detectors' stage-1 VLAD-space MLP is a normal 2-layer weight
        dict indistinguishable in shape from any other detector's, so the
        weight-shape check alone can't catch it - the exporter must gate on
        ``embedder_type`` explicitly.
        """
        from vtscore.exporters import get_exporter

        exp = get_exporter("portable_detector")
        descriptor = {
            "detector_name": "structural",
            "media_type": "image",
            "weights": _real_weights(768),
            "threshold": 0.5,
            "embedder": "sift_vlad",
            "embedder_type": "structural",
            "good_count": 2,
            "bad_count": 2,
        }
        out = tmp_path / "{detector_name}.zip"
        result = exp.export_cli_detectors([descriptor], {"filepath": str(out)})

        assert "No portable detector bundles written" in result["message"]
        assert "structural" in result["message"].lower()
        assert not (tmp_path / "structural.zip").exists()

    def test_exports_patch_semantic_with_caveat(self, tmp_path):
        """A patch detector exports in the degraded whole-item-only mode, flagged."""
        from vtscore.exporters import get_exporter

        exp = get_exporter("portable_detector")
        descriptor = {
            "detector_name": "patchy",
            "media_type": "image",
            "weights": _real_weights(768),
            "threshold": 0.5,
            "embedder": "dinov2",
            "embedder_type": "patch_semantic",
            "good_count": 4,
            "bad_count": 4,
        }
        out = tmp_path / "{detector_name}.zip"
        result = exp.export_cli_detectors([descriptor], {"filepath": str(out)})

        assert "Wrote 1" in result["message"]
        manifest = _assert_valid_bundle(tmp_path / "patchy.zip", detector_name="patchy")
        assert len(manifest["caveats"]) == 1
        assert "WHOLE item" in manifest["caveats"][0]

    def test_sanitizes_detector_name_in_path(self, tmp_path):
        """A detector name with path separators can't escape the target dir."""
        from vtscore.exporters import get_exporter

        exp = get_exporter("portable_detector")
        descriptor = {
            "detector_name": "a/../b",
            "media_type": "image",
            "weights": _real_weights(512),
            "threshold": 0.5,
            "embedder": "",
            "embedder_type": "",
            "good_count": 1,
            "bad_count": 1,
        }
        out = tmp_path / "{detector_name}-detector.zip"
        result = exp.export_cli_detectors([descriptor], {"filepath": str(out)})

        written = Path(result["filepaths"][0])
        assert written.parent == tmp_path
        assert "/" not in written.name.replace("-detector.zip", "")
        assert written.exists()
