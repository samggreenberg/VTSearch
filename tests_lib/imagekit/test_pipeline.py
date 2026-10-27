"""imagekit.face.pipeline and providers: the registry, the oss pipeline end to end, the ext wrapper on a fake backend."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from imagekit import RuntimeOptions
from imagekit.face import (
    DEFAULT_PROVIDER,
    PROVIDER_ENV_VAR,
    CompatibilityError,
    FacePipeline,
    FaceRecord,
    available_providers,
    check_compat,
    default_provider_name,
    get_pipeline,
    load_pipeline,
    read_pipeline_config,
    register_provider,
    resolve_config_paths,
    runtime_options_from_config,
    stamp_compat,
    unregister_provider,
)
from imagekit.face import pipeline as pipeline_module
from imagekit.face.providers.ext import (
    BACKEND_ENV_VAR,
    DEFAULT_BACKEND,
    ExtFacePipeline,
    ExtProviderUnavailable,
    backend_module_name,
    import_backend,
)
from imagekit.face.providers.oss import OssFacePipeline
from tests_lib.imagekit._synthetic_models import build_model_dir, pipeline_config, scrfd_expected, scrfd_face

KPS = ((-5, -5), (5, -5), (0, 0), (-4, 6), (4, 6))
FACES = [scrfd_face(0, 20, 30, 0.9, (16, 16, 16, 16), kps=KPS), scrfd_face(2, 10, 5, 0.7, (40, 40, 40, 40), kps=KPS)]
BLACK = np.zeros((640, 640, 3), np.uint8)


class FakePipeline:
    """A registrable stand-in that satisfies the protocol."""

    def __init__(self) -> None:
        self.initialized_with: tuple[Any, Any] | None = None

    def initialize(self, opts, config):
        self.initialized_with = (opts, config)

    def process(self, image, *, confidence_threshold=None, media_id=""):
        return []

    def process_buffer(self, buffer, *, confidence_threshold=None, media_id=""):
        return []

    def compatibility(self):
        return "fake:1"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtins_are_listed_and_built_fresh(self):
        assert available_providers() == ["ext", "oss"]
        a, b = get_pipeline("oss"), get_pipeline("oss")
        assert isinstance(a, OssFacePipeline) and a is not b
        assert isinstance(get_pipeline("ext"), ExtFacePipeline)
        assert isinstance(a, FacePipeline) and isinstance(FakePipeline(), FacePipeline)

    def test_unknown_provider_names_the_available_ones(self):
        with pytest.raises(KeyError, match="available: \\['ext', 'oss'\\]"):
            get_pipeline("nope")

    def test_default_comes_from_the_environment(self, monkeypatch):
        monkeypatch.delenv(PROVIDER_ENV_VAR, raising=False)
        assert default_provider_name() == DEFAULT_PROVIDER == "oss"
        register_provider("fake", FakePipeline)
        monkeypatch.setenv(PROVIDER_ENV_VAR, " fake ")
        assert default_provider_name() == "fake"
        assert isinstance(get_pipeline(), FakePipeline)

    def test_register_replace_unregister(self):
        register_provider("fake", FakePipeline)
        with pytest.raises(ValueError, match="already registered"):
            register_provider("fake", FakePipeline)
        register_provider("fake", FakePipeline, replace=True)
        assert "fake" in available_providers()
        assert unregister_provider("fake") and not unregister_provider("fake")
        assert "fake" not in available_providers()
        with pytest.raises(ValueError):
            register_provider("  ", FakePipeline)

    def test_entry_points_register_on_first_miss(self, monkeypatch, caplog):
        class EP:
            def __init__(self, name, value, factory):
                self.name, self.value, self._factory = name, value, factory

            def load(self):
                if isinstance(self._factory, Exception):
                    raise self._factory
                return self._factory

        eps = [
            EP("acme", "acme:Pipe", FakePipeline),
            EP("oss", "evil:Pipe", FakePipeline),
            EP("broken", "b:x", RuntimeError("boom")),
        ]
        monkeypatch.setattr(
            pipeline_module, "entry_points", lambda group: eps if group == pipeline_module.ENTRY_POINT_GROUP else []
        )
        with caplog.at_level(logging.WARNING, logger=pipeline_module.__name__):
            assert isinstance(get_pipeline("acme"), FakePipeline)
        assert isinstance(get_pipeline("oss"), OssFacePipeline)  # the built-in is not shadowed
        assert available_providers() == ["acme", "ext", "oss"]
        assert "taken" in caplog.text and "failed to load" in caplog.text


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_runtime_options_from_config(self):
        assert runtime_options_from_config({}) == RuntimeOptions()
        opts = runtime_options_from_config({"runtime": {"use_gpu": True, "gpu_device_id": 2, "num_threads": 3}})
        assert opts == RuntimeOptions(True, 2, 3)
        with pytest.raises(ValueError):
            runtime_options_from_config({"runtime": 5})

    def test_paths_resolve_against_the_config_file(self, tmp_path):
        raw = {
            "provider": "oss",
            "detector": {"model_bundle": "models/det.onnx", "confidence_threshold": 0.3},
            "embedder": {"model": "/abs/emb.onnx", "config": "models/emb.json"},
        }
        resolved = resolve_config_paths(raw, tmp_path)
        assert resolved["detector"]["model_bundle"] == str((tmp_path / "models/det.onnx").resolve())
        assert resolved["embedder"]["model"] == "/abs/emb.onnx"
        assert resolved["embedder"]["config"] == str((tmp_path / "models/emb.json").resolve())
        assert resolved["detector"]["confidence_threshold"] == 0.3
        assert raw["detector"]["model_bundle"] == "models/det.onnx"  # the input is untouched
        (tmp_path / "cfg.json").write_text(json.dumps(raw))
        assert read_pipeline_config(tmp_path / "cfg.json") == resolved
        (tmp_path / "list.json").write_text("[]")
        with pytest.raises(ValueError):
            read_pipeline_config(tmp_path / "list.json")

    def test_load_pipeline_picks_provider_and_runtime(self):
        register_provider("fake", FakePipeline)
        config = {"provider": "fake", "runtime": {"num_threads": 4}}
        pipe = load_pipeline(config)
        assert isinstance(pipe, FakePipeline) and pipe.initialized_with == (RuntimeOptions(num_threads=4), config)
        explicit = load_pipeline({}, RuntimeOptions(num_threads=1), name="fake")
        assert isinstance(explicit, FakePipeline)
        assert explicit.initialized_with == (RuntimeOptions(num_threads=1), {})


# ---------------------------------------------------------------------------
# The oss pipeline
# ---------------------------------------------------------------------------


class TestOssPipeline:
    @pytest.fixture
    def pipe(self, tmp_path: Path) -> OssFacePipeline:
        paths = build_model_dir(tmp_path / "models", FACES)
        pipe = OssFacePipeline()
        pipe.initialize(RuntimeOptions(), pipeline_config(paths))
        return pipe

    def test_before_initialize(self):
        pipe = OssFacePipeline()
        with pytest.raises(RuntimeError, match="Must initialize"):
            pipe.process(BLACK)
        with pytest.raises(RuntimeError):
            pipe.compatibility()

    def test_missing_model_keys_are_named(self, tmp_path):
        pipe = OssFacePipeline()
        with pytest.raises(ValueError, match="detector"):
            pipe.initialize(RuntimeOptions(), {"embedder": {"model": "x"}})
        with pytest.raises(ValueError, match="embedder"):
            pipe.initialize(RuntimeOptions(), {"detector": {"model_bundle": "x"}})
        with pytest.raises(ValueError):
            pipe.initialize(RuntimeOptions(), {"detector": "x"})

    def test_every_detection_becomes_a_record(self, pipe):
        records = pipe.process(BLACK, media_id="img")
        assert len(records) == 2 and all(isinstance(r, FaceRecord) for r in records)
        assert [r.face_index for r in records] == [0, 1] and {r.media_id for r in records} == {"img"}
        assert [round(r.confidence, 2) for r in records] == [0.9, 0.7]
        assert records[0].bbox == pytest.approx(scrfd_expected(FACES[0])["bbox"])
        assert all(r.compat == "oss-synthetic:1" for r in records) and pipe.compatibility() == "oss-synthetic:1"
        for r in records:
            assert r.embedding.shape == (512,) and r.quality == pytest.approx(np.linalg.norm(r.embedding))
            assert np.linalg.norm(r.unit_embedding()) == pytest.approx(1.0)

    def test_threshold_from_config_and_call(self, tmp_path):
        paths = build_model_dir(tmp_path / "models", FACES)
        pipe = OssFacePipeline()
        pipe.initialize(
            RuntimeOptions(),
            pipeline_config(paths, detector={"model_bundle": str(paths["detector"]), "confidence_threshold": 0.8}),
        )
        assert len(pipe.process(BLACK)) == 1
        assert len(pipe.process(BLACK, confidence_threshold=0.5)) == 2

    def test_process_buffer(self, pipe):
        ok, png = cv2.imencode(".png", BLACK)
        assert ok
        assert len(pipe.process_buffer(png.tobytes(), media_id="b")) == 2


# ---------------------------------------------------------------------------
# The ext wrapper
# ---------------------------------------------------------------------------


class _Det:
    def __init__(self, bbox, landmarks, confidence):
        self.bbox, self.landmarks, self.confidence = bbox, landmarks, confidence


class _AliasDet:
    """A detection spelt with the other plausible attribute names."""

    def __init__(self, box, kps, score):
        self.box, self.kps, self.score = box, kps, score


class _Embedding:
    def __init__(self, values):
        self.values = values


def _fake_backend(monkeypatch, detections, compat: Any = "fakeface:3") -> dict[str, Any]:
    """Install a fake backend module under the configured name (:func:`backend_module_name`); return a call log."""
    log: dict[str, Any] = {"runtime": [], "detector": [], "extractor": [], "detect": [], "extract": []}

    class RuntimeOptionsB:
        def __init__(self, *args):
            log["runtime"].append(args)

    class FaceDetectorB:
        def initialize(self, runtime_opts, model_bundle):
            log["detector"].append((runtime_opts, model_bundle))

        def detect(self, image_tensor, confidence_threshold):
            log["detect"].append((image_tensor.shape, confidence_threshold))
            return list(detections)

    class FaceEmbeddingExtractorB:
        def initialize(self, runtime_opts, model_filepath, model_config_path=None):
            log["extractor"].append((runtime_opts, model_filepath, model_config_path))

        def extract(self, detection, image_tensor):
            log["extract"].append(detection)
            return _Embedding([float(i) for i in range(512)])

        def get_compatibility(self):
            return compat

    name = backend_module_name()
    backend = types.ModuleType(name)
    face = types.ModuleType(f"{name}.face")
    face.FaceDetector = FaceDetectorB  # type: ignore[attr-defined]
    face.FaceEmbeddingExtractor = FaceEmbeddingExtractorB  # type: ignore[attr-defined]
    backend.face = face  # type: ignore[attr-defined]
    backend.RuntimeOptions = RuntimeOptionsB  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, backend)
    return log


_CONFIG = {"detector": {"model_bundle": "/v/det.bin"}, "embedder": {"model": "/v/emb.bin", "config": "/v/emb.cfg"}}


class TestExtBackendName:
    def test_the_default(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        assert backend_module_name() == backend_module_name({}) == DEFAULT_BACKEND == "ext_backend"

    def test_the_config_names_it_and_the_environment_wins(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        assert backend_module_name({"ext_backend": " acme.faces "}) == "acme.faces"
        monkeypatch.setenv(BACKEND_ENV_VAR, " corp_faces ")
        assert backend_module_name() == backend_module_name({"ext_backend": "acme.faces"}) == "corp_faces"


class TestExtPipeline:
    def test_the_module_imports_without_the_backend(self, monkeypatch):
        monkeypatch.setitem(sys.modules, DEFAULT_BACKEND, None)
        pipe = get_pipeline("ext")  # building the wrapper never imports it
        assert isinstance(pipe, ExtFacePipeline)
        with pytest.raises(ExtProviderUnavailable, match="'ext_backend'"):
            pipe.initialize(RuntimeOptions(), _CONFIG)
        assert issubclass(ExtProviderUnavailable, ImportError)

    def test_the_configured_module_is_the_one_imported_and_named(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "tests_fake_ext_backend")
        log = _fake_backend(monkeypatch, [])
        assert "tests_fake_ext_backend" in sys.modules
        monkeypatch.setitem(sys.modules, DEFAULT_BACKEND, None)  # the default is never consulted
        ExtFacePipeline().initialize(RuntimeOptions(), _CONFIG)
        assert log["detector"][0][1] == "/v/det.bin"
        monkeypatch.setenv(BACKEND_ENV_VAR, "no_such_pkg.backend")
        with pytest.raises(ExtProviderUnavailable, match="'no_such_pkg.backend'"):
            import_backend()

    def test_the_config_names_the_module_when_the_environment_does_not(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        monkeypatch.setitem(sys.modules, DEFAULT_BACKEND, None)
        with pytest.raises(ExtProviderUnavailable, match="'from_the_config'"):
            ExtFacePipeline().initialize(RuntimeOptions(), {**_CONFIG, "ext_backend": "from_the_config"})

    def test_before_initialize(self):
        pipe = ExtFacePipeline()
        with pytest.raises(RuntimeError, match="Must initialize"):
            pipe.process(BLACK)
        with pytest.raises(RuntimeError):
            pipe.compatibility()

    def test_conversion_compat_and_ordering(self, monkeypatch):
        lm = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        log = _fake_backend(monkeypatch, [_Det((1, 2, 3, 4), lm, 0.5), _AliasDet([10, 20, 30, 40], None, 0.9)])
        pipe = ExtFacePipeline()
        pipe.initialize(RuntimeOptions(use_gpu=True, gpu_device_id=1, num_threads=2), _CONFIG)
        assert log["runtime"] == [(True, 1, 2)]
        assert log["detector"][0][1] == "/v/det.bin" and log["extractor"][0][1:] == ("/v/emb.bin", "/v/emb.cfg")
        assert pipe.compatibility() == "fakeface:3"
        records = pipe.process(BLACK, confidence_threshold=0.4, media_id="m")
        assert log["detect"] == [((640, 640, 3), 0.4)] and len(log["extract"]) == 2
        assert [r.confidence for r in records] == [0.9, 0.5]  # best first, whatever the backend's order
        assert all(type(r) is FaceRecord for r in records) and all(r.compat == "fakeface:3" for r in records)
        best, other = records
        assert best.bbox == (10.0, 20.0, 30.0, 40.0) and best.landmarks is None and best.media_id == "m"
        assert other.landmarks is not None and other.landmarks.shape == (5, 2) and other.landmarks.dtype == np.float32
        assert best.embedding.shape == (512,) and best.embedding.dtype == np.float32 and best.embedding[511] == 511.0
        assert best.quality == pytest.approx(np.linalg.norm(best.embedding))

    @pytest.mark.parametrize("reported", ["acme-face:7", "fakeface:3 ", "x"])
    def test_the_backend_reported_compat_passes_through_verbatim(self, monkeypatch, reported):
        _fake_backend(monkeypatch, [_Det((1, 2, 3, 4), None, 0.5)], compat=reported)
        pipe = ExtFacePipeline()
        pipe.initialize(RuntimeOptions(), _CONFIG)
        assert pipe.compatibility() == reported
        assert [r.compat for r in pipe.process(BLACK)] == [reported]

    @pytest.mark.parametrize("reported", [3, "", None])
    def test_a_compat_that_is_not_a_tag_is_refused(self, monkeypatch, reported):
        _fake_backend(monkeypatch, [], compat=reported)
        with pytest.raises(ValueError, match="non-empty string"):
            ExtFacePipeline().initialize(RuntimeOptions(), _CONFIG)

    def test_mixed_compat_tags_are_still_refused(self, monkeypatch):
        _fake_backend(monkeypatch, [_Det((1, 2, 3, 4), None, 0.5)], compat="acme-face:7")
        pipe = ExtFacePipeline()
        pipe.initialize(RuntimeOptions(), _CONFIG)
        meta = stamp_compat({}, "oss-magface-ir100:1")
        with pytest.raises(CompatibilityError, match="acme-face:7"):
            check_compat(meta, pipe.process(BLACK)[0].compat)

    def test_a_detection_without_a_box_is_refused(self, monkeypatch):
        _fake_backend(monkeypatch, [object()])
        pipe = ExtFacePipeline()
        pipe.initialize(RuntimeOptions(), _CONFIG)
        with pytest.raises(ValueError, match="no bbox"):
            pipe.process(BLACK)

    def test_config_without_an_embedder_config_initialises_positionally(self, monkeypatch):
        log = _fake_backend(monkeypatch, [])
        pipe = ExtFacePipeline()
        pipe.initialize(RuntimeOptions(), {"detector": {"model_bundle": "d"}, "embedder": {"model": "e"}})
        assert log["extractor"] == [(log["runtime"] and log["detector"][0][0], "e", None)]
        with pytest.raises(ValueError, match="required"):
            ExtFacePipeline().initialize(RuntimeOptions(), {"detector": {"model_bundle": "d"}})

    def test_process_buffer_decodes_with_our_decoder(self, monkeypatch):
        log = _fake_backend(monkeypatch, [])
        pipe = ExtFacePipeline()
        pipe.initialize(RuntimeOptions(), _CONFIG)
        ok, png = cv2.imencode(".png", np.zeros((8, 6, 3), np.uint8))
        assert ok
        assert pipe.process_buffer(png.tobytes()) == []
        assert log["detect"] == [((8, 6, 3), None)]
