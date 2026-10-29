"""Face is on exactly when this server has face models on disk (#5320).

:func:`vtscore.media.face.imagekit_bridge.face_models` finds a pipeline config
(``IMAGEKIT_FACE_CONFIG``, else ``imagekit/face.json`` in the models folder) and
checks, without loading anything, that its provider exists and its model files
are there.  The ``image2face`` converter hides itself while it is off, and the
pipeline refuses to build, naming what is missing.  Every test starts with no
face models (``tests_shared.state_reset.no_face_models``), whatever this
machine has.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from imagekit.face.pipeline import register_provider
from vtscore.converters.image2face import CONVERTER
from vtscore.media.face import imagekit_bridge as bridge
from vtscore.media.face.imagekit_bridge import default_config_path as real_default_config_path


def _models(folder: Path, **config) -> Path:
    """A pipeline config at ``folder/face.json`` naming two model files beside it, which exist."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "det.onnx").write_bytes(b"")
    (folder / "emb.onnx").write_bytes(b"")
    body = {"detector": {"model_bundle": "det.onnx"}, "embedder": {"model": "emb.onnx"}, **config}
    path = folder / "face.json"
    path.write_text(json.dumps(body))
    return path


@pytest.fixture
def models_folder(tmp_path, monkeypatch) -> Path:
    """Point the models folder's face config at a tmp dir, empty until a test fills it."""
    path = tmp_path / "models" / bridge.DEFAULT_CONFIG_RELPATH
    monkeypatch.setattr(bridge, "default_config_path", lambda: path)
    return path


class TestWhereTheConfigComesFrom:
    def test_the_default_is_imagekit_face_json_in_the_models_folder(self, monkeypatch, tmp_path):
        from vtscore.config import paths

        monkeypatch.setattr(paths, "MODELS_CACHE_DIR", tmp_path)
        assert real_default_config_path() == tmp_path / "imagekit" / "face.json"

    def test_nothing_anywhere_is_off_and_says_where_models_go(self, models_folder):
        status = bridge.face_models()
        assert status.available is False and status.config_path is None
        assert str(models_folder) in status.reason and bridge.CONFIG_ENV_VAR in status.reason

    def test_a_config_dropped_in_the_models_folder_turns_face_on(self, models_folder):
        _models(models_folder.parent)
        status = bridge.face_models()
        assert status == bridge.FaceModels(True, "", models_folder, "oss")

    def test_the_environment_variable_wins_over_the_folder(self, models_folder, tmp_path, monkeypatch):
        _models(models_folder.parent)
        named = _models(tmp_path / "elsewhere")
        monkeypatch.setenv(bridge.CONFIG_ENV_VAR, str(named))
        assert bridge.face_models().config_path == named

    def test_a_named_config_that_is_not_there_is_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv(bridge.CONFIG_ENV_VAR, str(tmp_path / "nope.json"))
        status = bridge.face_models()
        assert status.available is False and "nope.json" in status.reason


class TestWhatMakesItOff:
    def test_a_missing_model_file_is_named(self, models_folder):
        _models(models_folder.parent)
        (models_folder.parent / "emb.onnx").unlink()
        status = bridge.face_models()
        assert status.available is False and "embedder model" in status.reason and "emb.onnx" in status.reason

    def test_a_config_naming_no_model_is_off(self, models_folder):
        _models(models_folder.parent, embedder={})
        assert "embedder model (the config names none)" in bridge.face_models().reason

    def test_an_unreadable_config_is_off(self, models_folder):
        models_folder.parent.mkdir(parents=True)
        models_folder.write_text("{not json")
        status = bridge.face_models()
        assert status.available is False and "not a readable pipeline config" in status.reason

    def test_an_unknown_provider_is_off(self, models_folder):
        _models(models_folder.parent, provider="nobody")
        assert "'nobody'" in bridge.face_models().reason

    def test_the_ext_provider_needs_its_backend_module(self, models_folder):
        _models(models_folder.parent, provider="ext")
        status = bridge.face_models()
        assert status.available is False and "'ext_backend'" in status.reason and status.provider == "ext"

    def test_the_ext_backend_module_is_configurable(self, models_folder, monkeypatch):
        _models(models_folder.parent, provider="ext", ext_backend="from_the_config.backend")
        assert "'from_the_config.backend'" in bridge.face_models().reason  # its parent package is missing too
        monkeypatch.setenv("IMAGEKIT_EXT_BACKEND", "tests_fake_ext_backend")
        monkeypatch.setitem(sys.modules, "tests_fake_ext_backend", types.ModuleType("tests_fake_ext_backend"))
        assert bridge.face_models() == bridge.FaceModels(True, "", models_folder, "ext")

    def test_the_environment_picks_the_provider_over_the_config(self, models_folder, monkeypatch):
        _models(models_folder.parent, provider="ext")
        monkeypatch.setenv("IMAGEKIT_FACE_PROVIDER", "oss")
        assert bridge.face_models() == bridge.FaceModels(True, "", models_folder, "oss")

    def test_a_server_that_cannot_import_imagekit_is_off_not_broken(self, models_folder, monkeypatch):
        """The slim LabBench images install no ``cv2``, which ``imagekit`` imports at the top; server
        startup asks this, so raising there stopped every LabBench container booting (#5339)."""
        _models(models_folder.parent)
        monkeypatch.setitem(sys.modules, "imagekit.face.providers.ext", None)
        status = bridge.face_models()
        assert status.available is False and status.config_path is None
        assert "not installed here" in status.reason


class TestTheAnswerIsLive:
    def test_models_put_in_place_later_need_no_restart(self, models_folder, monkeypatch):
        monkeypatch.setattr(bridge, "_STATUS_TTL_S", 0.0)
        assert bridge.face_models().available is False
        _models(models_folder.parent)
        assert bridge.face_models().available is True

    def test_the_answer_is_reused_within_its_ttl(self, models_folder, monkeypatch):
        monkeypatch.setattr(bridge, "_STATUS_TTL_S", 3600.0)
        assert bridge.face_models().available is False
        _models(models_folder.parent)
        assert bridge.face_models().available is False


class TestTheConverterFollowsIt:
    def test_hidden_while_off_and_shown_while_on(self, models_folder, monkeypatch):
        monkeypatch.setattr(bridge, "_STATUS_TTL_S", 0.0)
        assert CONVERTER.hidden_from_picker is True
        _models(models_folder.parent)
        assert CONVERTER.hidden_from_picker is False

    def test_the_importer_listing_offers_no_face_output_while_off(self, models_folder, monkeypatch):
        from vtscore.datasets.importers import list_importers

        monkeypatch.setattr(bridge, "_STATUS_TTL_S", 0.0)
        importer = next(imp for imp in list_importers() if "available_converters_by_media_type" in imp.to_dict())
        assert "face" not in importer.to_dict()["available_converters_by_media_type"]
        _models(models_folder.parent)
        assert [c["name"] for c in importer.to_dict()["available_converters_by_media_type"]["face"]] == ["image2face"]


class TestThePipeline:
    def test_it_refuses_to_build_without_models_naming_why(self, models_folder):
        with pytest.raises(bridge.FaceProviderUnconfigured, match="no face models"):
            bridge.get_face_pipeline()

    def test_it_builds_the_provider_the_config_selects(self, models_folder):
        built: list = []

        class Fake:
            def initialize(self, opts, config):
                built.append(config)

            def process(self, image, *, confidence_threshold=None, media_id=""):
                return []

            def process_buffer(self, buffer, *, confidence_threshold=None, media_id=""):
                return []

            def compatibility(self):
                return "fake:7"

        register_provider("fake_5320", Fake)
        _models(models_folder.parent, provider="fake_5320")
        pipeline = bridge.get_face_pipeline()
        assert isinstance(pipeline, Fake) and bridge.get_face_pipeline() is pipeline
        assert built[0]["detector"]["model_bundle"] == str(models_folder.parent / "det.onnx")
        assert bridge.loaded_compat() == "fake:7" and bridge.provider_name() == "fake_5320"


class TestHardView:
    """A face under its space's Face Quality cut is a hard view (#5372); elsewhere there is no verdict."""

    def test_the_cut_is_per_space(self):
        assert bridge.is_hard_view(13.9, "oss-arcface-mbf:1") is True
        assert bridge.is_hard_view(14.0, "oss-arcface-mbf:1") is False
        assert bridge.is_hard_view(13.9, "oss-arcface-r50:1") is None
        assert bridge.is_hard_view(None, "oss-arcface-mbf:1") is None
        assert bridge.is_hard_view(True, "oss-arcface-mbf:1") is None

    def test_a_loaded_face_reads_what_its_converter_stamped(self):
        meta = {bridge.QUALITY_METADATA_KEY: 11.0, bridge.COMPAT_METADATA_KEY: "oss-arcface-mbf:1"}
        assert bridge.media_hard_view({"custom_metadata": meta}) is True
        assert bridge.media_hard_view({}) is None

    def test_a_hard_face_weighs_less_only_beside_a_clear_one(self, monkeypatch):
        """#5389: the example centroid leans on the clear faces; with none clear, every face weighs the same."""
        monkeypatch.setattr(bridge, "HARD_VIEW_EXAMPLE_WEIGHT", 0.25)
        assert bridge.example_weights([True, False, None, False]) == [0.25, 1.0, 1.0, 1.0]
        assert bridge.example_weights([True, True]) is None  # nothing clear to lean on
        assert bridge.example_weights([True, None]) is None  # no verdict is not clear
        assert bridge.example_weights([False, None]) is None  # nothing hard
        assert bridge.example_weights([]) is None
        monkeypatch.setattr(bridge, "HARD_VIEW_EXAMPLE_WEIGHT", 1.0)
        assert bridge.example_weights([True, False]) is None

    def test_a_picture_is_read_by_the_provider(self, models_folder, tmp_path, monkeypatch):
        from PIL import Image

        qualities = [12.0]

        class Fake:
            def initialize(self, opts, config):
                pass

            def process(self, image, *, confidence_threshold=None, media_id="") -> list[Any]:
                return [types.SimpleNamespace(quality=q) for q in qualities]

            def process_buffer(self, buffer, *, confidence_threshold=None, media_id="") -> list[Any]:
                return []

            def compatibility(self):
                return "fake:5372"

        register_provider("fake_5372", Fake)
        picture = tmp_path / "face.png"
        Image.new("RGB", (32, 32)).save(picture)
        assert bridge.hard_view_of_file(picture) is None  # Face is off
        _models(models_folder.parent, provider="fake_5372")
        bridge.reset_for_tests()  # face_models() keeps its "off" answer for a moment
        assert bridge.hard_view_of_file(picture) is None  # no cut for this space
        monkeypatch.setitem(bridge.HARD_VIEW_QUALITY, "fake:5372", 14.0)
        assert bridge.hard_view_of_file(picture) is True
        qualities[:] = [20.0]
        assert bridge.hard_view_of_file(picture) is False
        qualities[:] = []
        assert bridge.hard_view_of_file(picture) is None  # no face found
