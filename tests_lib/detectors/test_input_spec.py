"""Tests for the detector input_spec helpers.

Covers extraction from media origins, detector_meta assembly for labelset
export, and apply-on-import semantics.  The CLI / sync round-trip is
exercised separately in tests/cli and tests/io.
"""

from __future__ import annotations

# Plugin registries are populated eagerly at PluginRegistry construction
# (Phase 5 of ../../vtscore/docs/architecture.md), so no app-side import is
# needed to ensure converters/clippers are discoverable.
from vtscore.detectors.input_spec import (
    apply_detector_meta,
    build_detector_meta,
    clipper_matches,
    extract_input_spec_from_medias,
)


class TestExtractInputSpec:
    def test_empty_medias(self):
        assert extract_input_spec_from_medias({}) is None

    def test_no_clipper_in_origin(self):
        medias = {1: {"origin": {"importer": "server_folder", "params": {"path": "/tmp"}}}}
        assert extract_input_spec_from_medias(medias) is None

    def test_default_clipper_returns_none(self):
        """A pass-through *_default clipper has no meaningful input spec."""
        medias = {
            1: {
                "origin": {
                    "importer": "server_folder",
                    "params": {"clipper": "sound_default"},
                }
            }
        }
        assert extract_input_spec_from_medias(medias) is None

    def test_clipper_name_only(self):
        medias = {
            1: {
                "origin": {
                    "importer": "server_folder",
                    "params": {"clipper": "sound_tiling"},
                }
            }
        }
        spec = extract_input_spec_from_medias(medias)
        assert spec == {"clipper": "sound_tiling"}

    def test_clipper_with_params(self):
        medias = {
            1: {
                "origin": {
                    "importer": "server_folder",
                    "params": {
                        "clipper": "sound_tiling",
                        "clipper_duration": "2.0",
                        "clipper_min_overlap": "0.0",
                    },
                }
            }
        }
        spec = extract_input_spec_from_medias(medias)
        assert spec == {
            "clipper": "sound_tiling",
            "clipper_params": {"duration": "2.0", "min_overlap": "0.0"},
        }

    def test_ignores_per_clip_boundary_keys(self):
        """clipper_start / clipper_end / clipper_box / clipper_index are not params."""
        medias = {
            1: {
                "origin": {
                    "importer": "server_folder",
                    "params": {
                        "clipper": "sound_tiling",
                        "clipper_start": "0.0",
                        "clipper_end": "2.0",
                        "clipper_index": "0",
                        "clipper_duration": "2.0",
                    },
                }
            }
        }
        spec = extract_input_spec_from_medias(medias)
        assert spec == {
            "clipper": "sound_tiling",
            "clipper_params": {"duration": "2.0"},
        }

    def test_uses_first_origin_with_params(self):
        """Medias without origins are skipped until one is found."""
        medias = {
            1: {},
            2: {"origin": {}},
            3: {
                "origin": {
                    "importer": "server_folder",
                    "params": {"clipper": "sound_tiling"},
                }
            },
        }
        spec = extract_input_spec_from_medias(medias)
        assert spec == {"clipper": "sound_tiling"}


class TestBuildDetectorMeta:
    def test_minimal(self):
        meta = build_detector_meta({"media_type": "audio"})
        assert meta == {"media_type": "audio"}

    def test_with_input_spec(self):
        det = {
            "media_type": "audio",
            "input_spec": {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}},
        }
        meta = build_detector_meta(det)
        assert meta["media_type"] == "audio"
        assert meta["input_spec"] == det["input_spec"]
        # Defensive copy; mutating the meta dict must not affect the source.
        meta["input_spec"]["clipper"] = "changed"
        assert det["input_spec"]["clipper"] == "sound_tiling"

    def test_with_threshold(self):
        meta = build_detector_meta({"media_type": "audio"}, threshold=0.42)
        assert meta == {"media_type": "audio", "threshold": 0.42}

    def test_skips_empty_input_spec(self):
        meta = build_detector_meta({"media_type": "audio", "input_spec": {}})
        assert "input_spec" not in meta

    def test_threshold_none_omitted(self):
        meta = build_detector_meta({"media_type": "audio"}, threshold=None)
        assert "threshold" not in meta


class TestApplyDetectorMeta:
    def test_no_meta(self):
        data: dict = {"media_type": "audio"}
        assert apply_detector_meta(data, None) is False
        assert data == {"media_type": "audio"}

    def test_writes_input_spec(self):
        data: dict = {"media_type": "audio"}
        meta = {"input_spec": {"clipper": "sound_tiling"}}
        assert apply_detector_meta(data, meta) is True
        assert data["input_spec"] == {"clipper": "sound_tiling"}

    def test_does_not_persist_threshold(self):
        """threshold is informational; receiver retrains so we don't keep it."""
        data: dict = {"media_type": "audio"}
        meta = {"threshold": 0.7}
        assert apply_detector_meta(data, meta) is False
        assert "threshold" not in data

    def test_fills_in_missing_media_type(self):
        data: dict = {}
        meta = {"media_type": "audio", "input_spec": {"clipper": "sound_tiling"}}
        assert apply_detector_meta(data, meta) is True
        assert data["media_type"] == "audio"

    def test_does_not_overwrite_existing_media_type(self):
        data: dict = {"media_type": "image"}
        meta = {"media_type": "audio"}
        assert apply_detector_meta(data, meta) is False
        assert data["media_type"] == "image"

    def test_no_change_when_spec_matches(self):
        data: dict = {
            "media_type": "audio",
            "input_spec": {"clipper": "sound_tiling"},
        }
        meta = {"input_spec": {"clipper": "sound_tiling"}}
        assert apply_detector_meta(data, meta) is False


class TestClipperMatches:
    def test_detector_without_spec_matches_anything(self):
        assert clipper_matches(None, None) is True
        assert clipper_matches(None, {"clipper": "sound_tiling"}) is True

    def test_dataset_without_clipper_doesnt_match_clipped_detector(self):
        assert clipper_matches({"clipper": "sound_tiling"}, None) is False

    def test_names_must_match(self):
        det = {"clipper": "sound_tiling"}
        assert clipper_matches(det, {"clipper": "sound_tiling"}) is True
        assert clipper_matches(det, {"clipper": "image_tiling"}) is False

    def test_params_must_match(self):
        det = {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}}
        assert clipper_matches(det, {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}})
        assert not clipper_matches(det, {"clipper": "sound_tiling", "clipper_params": {"duration": "5.0"}})

    def test_string_int_param_equivalence(self):
        """load_pipeline stores values as strings; comparisons coerce."""
        det = {"clipper": "sound_tiling", "clipper_params": {"duration": "2"}}
        assert clipper_matches(det, {"clipper": "sound_tiling", "clipper_params": {"duration": 2}})
