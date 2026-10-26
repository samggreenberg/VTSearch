"""Tests for the detector input_spec helpers.

Covers extraction from media origins, detector_meta assembly for labelset
export, apply-on-import semantics, and resolving a clipper choice into a
spec (the CLI's clipper override, #5282).  The CLI / sync round-trip is
exercised separately in tests/cli and tests/io.
"""

from __future__ import annotations

import warnings

import pytest

# Plugin registries are populated eagerly at PluginRegistry construction
# (seam 5 of ../../vtscore/docs/architecture.md), so no app-side import is
# needed to ensure converters/clippers are discoverable.
from vtscore.detectors.input_spec import (
    apply_detector_meta,
    build_detector_meta,
    clipper_matches,
    extract_input_spec_from_medias,
    resolve_clipper_spec,
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

    def test_chain_trail_is_not_a_parameter(self):
        """``clipper_chain`` is the per-clip trail, not a ``chain`` parameter: it
        differs clip to clip, so reading it in kept every chain-clipped
        dataset from matching any spec."""
        medias = {
            1: {
                "origin": {
                    "params": {
                        "clipper": "sound_tiling",
                        "clipper_duration": "5.0",
                        "clipper_chain": '[{"kind":"clipper","name":"sound_tiling","clip_start":"0.0"}]',
                    }
                }
            }
        }
        assert extract_input_spec_from_medias(medias) == {
            "clipper": "sound_tiling",
            "clipper_params": {"duration": "5.0"},
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

    def test_a_threshold_is_deprecated_and_never_emitted(self):
        """#5131: a head's threshold means nothing to a receiver, so the block never carries one."""
        with pytest.warns(DeprecationWarning, match="threshold"):
            meta = build_detector_meta({"media_type": "audio"}, threshold=0.42)
        assert meta == {"media_type": "audio"}

    def test_skips_empty_input_spec(self):
        meta = build_detector_meta({"media_type": "audio", "input_spec": {}})
        assert "input_spec" not in meta

    def test_threshold_none_omitted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
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


class TestResolveClipperSpec:
    """``resolve_clipper_spec`` validates a clipper choice into an ``input_spec`` (#5282)."""

    def test_fills_in_the_effective_parameters_as_strings(self):
        spec = resolve_clipper_spec("sound_tiling", {"duration": "5"})
        assert spec == {"clipper": "sound_tiling", "clipper_params": {"duration": "5.0", "min_overlap": "1.0"}}

    def test_no_params_takes_the_defaults(self):
        spec = resolve_clipper_spec("sound_tiling")
        assert spec["clipper"] == "sound_tiling"
        assert set(spec["clipper_params"]) == {"duration", "min_overlap"}

    def test_a_clipper_with_no_parameters_has_no_params_key(self):
        assert resolve_clipper_spec("sound_default") == {"clipper": "sound_default"}

    def test_resolving_a_resolved_spec_gives_it_back(self):
        spec = resolve_clipper_spec("sound_tiling", {"duration": 5})
        assert resolve_clipper_spec(spec["clipper"], spec["clipper_params"]) == spec

    def test_unknown_clipper_lists_the_available_ones(self):
        with pytest.raises(ValueError, match=r"Unknown clipper 'nope'\. Available: .*sound_tiling"):
            resolve_clipper_spec("nope")

    def test_unknown_parameter_names_the_clippers_own(self):
        with pytest.raises(ValueError, match=r"no parameter 'duraton'\. Its parameters: duration, min_overlap"):
            resolve_clipper_spec("sound_tiling", {"duraton": "5"})

    def test_a_parameterless_clipper_refuses_any(self):
        with pytest.raises(ValueError, match="It takes none"):
            resolve_clipper_spec("sound_default", {"duration": "5"})

    @pytest.mark.parametrize("value", ["abc", "0"])
    def test_a_value_the_clipper_refuses_is_refused(self, value):
        with pytest.raises(ValueError, match="Clipper 'sound_tiling' cannot take"):
            resolve_clipper_spec("sound_tiling", {"duration": value})

    def test_matches_a_dataset_the_loader_clipped_with_the_same_settings(self):
        """The spec is what a clipped dataset's origins read back as, so the two compare equal."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips
        from vtscore.media.audio.audio_generator import generate_wav

        clips = {
            1: {
                "id": 1,
                "media_type": "audio",
                "media_bytes": generate_wav(440, 12.0),
                "origin": {"importer": "server_folder", "params": {}},
                "origin_name": "tone.wav",
            }
        }
        apply_chain_to_clips(clips, [{"kind": "clipper", "name": "sound_tiling", "params": {"duration": 5.0}}])
        dataset_spec = extract_input_spec_from_medias(clips)

        assert len(clips) > 1
        assert clipper_matches(resolve_clipper_spec("sound_tiling", {"duration": "5"}), dataset_spec)
        assert not clipper_matches(resolve_clipper_spec("sound_tiling", {"duration": "2"}), dataset_spec)
