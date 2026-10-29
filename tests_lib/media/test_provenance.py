"""Tests for :mod:`vtscore.media.provenance`.

Covers the curated ``Source`` / ``Derived Via`` / ``Imported Via`` display
lines distilled from a media's ``origin.params`` recipe, and the denylist
that keeps machine-only recipe keys out of them.
"""

import json

from vtscore.media.provenance import (
    DERIVED_VIA_LABEL,
    IMPORTED_VIA_LABEL,
    SOURCE_LABEL,
    provenance_metadata,
)


def _converter_media(**param_overrides):
    """A converter-runner output media (video frame extracted from a video)."""
    params = {
        "converter": "video2image",
        "source_file": "movie.mp4",
        "source_path": "/data/videos/movie.mp4",
        "converter_param_n_clips": "2",
        "parent_importer": "server_files",
        "parent_paths_file": "/data/list.txt",
        "converter_out_index": "1",
        "converter_n_out": "2",
        "converter_content_hash": "6ae4b81699fb",
    }
    params.update(param_overrides)
    return {
        "media_type": "image",
        "filename": "movie.mp4→frame_1.png",
        "origin_name": "movie.mp4→frame_1.png",
        "media_path": "/data/videos/movie.mp4",
        "origin": {"importer": "converter", "params": params},
    }


class TestProvenanceMetadata:
    def test_plain_import_reports_only_the_importer(self):
        """A file imported directly is its own source; nothing to derive."""
        media = {
            "media_type": "audio",
            "filename": "song.wav",
            "origin": {"importer": "server_folder", "params": {"path": "/data/sounds"}},
        }
        meta = provenance_metadata(media)
        assert SOURCE_LABEL not in meta
        assert DERIVED_VIA_LABEL not in meta
        assert meta[IMPORTED_VIA_LABEL] == "Folder (path=/data/sounds)"

    def test_import_line_omits_machine_only_params(self):
        """Dataset-level knobs are kept, replay recipe keys are not."""
        media = {
            "media_type": "image",
            "origin": {
                "importer": "synthetic",
                "params": {"media_type": "images", "size": "60", "source_specs": "[]"},
            },
        }
        assert provenance_metadata(media)[IMPORTED_VIA_LABEL] == "Synthetic Media (size=60)"

    def test_import_line_omitted_when_origin_names_no_importer(self):
        media = {"media_type": "audio", "origin": {"params": {"path": "/data"}}}
        assert provenance_metadata(media) == {}

    def test_missing_origin_is_tolerated(self):
        assert provenance_metadata({"filename": "x.wav"}) == {}
        assert provenance_metadata({"origin": "not-a-dict"}) == {}
        assert provenance_metadata({"origin": {"params": "not-a-dict"}}) == {}

    def test_converter_output_records_source_path(self):
        meta = provenance_metadata(_converter_media())
        assert meta[SOURCE_LABEL] == "/data/videos/movie.mp4"

    def test_converter_output_falls_back_to_source_file(self):
        """Datasets imported before ``source_path`` existed still resolve."""
        media = _converter_media()
        del media["origin"]["params"]["source_path"]
        assert provenance_metadata(media)[SOURCE_LABEL] == "movie.mp4"

    def test_converter_derivation_names_the_converter_and_params(self):
        meta = provenance_metadata(_converter_media())
        assert meta[DERIVED_VIA_LABEL] == "Video → Images (n_clips=2)"

    def test_converter_output_reports_the_parent_importer_not_the_converter(self):
        """ "Imported via converter" would say nothing about where the corpus came from."""
        meta = provenance_metadata(_converter_media())
        assert meta[IMPORTED_VIA_LABEL] == "Manifest (paths_file=/data/list.txt)"

    def test_unregistered_converter_falls_back_to_raw_name(self):
        media = _converter_media(converter="ghost2image")
        media["origin"]["params"].pop("converter_param_n_clips")
        assert provenance_metadata(media)[DERIVED_VIA_LABEL] == "ghost2image"

    def test_clipper_chain_renders_every_step(self):
        trail = [
            {"kind": "converter", "name": "video2image", "params": {"n_clips": 2}},
            {"kind": "clipper", "name": "image_object", "params": {"threshold": 0.4}},
        ]
        media = {
            "media_type": "image",
            "origin_name": "/data/videos/movie.mp4",
            "origin": {
                "importer": "server_folder",
                "params": {"path": "/data/videos", "clipper_chain": json.dumps(trail)},
            },
        }
        meta = provenance_metadata(media)
        assert meta[SOURCE_LABEL] == "/data/videos/movie.mp4"
        assert meta[DERIVED_VIA_LABEL] == "Video → Images (n_clips=2) → Object (threshold=0.4)"

    def test_malformed_chain_falls_back_to_flat_keys(self):
        media = {
            "media_type": "audio",
            "origin_name": "/data/rec.wav",
            "origin": {
                "importer": "server_folder",
                "params": {"clipper_chain": "{not json", "clipper": "sound_tiling", "clipper_duration": "5"},
            },
        }
        meta = provenance_metadata(media)
        assert meta[SOURCE_LABEL] == "/data/rec.wav"
        assert meta[DERIVED_VIA_LABEL] == "Tiling (duration=5)"

    def test_default_clipper_stamp_is_not_a_derivation(self):
        """``*_default`` clippers are a no-op stamp, not a real derivation."""
        media = {
            "media_type": "audio",
            "origin_name": "/data/rec.wav",
            "origin": {
                "importer": "server_folder",
                "params": {"path": "/data", "clipper": "sound_default", "clipper_duration": "5"},
            },
        }
        meta = provenance_metadata(media)
        assert SOURCE_LABEL not in meta
        assert DERIVED_VIA_LABEL not in meta

    def test_clip_without_origin_name_falls_back_to_media_path(self):
        media = {
            "media_type": "audio",
            "media_path": "/data/rec.wav",
            "origin": {"importer": "server_folder", "params": {"clipper": "sound_tiling"}},
        }
        assert provenance_metadata(media)[SOURCE_LABEL] == "/data/rec.wav"

    def test_derivation_without_a_resolvable_source_still_reports_it(self):
        media = {
            "media_type": "audio",
            "origin": {"importer": "server_folder", "params": {"clipper": "sound_tiling"}},
        }
        meta = provenance_metadata(media)
        assert SOURCE_LABEL not in meta
        assert DERIVED_VIA_LABEL in meta


class TestRecipeKeysStayOutOfTheImportLine:
    """The replay recipe is machine-facing; it must not leak into the grid."""

    def test_converter_recipe_keys_are_not_listed(self):
        line = provenance_metadata(_converter_media())[IMPORTED_VIA_LABEL]
        for hidden in (
            "converter=",
            "source_file=",
            "source_path=",
            "n_clips=",
            "converter_out_index=",
            "converter_n_out=",
            "converter_content_hash=",
        ):
            assert hidden not in line

    def test_clipper_chain_and_clip_extents_are_not_listed(self):
        media = {
            "origin": {
                "importer": "server_folder",
                "params": {
                    "path": "/data",
                    "clipper_chain": "[]",
                    "clipper": "sound_tiling",
                    "clipper_duration": "5",
                    "clip_start": 1.0,
                    "clip_end": 6.0,
                    "clip_index": "0",
                    "clip_box": [0, 0, 1, 1],
                    "source_specs": "[]",
                    "media_type": "audio",
                },
            }
        }
        assert provenance_metadata(media)[IMPORTED_VIA_LABEL] == "Folder (path=/data)"

    def test_every_line_is_a_string(self):
        """The grid renders values verbatim; a nested dict/list would leak JSON."""
        for value in provenance_metadata(_converter_media()).values():
            assert isinstance(value, str)
