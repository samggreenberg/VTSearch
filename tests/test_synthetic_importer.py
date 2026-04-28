"""Tests for the SyntheticDatasetImporter and its generators.

The generators need PIL (image, video) and imageio-ffmpeg (video). Tests
that only exercise audio generation work without those, so they are
separated out and the image/video tests skip cleanly when their deps are
missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtsearch.datasets.importers import get_importer, list_importers
from vtsearch.datasets.importers.synthetic import SyntheticDatasetImporter


def _has_module(name: str) -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Importer registration / metadata
# ---------------------------------------------------------------------------


class TestSyntheticImporterRegistration:
    def test_importer_is_discovered(self):
        names = [imp.name for imp in list_importers()]
        assert "synthetic" in names

    def test_importer_metadata(self):
        imp = get_importer("synthetic")
        assert isinstance(imp, SyntheticDatasetImporter)
        assert imp.display_name
        assert imp.description
        assert imp.icon  # has an icon
        assert not getattr(imp, "hidden_from_picker", False)

    def test_importer_fields_are_just_media_type_and_size(self):
        imp = get_importer("synthetic")
        keys = [f.key for f in imp.fields]
        assert keys == ["media_type", "size"]
        media_type_field = imp.fields[0]
        assert media_type_field.field_type == "select"
        assert set(media_type_field.options) == {"image", "audio", "video"}


class TestSyntheticImporterValidation:
    def test_rejects_unknown_media_type(self, tmp_path):
        imp = SyntheticDatasetImporter()
        with pytest.raises(ValueError):
            imp.run({"media_type": "text", "size": "5"}, {})

    def test_rejects_non_positive_size(self, tmp_path):
        imp = SyntheticDatasetImporter()
        with pytest.raises(ValueError):
            imp.run({"media_type": "audio", "size": "0"}, {})

    def test_rejects_non_numeric_size(self):
        imp = SyntheticDatasetImporter()
        with pytest.raises(ValueError):
            imp.run({"media_type": "audio", "size": "lots"}, {})

    def test_resolve_display_name_includes_size(self):
        imp = SyntheticDatasetImporter()
        assert "audio" in imp.resolve_display_name({"media_type": "audio", "size": "7"})
        assert "7" in imp.resolve_display_name({"media_type": "audio", "size": "7"})


class TestOriginRoundTrip:
    def test_build_and_reload_origin(self):
        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "image", "size": "10"})
        assert origin["importer"] == "synthetic"
        assert origin["params"]["media_type"] == "image"
        assert origin["params"]["size"] == "10"
        assert imp.can_reload_from_origin(origin)
        assert imp.reload_from_origin(origin) == {"media_type": "image", "size": "10"}

    def test_cannot_reload_with_bad_params(self):
        imp = SyntheticDatasetImporter()
        assert not imp.can_reload_from_origin({"importer": "synthetic", "params": {}})
        assert imp.reload_from_origin({"importer": "synthetic", "params": {}}) is None


# ---------------------------------------------------------------------------
# Audio generator (no extra deps required beyond numpy)
# ---------------------------------------------------------------------------


class TestAudioGenerator:
    def test_generates_requested_count(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        paths = generate_audio_dataset(tmp_path, 12, seed=1)
        assert len(paths) == 12
        for p in paths:
            assert p.exists()
            assert p.suffix == ".wav"
            assert p.stat().st_size > 100  # something was actually written

    def test_cycles_through_all_ideas(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        paths = generate_audio_dataset(tmp_path, 12, seed=1)
        prefixes = {p.name.split("_")[0] for p in paths}
        # Six ideas: tone, chord, drum, rain, wind, bird.
        assert prefixes == {"tone", "chord", "drum", "rain", "wind", "bird"}

    def test_caches_existing_files(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        paths = generate_audio_dataset(tmp_path, 6, seed=1)
        first_mtime = paths[0].stat().st_mtime_ns
        # Second call must not rewrite the cached files.
        generate_audio_dataset(tmp_path, 6, seed=1)
        assert paths[0].stat().st_mtime_ns == first_mtime


# ---------------------------------------------------------------------------
# Image generator (needs Pillow)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_module("PIL"), reason="Pillow not installed")
class TestImageGenerator:
    def test_generates_pngs(self, tmp_path):
        from vtsearch.utils.synthetic.images import generate_image_dataset  # noqa: PLC0415

        paths = generate_image_dataset(tmp_path, 8, seed=1)
        assert len(paths) == 8
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 100

    def test_includes_both_smiley_and_shapes(self, tmp_path):
        from vtsearch.utils.synthetic.images import generate_image_dataset  # noqa: PLC0415

        paths = generate_image_dataset(tmp_path, 4, seed=1)
        prefixes = {p.name.split("_")[0] for p in paths}
        assert "smiley" in prefixes
        assert "shapes" in prefixes


# ---------------------------------------------------------------------------
# Video generator (needs Pillow + imageio-ffmpeg)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_has_module("PIL") and _has_module("imageio_ffmpeg")),
    reason="Pillow and imageio-ffmpeg required",
)
class TestVideoGenerator:
    def test_generates_mp4s(self, tmp_path):
        from vtsearch.utils.synthetic.video import generate_video_dataset  # noqa: PLC0415

        paths = generate_video_dataset(tmp_path, 4, seed=1)
        assert len(paths) == 4
        for p in paths:
            assert p.exists()
            assert p.suffix == ".mp4"
            assert p.stat().st_size > 100


# ---------------------------------------------------------------------------
# resolve_file integration
# ---------------------------------------------------------------------------


class TestResolveFile:
    def test_resolves_known_file_under_cache_dir(self, tmp_path, monkeypatch):
        # Redirect DATA_DIR to tmp so we don't pollute the real one.
        from vtsearch.datasets.importers import synthetic as syn  # noqa: PLC0415

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)

        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        cache_dir = tmp_path / "synthetic" / "audio_3"
        generate_audio_dataset(cache_dir, 3, seed=42)

        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "audio", "size": "3"})
        # The first generated file is "tone_0000.wav".
        resolved = imp.resolve_file(origin, origin_name="tone_0000.wav", filename="tone_0000.wav")
        assert resolved is not None
        assert resolved.name == "tone_0000.wav"

    def test_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        from vtsearch.datasets.importers import synthetic as syn  # noqa: PLC0415

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)
        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "audio", "size": "3"})
        assert imp.resolve_file(origin, "nope.wav", "nope.wav") is None


# ---------------------------------------------------------------------------
# End-to-end: run() through to medias
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_module("librosa"), reason="librosa needed by audio loader")
class TestRunEndToEnd:
    def test_audio_run_populates_medias(self, tmp_path, monkeypatch):
        from vtsearch.datasets.importers import synthetic as syn  # noqa: PLC0415

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)

        imp = SyntheticDatasetImporter()
        medias: dict[int, dict] = {}
        imp.run({"media_type": "audio", "size": "6"}, medias)
        assert len(medias) == 6
        # All medias should carry the synthetic origin.
        for m in medias.values():
            assert m["origin"]["importer"] == "synthetic"
            assert m["origin"]["params"]["media_type"] == "audio"
            assert m["origin"]["params"]["size"] == "6"
            assert m["type"] == "audio"
            assert isinstance(m["filename"], str)
            assert Path(m["filename"]).suffix == ".wav"
