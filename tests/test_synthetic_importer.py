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

    def test_icon_is_factory(self):
        """Importer should send the factory emoji so the frontend renders the
        line-drawing factory SVG (see frontend icon.component.ts)."""
        imp = get_importer("synthetic")
        assert imp.icon == "\U0001f3ed"  # 🏭

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

    def test_emits_per_file_progress(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        events: list[tuple[str, str, int, int]] = []

        def on_progress(status, message, current, total):
            events.append((status, message, current, total))

        generate_audio_dataset(tmp_path, 4, seed=1, on_progress=on_progress)

        # Status is "downloading" so it maps to step 1 in the load pipeline.
        assert all(ev[0] == "downloading" for ev in events)
        # Total is reported correctly on every event.
        assert all(ev[3] == 4 for ev in events)
        # Per-file events count up 0..3 (with start at 0 and final at 4).
        per_file_currents = [ev[2] for ev in events[1:-1]]
        assert per_file_currents == [0, 1, 2, 3]
        # Final event marks completion.
        assert events[-1][2] == 4
        # Messages mention "Synthesising" the first time around.
        assert any("Synthesising" in ev[1] for ev in events)

    def test_progress_marks_cached_runs(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        generate_audio_dataset(tmp_path, 3, seed=1)

        events: list[str] = []
        generate_audio_dataset(
            tmp_path,
            3,
            seed=1,
            on_progress=lambda status, message, current, total: events.append(message),
        )
        # On the second run all files are cached, so messages should say so.
        assert any("Reusing cached" in m for m in events)


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

    def test_importer_forwards_thread_progress_to_generator(self, tmp_path, monkeypatch):
        """The importer must hand the per-thread progress callback to the
        generator so the loading-task progress bar updates while files are
        being synthesised, instead of stalling on "Preparing new dataset…".
        """
        from vtsearch.datasets.importers import synthetic as syn  # noqa: PLC0415
        from vtsearch.utils.progress import (  # noqa: PLC0415
            clear_thread_progress,
            set_thread_progress,
        )

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)
        events: list[tuple[str, int, int]] = []

        def cb(status, message="", current=0, total=0, **_kw):
            events.append((status, current, total))

        set_thread_progress(cb)
        try:
            imp = SyntheticDatasetImporter()
            imp.run({"media_type": "audio", "size": "4"}, {})
        finally:
            clear_thread_progress()

        per_file = [(s, c, t) for (s, c, t) in events if s == "downloading" and t == 4]
        # We expect a start event (current=0), 4 per-file events (0..3), and a final event (current=4).
        assert any(c == 0 for _, c, _ in per_file)
        assert any(c == 3 for _, c, _ in per_file)
        assert any(c == 4 for _, c, _ in per_file)

    def test_run_then_resolve_file_round_trip(self, tmp_path, monkeypatch):
        """Origin produced by run() must resolve back to a real file via resolve_file."""
        from vtsearch.datasets.importers import synthetic as syn  # noqa: PLC0415

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)
        imp = SyntheticDatasetImporter()
        medias: dict[int, dict] = {}
        imp.run({"media_type": "audio", "size": "3"}, medias)
        media = next(iter(medias.values()))
        resolved = imp.resolve_file(media["origin"], origin_name=media["origin_name"], filename=media["filename"])
        assert resolved is not None
        assert resolved.is_file()


# ---------------------------------------------------------------------------
# Determinism / variety
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_same_audio_bytes(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a_paths = generate_audio_dataset(a_dir, 3, seed=7)
        b_paths = generate_audio_dataset(b_dir, 3, seed=7)
        for ap, bp in zip(a_paths, b_paths):
            assert ap.read_bytes() == bp.read_bytes(), f"non-deterministic for {ap.name}"

    def test_different_seed_produces_different_audio_bytes(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        a_paths = generate_audio_dataset(tmp_path / "a", 3, seed=1)
        b_paths = generate_audio_dataset(tmp_path / "b", 3, seed=2)
        # At least one file should differ across seeds.
        diffs = sum(1 for ap, bp in zip(a_paths, b_paths) if ap.read_bytes() != bp.read_bytes())
        assert diffs >= 1


class TestPartialRegeneration:
    """Pre-existing files in the cache dir must be respected."""

    def test_generator_does_not_overwrite_existing_files(self, tmp_path):
        from vtsearch.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        # Create the file paths the generator would produce, but with sentinel
        # contents. The generator must not overwrite them.
        tmp_path.mkdir(parents=True, exist_ok=True)
        sentinel = b"DO_NOT_OVERWRITE"
        first = tmp_path / "tone_0000.wav"
        first.write_bytes(sentinel)
        generate_audio_dataset(tmp_path, 4, seed=99)
        assert first.read_bytes() == sentinel
        # Other files should still be generated.
        assert (tmp_path / "chord_0001.wav").stat().st_size > 100


# ---------------------------------------------------------------------------
# build_cli_args / origin_display
# ---------------------------------------------------------------------------


class TestCliAndDisplay:
    def test_build_cli_args(self):
        imp = SyntheticDatasetImporter()
        args = imp.build_cli_args({"media_type": "audio", "size": "10"})
        assert "--importer synthetic" in args
        assert "--media-type audio" in args
        assert "--size 10" in args

    def test_origin_display(self):
        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "image", "size": "25"})
        assert imp.origin_display(origin) == "synthetic:image_25"
