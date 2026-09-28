"""Tests for the SyntheticDatasetImporter and its generators.

The generators need PIL (image, video) and imageio-ffmpeg (video). Tests
that only exercise audio generation work without those, so they are
separated out and the image/video tests skip cleanly when their deps are
missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtscore.datasets.importers import get_importer, list_importers
from vtscore.datasets.importers.synthetic import SyntheticDatasetImporter


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

    def test_importer_fields_are_media_type_size_and_seed(self):
        imp = get_importer("synthetic")
        keys = [f.key for f in imp.fields]
        assert keys == ["media_type", "size", "seed"]
        media_type_field = imp.fields[0]
        assert media_type_field.field_type == "select"
        assert set(media_type_field.options) == {"image", "audio", "video"}
        size_field = imp.fields[1]
        assert size_field.field_type == "number"
        assert size_field.min == "1"
        assert size_field.step == "1"
        seed_field = imp.fields[2]
        assert seed_field.field_type == "number"
        assert seed_field.default == "1"
        assert seed_field.min == "0"
        assert not seed_field.required


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

    def test_user_dataset_name_overrides_default(self):
        imp = SyntheticDatasetImporter()
        out = imp.resolve_display_name({"media_type": "audio", "size": "7", "dataset_name": "My Tones"})
        assert out == "My Tones"

    def test_display_name_names_a_non_default_seed(self):
        imp = SyntheticDatasetImporter()
        assert imp.default_display_name({"media_type": "image", "size": "5", "seed": "1"}) == "Synthetic image (5)"
        assert (
            imp.default_display_name({"media_type": "image", "size": "5", "seed": "2"}) == "Synthetic image (5, seed 2)"
        )

    @pytest.mark.parametrize(("raw", "seed"), [("", 1), (None, 1), ("7", 7), (" 3 ", 3), (0, 0), ("0", 0)])
    def test_parses_seed(self, raw, seed):
        assert SyntheticDatasetImporter._parse_seed({"seed": raw}) == seed

    def test_missing_seed_is_the_default(self):
        assert SyntheticDatasetImporter._parse_seed({}) == 1

    @pytest.mark.parametrize("raw", ["-1", "lots", "1.5"])
    def test_rejects_bad_seed(self, raw):
        with pytest.raises(ValueError):
            SyntheticDatasetImporter._parse_seed({"seed": raw})


class TestOriginRoundTrip:
    def test_build_and_reload_origin(self):
        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "image", "size": "10", "seed": "4"})
        assert origin["importer"] == "synthetic"
        assert origin["params"]["media_type"] == "image"
        assert origin["params"]["size"] == "10"
        assert origin["params"]["seed"] == "4"
        assert imp.can_reload_from_origin(origin)
        assert imp.reload_from_origin(origin) == {"media_type": "image", "size": "10", "seed": "4"}

    def test_origin_without_a_seed_reloads_the_default_seed(self):
        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "image", "size": "10"})
        assert imp.reload_from_origin(origin) == {"media_type": "image", "size": "10", "seed": "1"}

    def test_cannot_reload_with_bad_params(self):
        imp = SyntheticDatasetImporter()
        assert not imp.can_reload_from_origin({"importer": "synthetic", "params": {}})
        assert imp.reload_from_origin({"importer": "synthetic", "params": {}}) is None


# ---------------------------------------------------------------------------
# Audio generator (no extra deps required beyond numpy)
# ---------------------------------------------------------------------------


class TestAudioGenerator:
    def test_generates_requested_count(self, tmp_path):
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        paths = generate_audio_dataset(tmp_path, 12, seed=1)
        assert len(paths) == 12
        for p in paths:
            assert p.exists()
            assert p.suffix == ".wav"
            assert p.stat().st_size > 100  # something was actually written

    def test_cycles_through_all_ideas(self, tmp_path):
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        paths = generate_audio_dataset(tmp_path, 12, seed=1)
        prefixes = {p.name.split("_")[0] for p in paths}
        # Six ideas: tone, chord, drum, rain, wind, bird.
        assert prefixes == {"tone", "chord", "drum", "rain", "wind", "bird"}

    def test_caches_existing_files(self, tmp_path):
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        paths = generate_audio_dataset(tmp_path, 6, seed=1)
        first_mtime = paths[0].stat().st_mtime_ns
        # Second call must not rewrite the cached files.
        generate_audio_dataset(tmp_path, 6, seed=1)
        assert paths[0].stat().st_mtime_ns == first_mtime

    def test_emits_per_file_progress(self, tmp_path):
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

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
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

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
        from vtscore.utils.synthetic.images import generate_image_dataset  # noqa: PLC0415

        paths = generate_image_dataset(tmp_path, 8, seed=1)
        assert len(paths) == 8
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 100

    def test_includes_every_kind(self, tmp_path):
        from vtscore.utils.synthetic.images import generate_image_dataset  # noqa: PLC0415

        paths = generate_image_dataset(tmp_path, 5, seed=1)
        prefixes = {p.name.split("_")[0] for p in paths}
        assert prefixes == {"face", "shapes", "scene"}

    def test_pictures_are_canvas_sized(self, tmp_path):
        from PIL import Image  # noqa: PLC0415

        from vtscore.utils.synthetic.images import CANVAS_SIZE, generate_image_dataset  # noqa: PLC0415

        for path in generate_image_dataset(tmp_path, 3, seed=1):
            with Image.open(path) as img:
                assert img.size == (CANVAS_SIZE, CANVAS_SIZE)
                assert img.mode == "RGB"

    def test_same_seed_produces_same_bytes(self, tmp_path):
        from vtscore.utils.synthetic.images import generate_image_dataset  # noqa: PLC0415

        a = generate_image_dataset(tmp_path / "a", 5, seed=3)
        b = generate_image_dataset(tmp_path / "b", 5, seed=3)
        assert [p.read_bytes() for p in a] == [p.read_bytes() for p in b]

    def test_two_seeds_share_no_picture(self, tmp_path):
        """Seeding on ``seed + index`` made picture 1 of seed 1 picture 0 of seed 2."""
        from vtscore.utils.synthetic.images import generate_image_dataset  # noqa: PLC0415

        a = {p.read_bytes() for p in generate_image_dataset(tmp_path / "a", 10, seed=1)}
        b = {p.read_bytes() for p in generate_image_dataset(tmp_path / "b", 10, seed=2)}
        assert not a & b


class TestDescribeImageDataset:
    """``describe_image_dataset`` is the ground truth of what was drawn."""

    def test_names_the_files_generate_writes(self, tmp_path):
        pytest.importorskip("PIL")
        from vtscore.utils.synthetic.images import describe_image_dataset, generate_image_dataset  # noqa: PLC0415

        paths = generate_image_dataset(tmp_path, 12, seed=5)
        assert [p.name for p in paths] == [d["filename"] for d in describe_image_dataset(12, seed=5)]

    def test_is_deterministic(self):
        from vtscore.utils.synthetic.images import describe_image_dataset  # noqa: PLC0415

        assert describe_image_dataset(20, seed=9) == describe_image_dataset(20, seed=9)

    def test_is_json_serialisable(self):
        import json  # noqa: PLC0415

        from vtscore.utils.synthetic.images import describe_image_dataset  # noqa: PLC0415

        plans = describe_image_dataset(20, seed=2)
        assert json.loads(json.dumps(plans)) == plans

    def test_stable_keys(self):
        from vtscore.utils.synthetic.images import SMILING_EXPRESSIONS, describe_image_dataset  # noqa: PLC0415

        shapes = {"face", "circle", "square", "triangle", "star"}
        styles = {"plain", "dots", "stripes", "checks", "gradient"}
        for plan in describe_image_dataset(60, seed=4):
            assert plan["kind"] in {"face", "shapes", "scene"}
            assert plan["filename"].startswith(plan["kind"] + "_")
            assert plan["background"]["style"] in styles
            assert len(plan["background"]["colors"]) == (1 if plan["background"]["style"] == "plain" else 2)
            assert plan["objects"]
            for obj in plan["objects"]:
                assert obj["shape"] in shapes
                assert isinstance(obj["color"], str)
                x0, y0, x1, y1 = obj["box"]
                assert 0 <= x0 < x1 <= 1
                assert 0 <= y0 < y1 <= 1
                if obj["shape"] == "face":
                    assert obj["smiling"] == (obj["expression"] in SMILING_EXPRESSIONS)

    def test_a_face_picture_holds_one_face_not_the_colour_of_its_background(self):
        from vtscore.utils.synthetic.images import describe_image_dataset  # noqa: PLC0415

        faces = [p for p in describe_image_dataset(100, seed=6) if p["kind"] == "face"]
        assert faces
        for plan in faces:
            (face,) = plan["objects"]
            assert face["shape"] == "face"
            assert plan["background"]["colors"][0] != face["color"]

    def test_mix_holds_the_near_misses(self):
        """Enough yellow smileys to hunt, and the pictures that make it hard."""
        from vtscore.utils.synthetic.images import describe_image_dataset  # noqa: PLC0415

        faces = [p["objects"][0] for p in describe_image_dataset(240, seed=1) if p["kind"] == "face"]
        yellow_smiling = [f for f in faces if f["color"] == "yellow" and f["smiling"]]
        yellow_not = [f for f in faces if f["color"] == "yellow" and not f["smiling"]]
        other_smiling = [f for f in faces if f["color"] != "yellow" and f["smiling"]]
        assert len(yellow_smiling) >= 20
        assert len(yellow_not) >= 15
        assert len(other_smiling) >= 20

    def test_scene_objects_do_not_overlap(self):
        import math  # noqa: PLC0415

        from vtscore.utils.synthetic.images import describe_image_dataset  # noqa: PLC0415

        for plan in describe_image_dataset(50, seed=8):
            objs = plan["objects"]
            for i, a in enumerate(objs):
                for b in objs[i + 1 :]:
                    gap = math.dist(a["center"], b["center"]) - a["radius"] - b["radius"]
                    assert gap > 0


# ---------------------------------------------------------------------------
# Video generator (needs Pillow + imageio-ffmpeg)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_has_module("PIL") and _has_module("imageio_ffmpeg")),
    reason="Pillow and imageio-ffmpeg required",
)
class TestVideoGenerator:
    def test_generates_mp4s(self, tmp_path):
        from vtscore.utils.synthetic.video import generate_video_dataset  # noqa: PLC0415

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
        from vtscore.datasets.importers import synthetic as syn  # noqa: PLC0415

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)

        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        cache_dir = tmp_path / "synthetic" / "audio_3_seed1"
        generate_audio_dataset(cache_dir, 3, seed=1)

        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "audio", "size": "3", "seed": "1"})
        # The first generated file is "tone_0000.wav".
        resolved = imp.resolve_file(origin, origin_name="tone_0000.wav", filename="tone_0000.wav")
        assert resolved is not None
        assert resolved.name == "tone_0000.wav"

    def test_image_cache_folder_names_seed_and_generator_version(self, tmp_path, monkeypatch):
        """A cache the old image generator wrote must never be served as the new one's."""
        pytest.importorskip("PIL")
        from vtscore.datasets.importers import synthetic as syn  # noqa: PLC0415

        monkeypatch.setattr(syn, "DATA_DIR", tmp_path)
        imp = SyntheticDatasetImporter()
        one = imp._generate("image", 2, 1)
        two = imp._generate("image", 2, 2)
        assert one == tmp_path / "synthetic" / "image_2_seed1_v2"
        assert two == tmp_path / "synthetic" / "image_2_seed2_v2"
        assert {p.read_bytes() for p in one.iterdir()}.isdisjoint(p.read_bytes() for p in two.iterdir())

    def test_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        from vtscore.datasets.importers import synthetic as syn  # noqa: PLC0415

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
        from vtscore.datasets.importers import synthetic as syn  # noqa: PLC0415

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
            assert m["media_type"] == "audio"
            assert isinstance(m["filename"], str)
            assert Path(m["filename"]).suffix == ".wav"

    def test_importer_forwards_thread_progress_to_generator(self, tmp_path, monkeypatch):
        """The importer must hand the per-thread progress callback to the
        generator so the loading-task progress bar updates while files are
        being synthesised, instead of stalling on "Preparing new dataset…".
        """
        from vtscore.datasets.importers import synthetic as syn  # noqa: PLC0415
        from vtscore.concurrency.progress import (  # noqa: PLC0415
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
        from vtscore.datasets.importers import synthetic as syn  # noqa: PLC0415

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
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a_paths = generate_audio_dataset(a_dir, 3, seed=7)
        b_paths = generate_audio_dataset(b_dir, 3, seed=7)
        for ap, bp in zip(a_paths, b_paths):
            assert ap.read_bytes() == bp.read_bytes(), f"non-deterministic for {ap.name}"

    def test_different_seed_produces_different_audio_bytes(self, tmp_path):
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

        a_paths = generate_audio_dataset(tmp_path / "a", 3, seed=1)
        b_paths = generate_audio_dataset(tmp_path / "b", 3, seed=2)
        # At least one file should differ across seeds.
        diffs = sum(1 for ap, bp in zip(a_paths, b_paths) if ap.read_bytes() != bp.read_bytes())
        assert diffs >= 1


class TestPartialRegeneration:
    """Pre-existing files in the cache dir must be respected."""

    def test_generator_does_not_overwrite_existing_files(self, tmp_path):
        from vtscore.utils.synthetic.audio import generate_audio_dataset  # noqa: PLC0415

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
        args = imp.build_cli_args({"media_type": "audio", "size": "10", "seed": "3"})
        assert "--importer synthetic" in args
        assert "--media-type audio" in args
        assert "--size 10" in args
        assert "--seed 3" in args

    def test_origin_display(self):
        imp = SyntheticDatasetImporter()
        origin = imp.build_origin({"media_type": "image", "size": "25", "seed": "2"})
        assert imp.origin_display(origin) == "synthetic:image_25_seed2"
        assert imp.origin_display(imp.build_origin({"media_type": "image", "size": "25"})) == "synthetic:image_25_seed1"
