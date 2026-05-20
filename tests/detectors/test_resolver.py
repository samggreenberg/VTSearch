"""Tests for vtscore.detectors.resolver — media file resolution from origin trails."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.detectors.resolver import (
    ResolvedLabels,
    resolve_file_from_origin,
    resolve_label_embeddings,
)


class TestResolvedLabels:
    def test_empty(self):
        r = ResolvedLabels()
        assert r.resolved_count == 0
        assert r.total_count == 0
        assert r.available_fraction == 0.0
        assert not r.has_good_and_bad

    def test_fraction(self):
        r = ResolvedLabels(resolved_count=3, total_count=5)
        assert r.available_fraction == pytest.approx(0.6)

    def test_has_good_and_bad(self):
        r = ResolvedLabels(labels=[1.0, 0.0])
        assert r.has_good_and_bad

    def test_only_good(self):
        r = ResolvedLabels(labels=[1.0, 1.0])
        assert not r.has_good_and_bad


class TestResolveFolderOrigin:
    def test_resolves_by_origin_name(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"fake_audio")

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="clip.wav")
        assert result == folder / "clip.wav"

    def test_resolves_by_filename_fallback(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"fake_audio")

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, filename="clip.wav")
        assert result == folder / "clip.wav"

    def test_resolves_nested_path(self, tmp_path):
        folder = tmp_path / "data"
        subfolder = folder / "category"
        subfolder.mkdir(parents=True)
        (subfolder / "item.wav").write_bytes(b"data")

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="category/item.wav")
        assert result == subfolder / "item.wav"

    def test_returns_none_for_missing_file(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="nonexistent.wav")
        assert result is None

    def test_returns_none_for_missing_folder(self):
        origin = {"importer": "server_folder", "params": {"path": "/nonexistent/path"}}
        result = resolve_file_from_origin(origin, origin_name="clip.wav")
        assert result is None

    def test_returns_none_for_empty_path(self):
        origin = {"importer": "server_folder", "params": {"path": ""}}
        result = resolve_file_from_origin(origin, origin_name="clip.wav")
        assert result is None


class TestResolvePdfOrigin:
    def test_resolves_existing_pdf(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        origin = {"importer": "pdf", "params": {"path": str(pdf)}}
        result = resolve_file_from_origin(origin)
        assert result == pdf

    def test_returns_none_for_missing_pdf(self):
        origin = {"importer": "pdf", "params": {"path": "/nonexistent/doc.pdf"}}
        result = resolve_file_from_origin(origin)
        assert result is None


class TestResolveDupeSetOrigin:
    def test_resolves_first_available_member(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "b.wav").write_bytes(b"audio_b")

        origin = {
            "importer": "dupe_set",
            "params": {"name": "a.wav"},
            "members": [
                {
                    "origin": {"importer": "server_folder", "params": {"path": "/gone"}},
                    "origin_name": "a.wav",
                    "filename": "a.wav",
                },
                {
                    "origin": {"importer": "server_folder", "params": {"path": str(folder)}},
                    "origin_name": "b.wav",
                    "filename": "b.wav",
                },
            ],
        }
        result = resolve_file_from_origin(origin)
        assert result == folder / "b.wav"

    def test_returns_none_when_no_members_resolve(self):
        origin = {
            "importer": "dupe_set",
            "params": {"name": "x.wav"},
            "members": [
                {
                    "origin": {"importer": "server_folder", "params": {"path": "/gone"}},
                    "origin_name": "x.wav",
                },
            ],
        }
        result = resolve_file_from_origin(origin)
        assert result is None


class TestResolveConverterOrigin:
    def test_resolves_folder_parent(self, tmp_path):
        folder = tmp_path / "videos"
        folder.mkdir()
        (folder / "clip.mp4").write_bytes(b"video")

        origin = {
            "importer": "converter",
            "params": {
                "converter": "video2image",
                "source_file": "clip.mp4",
                "parent_importer": "server_folder",
                "parent_path": str(folder),
            },
        }
        result = resolve_file_from_origin(origin)
        assert result == folder / "clip.mp4"


class TestResolveDemoOrigin:
    """Verify that resolve_file_from_origin handles demo dataset origins."""

    def test_resolves_demo_file(self, tmp_path):
        """Demo importer resolve_file finds files in the expected download dir."""
        from vtscore.datasets.importers.demo import DemoDatasetImporter

        # Create a fake download directory structure
        img_dir = tmp_path / "caltech-101" / "101_ObjectCategories"
        (img_dir / "kangaroo").mkdir(parents=True)
        target = img_dir / "kangaroo" / "image_0017.jpg"
        target.write_bytes(b"fake_image")

        importer = DemoDatasetImporter()
        origin = {"importer": "demo", "params": {"name": "caltech101_s"}}

        # Patch _SOURCE_DIRS to use our tmp_path
        import vtscore.datasets.importers.demo as demo_mod

        old = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"caltech101": img_dir}
        try:
            result = importer.resolve_file(origin, origin_name="kangaroo/image_0017.jpg")
            assert result == target
        finally:
            demo_mod._SOURCE_DIRS = old

    def test_returns_none_for_missing_file(self, tmp_path):
        """resolve_file returns None when the file doesn't exist on disk."""
        from vtscore.datasets.importers.demo import DemoDatasetImporter

        img_dir = tmp_path / "caltech-101" / "101_ObjectCategories"
        img_dir.mkdir(parents=True)

        importer = DemoDatasetImporter()
        origin = {"importer": "demo", "params": {"name": "caltech101_s"}}

        import vtscore.datasets.importers.demo as demo_mod

        old = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"caltech101": img_dir}
        try:
            result = importer.resolve_file(origin, origin_name="kangaroo/no_such_file.jpg")
            assert result is None
        finally:
            demo_mod._SOURCE_DIRS = old

    def test_returns_none_for_unknown_demo(self):
        """resolve_file returns None for an unrecognized demo dataset name."""
        from vtscore.datasets.importers.demo import DemoDatasetImporter

        importer = DemoDatasetImporter()
        origin = {"importer": "demo", "params": {"name": "nonexistent_dataset"}}
        assert importer.resolve_file(origin, origin_name="foo.jpg") is None

    def test_resolves_flat_dir_with_category_prefix(self, tmp_path):
        """ESC-50 style: origin_name has category prefix but dir is flat."""
        from vtscore.datasets.importers.demo import DemoDatasetImporter

        # Create a flat audio directory (like ESC-50-master/audio/)
        audio_dir = tmp_path / "ESC-50-master" / "audio"
        audio_dir.mkdir(parents=True)
        target = audio_dir / "1-100032-A-0.wav"
        target.write_bytes(b"fake_audio")

        importer = DemoDatasetImporter()
        origin = {"importer": "demo", "params": {"name": "esc50_s"}}

        import vtscore.datasets.importers.demo as demo_mod

        old = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"esc50": audio_dir}
        try:
            # origin_name includes category prefix, but file is flat
            result = importer.resolve_file(origin, origin_name="dog/1-100032-A-0.wav")
            assert result == target
        finally:
            demo_mod._SOURCE_DIRS = old

    def test_resolves_fold_dir_with_category_prefix(self, tmp_path):
        """UrbanSound8K style: origin_name has category prefix, files in fold subdirs."""
        from vtscore.datasets.importers.demo import DemoDatasetImporter

        # Create fold-based directory structure (like UrbanSound8K/audio/)
        audio_dir = tmp_path / "UrbanSound8K" / "audio"
        fold_dir = audio_dir / "fold3"
        fold_dir.mkdir(parents=True)
        target = fold_dir / "100032-3-0-0.wav"
        target.write_bytes(b"fake_audio")

        importer = DemoDatasetImporter()
        origin = {"importer": "demo", "params": {"name": "urbansound8k_a"}}

        import vtscore.datasets.importers.demo as demo_mod

        old = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"urbansound8k": audio_dir}
        try:
            # origin_name has category prefix, actual file in fold subdir
            result = importer.resolve_file(origin, origin_name="car_horn/100032-3-0-0.wav")
            assert result == target
        finally:
            demo_mod._SOURCE_DIRS = old

    def test_basename_fallback_skips_ambiguous(self, tmp_path):
        """When multiple files share the same basename, fallback returns None."""
        from vtscore.datasets.importers.demo import DemoDatasetImporter

        audio_dir = tmp_path / "audio"
        (audio_dir / "fold1").mkdir(parents=True)
        (audio_dir / "fold2").mkdir(parents=True)
        (audio_dir / "fold1" / "same.wav").write_bytes(b"a")
        (audio_dir / "fold2" / "same.wav").write_bytes(b"b")

        importer = DemoDatasetImporter()
        origin = {"importer": "demo", "params": {"name": "esc50_s"}}

        import vtscore.datasets.importers.demo as demo_mod

        old = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"esc50": audio_dir}
        try:
            result = importer.resolve_file(origin, origin_name="cat/same.wav")
            assert result is None
        finally:
            demo_mod._SOURCE_DIRS = old

    def test_dispatches_through_resolver(self, tmp_path):
        """resolve_file_from_origin dispatches to DemoDatasetImporter.resolve_file."""
        img_dir = tmp_path / "caltech-101" / "101_ObjectCategories"
        (img_dir / "elephant").mkdir(parents=True)
        target = img_dir / "elephant" / "image_0012.jpg"
        target.write_bytes(b"fake_image")

        import vtscore.datasets.importers.demo as demo_mod

        old = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"caltech101": img_dir}
        try:
            origin = {"importer": "demo", "params": {"name": "caltech101_s"}}
            result = resolve_file_from_origin(origin, origin_name="elephant/image_0012.jpg")
            assert result == target
        finally:
            demo_mod._SOURCE_DIRS = old


class TestAudioDemoDatasetSources:
    """Verify all audio demo datasets have source fields for proper resolution."""

    def test_esc50_datasets_have_source(self):
        from vtscore.datasets.config import DEMO_DATASETS

        for ds_id in ("esc50_s", "esc50_m", "esc50_l"):
            assert ds_id in DEMO_DATASETS, f"{ds_id} not in DEMO_DATASETS"
            assert "source" in DEMO_DATASETS[ds_id], f"{ds_id} missing 'source' field"
            assert DEMO_DATASETS[ds_id]["source"] == "esc50"

    def test_all_audio_demos_have_source(self):
        from vtscore.datasets.config import DEMO_DATASETS

        audio_demos = {k: v for k, v in DEMO_DATASETS.items() if v.get("media_type") == "audio"}
        for ds_id, info in audio_demos.items():
            assert "source" in info, (
                f"Audio demo dataset {ds_id!r} is missing a 'source' field — "
                f"resolve_file() won't be able to map it to a download directory"
            )


class TestResolveNoneOrigin:
    def test_none_origin(self):
        assert resolve_file_from_origin(None) is None

    def test_unknown_importer(self):
        origin = {"importer": "unknown_thing", "params": {}}
        assert resolve_file_from_origin(origin) is None

    def test_unknown_importer_with_path_fallback(self, tmp_path):
        """Unregistered origins with a path param still resolve via fallback."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        origin = {"importer": "some_future_type", "params": {"path": str(f)}}
        assert resolve_file_from_origin(origin) == f


class TestDynamicImporterDispatch:
    """Verify that resolve_file_from_origin dispatches through the importer
    registry, so adding a new DatasetImporter automatically extends resolution."""

    def test_custom_importer_resolve_file_is_called(self, tmp_path):
        """A custom importer registered at runtime has its resolve_file called."""
        from vtscore.datasets.importers.base import DatasetImporter, ImporterField

        marker_file = tmp_path / "custom_media.wav"
        marker_file.write_bytes(b"custom_audio")

        class CustomImporter(DatasetImporter):
            name = "test_custom"
            display_name = "Test Custom"
            description = "A test importer"
            fields = [ImporterField(key="path", label="Path", field_type="text")]

            def resolve_file(self, origin, origin_name="", filename="", **kw):
                p = origin.get("params", {}).get("path", "")
                if p:
                    for n in [origin_name, filename]:
                        if n:
                            candidate = Path(p) / n
                            if candidate.is_file():
                                return candidate
                return None

        custom = CustomImporter()

        # Temporarily register it in the importer registry
        from vtscore.datasets.importers import get_importer

        registry = get_importer.__self__
        registry._ensure_discovered()
        registry._items[custom.name] = custom
        try:
            origin = {"importer": "test_custom", "params": {"path": str(tmp_path)}}
            result = resolve_file_from_origin(origin, origin_name="custom_media.wav")
            assert result == marker_file
        finally:
            registry._items.pop(custom.name, None)


class TestResolveLabelEmbeddings:
    def test_resolves_folder_labels(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "good1.wav").write_bytes(b"good_audio")
        (folder / "bad1.wav").write_bytes(b"bad_audio")

        origin = {"importer": "server_folder", "params": {"path": str(folder), "media_type": "audio"}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "good1.wav", "md5": "aaa", "filename": "good1.wav"},
            {"label": "bad", "origin": origin, "origin_name": "bad1.wav", "md5": "bbb", "filename": "bad1.wav"},
        ]

        fake_emb = np.zeros(512, dtype=np.float32)
        with patch("vtscore.detectors.resolver.embed_file", return_value=fake_emb):
            result = resolve_label_embeddings(labels, "audio")

        assert result.resolved_count == 2
        assert result.total_count == 2
        assert result.has_good_and_bad
        assert len(result.embeddings) == 2
        assert result.labels == [1.0, 0.0]
        assert len(result.missing_entries) == 0

    def test_skips_missing_files(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "good1.wav").write_bytes(b"good_audio")

        origin = {"importer": "server_folder", "params": {"path": str(folder), "media_type": "audio"}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "good1.wav", "md5": "aaa", "filename": "good1.wav"},
            {"label": "bad", "origin": origin, "origin_name": "missing.wav", "md5": "bbb", "filename": "missing.wav"},
        ]

        fake_emb = np.zeros(512, dtype=np.float32)
        with patch("vtscore.detectors.resolver.embed_file", return_value=fake_emb):
            result = resolve_label_embeddings(labels, "audio")

        assert result.resolved_count == 1
        assert result.total_count == 2
        assert not result.has_good_and_bad
        assert len(result.missing_entries) == 1

    def test_skips_invalid_labels(self):
        labels = [
            {"label": "neutral", "origin": None, "origin_name": "x.wav"},
        ]
        result = resolve_label_embeddings(labels, "audio")
        assert result.total_count == 0
        assert result.resolved_count == 0

    def test_handles_embed_failure(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"audio")

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "clip.wav", "md5": "aaa", "filename": "clip.wav"},
        ]

        with patch("vtscore.detectors.resolver.embed_file", return_value=None):
            result = resolve_label_embeddings(labels, "audio")

        assert result.resolved_count == 0
        assert result.total_count == 1
        assert len(result.missing_entries) == 1

    def test_forwards_embedder_name_to_embed_file(self, tmp_path):
        """``embedder_name`` must reach ``embed_file`` so training vectors
        live in the same space as the snap embeddings they'll be mixed with.

        Mixing vectors from two embedders into a single MLP produces
        garbage (different output dimensions crash; same dim silently
        corrupts), so the dataset's embedder propagating through to the
        resolver is part of the H5-secondary fix.
        """
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"audio")

        origin = {"importer": "server_folder", "params": {"path": str(folder), "media_type": "audio"}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "clip.wav", "md5": "aaa", "filename": "clip.wav"},
        ]
        fake_emb = np.zeros(768, dtype=np.float32)
        with patch("vtscore.detectors.resolver.embed_file", return_value=fake_emb) as embed_mock:
            resolve_label_embeddings(labels, "audio", embedder_name="some-specific-embedder")

        # The resolver must forward the embedder name through to embed_file
        # so the resolved label vector matches the snap's space.
        embed_mock.assert_called()
        last_call_args = embed_mock.call_args_list[-1]
        # embed_file signature: (file_path, media_type, embedder_name)
        passed_args = list(last_call_args.args) + [last_call_args.kwargs.get("embedder_name")]
        assert "some-specific-embedder" in passed_args, (
            f"embed_file was called without the embedder_name: {last_call_args}"
        )

    def test_forwards_embedder_name_to_clip_path(self, tmp_path):
        """Clipper-bearing labels must also receive the requested embedder."""
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"audio")

        origin = {
            "importer": "server_folder",
            "params": {
                "path": str(folder),
                "media_type": "audio",
                "clipper": "audio_window",
                "clip_start": 0.0,
                "clip_end": 0.1,
            },
        }
        labels = [
            {"label": "bad", "origin": origin, "origin_name": "clip.wav", "md5": "bbb", "filename": "clip.wav"},
        ]
        fake_emb = np.zeros(768, dtype=np.float32)
        # ``_apply_clip_and_embed`` returns ``(embedding, clip_bytes)`` since
        # the H10 clip-aware ingest refactor; ``clip_bytes=None`` mirrors the
        # full-file fallback path.
        with patch("vtscore.detectors.resolver._apply_clip_and_embed", return_value=(fake_emb, None)) as clip_mock:
            resolve_label_embeddings(labels, "audio", embedder_name="specific-clip-embedder")

        clip_mock.assert_called()
        last_call_args = clip_mock.call_args_list[-1]
        passed_args = list(last_call_args.args) + list(last_call_args.kwargs.values())
        assert "specific-clip-embedder" in passed_args


class TestMultiFindCrossDatasetFallback:
    """Test that multi_find uses the resolver when labels don't match the target dataset."""

    def test_trainable_model_falls_back_to_resolver(self, client, tmp_path):
        """When a detector's labels don't match the target dataset,
        multi_find should fall back to resolving labels from their origins."""
        import json
        import pickle
        import time

        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.registry import register_detector

        # Create a folder with media files for the detector's labels
        label_folder = tmp_path / "label_audio"
        label_folder.mkdir()
        (label_folder / "good.wav").write_bytes(b"good_audio_content")
        (label_folder / "bad.wav").write_bytes(b"bad_audio_content")

        # Create a target dataset pkl with different medias (no overlap)
        target_medias = {}
        for i in range(5):
            emb = np.random.RandomState(i).randn(512).astype(np.float32)
            target_medias[i] = {
                "id": i,
                "type": "audio",
                "embedding": emb,
                "md5": f"target_md5_{i}",
                "filename": f"target_{i}.wav",
                "origin_name": f"target_{i}.wav",
                "origin": {"importer": "server_folder", "params": {"path": "/other/folder"}},
            }

        pkl_path = tmp_path / "target.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": target_medias}, f)

        # Register the target dataset
        ds = register_dataset(
            name="target_ds",
            media_type="audio",
            num_items=5,
            pkl_path=str(pkl_path),
        )

        # Create a detector with labels from label_folder
        label_origin = {"importer": "server_folder", "params": {"path": str(label_folder), "media_type": "audio"}}
        from vtscore.detectors.store import _detector_path, _write_detector

        tm_name = "Test Cross Detector"
        tm_path = _detector_path(tm_name)
        tm_data = {
            "name": "Test Cross Detector",
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {
                        "md5": "good_md5",
                        "label": "good",
                        "origin": label_origin,
                        "origin_name": "good.wav",
                        "filename": "good.wav",
                    },
                    {
                        "md5": "bad_md5",
                        "label": "bad",
                        "origin": label_origin,
                        "origin_name": "bad.wav",
                        "filename": "bad.wav",
                    },
                ]
            },
        }
        _write_detector(tm_path, tm_data)

        # Register the model
        model_entry = register_detector(
            name="Test Cross Detector",
            media_type="audio",
            num_training=2,
        )

        # Mock the embedder to return deterministic vectors
        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)
        bad_emb = np.random.RandomState(200).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            name = Path(path).name
            if "good" in name:
                return good_emb
            return bad_emb

        with patch("vtscore.detectors.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find",
                data=json.dumps(
                    {
                        "dataset_ids": [ds["id"]],
                        "detector_ids": [model_entry["id"]],
                    }
                ),
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = resp.get_json()
        # The resolver should have kicked in — results should not all be N/A
        assert "results" in data
        # Every media should have a verdict from the model
        # (either Good or Bad, not N/A since resolver found the files)
        for r in data["results"]:
            verdicts = r["detector_verdicts"]
            assert "Test Cross Detector" in verdicts
            assert verdicts["Test Cross Detector"]["verdict"] in ("Good", "Bad")

    def test_find_returns_media_type(self, client, tmp_path):
        """The /api/find response should include the media_type from the dataset."""
        import json
        import pickle
        import time

        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.registry import register_detector

        # Create a target dataset pkl
        target_medias = {}
        for i in range(5):
            emb = np.random.RandomState(i).randn(512).astype(np.float32)
            target_medias[i] = {
                "id": i,
                "type": "audio",
                "embedding": emb,
                "md5": f"mt_md5_{i}",
                "filename": f"mt_{i}.wav",
                "origin_name": f"mt_{i}.wav",
                "origin": {"importer": "server_folder", "params": {"path": "/mt/folder"}},
            }

        pkl_path = tmp_path / "mt_target.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": target_medias}, f)

        ds = register_dataset(name="mt_ds", media_type="audio", num_items=5, pkl_path=str(pkl_path))

        # Create a detector with labels
        label_folder = tmp_path / "mt_label_audio"
        label_folder.mkdir()
        (label_folder / "good.wav").write_bytes(b"good_content")
        (label_folder / "bad.wav").write_bytes(b"bad_content")

        label_origin = {"importer": "server_folder", "params": {"path": str(label_folder)}}
        from vtscore.detectors.store import _detector_path, _write_detector

        tm_name = "Test MT Detector"
        tm_data = {
            "name": tm_name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {
                        "md5": "g_md5",
                        "label": "good",
                        "origin": label_origin,
                        "origin_name": "good.wav",
                        "filename": "good.wav",
                    },
                    {
                        "md5": "b_md5",
                        "label": "bad",
                        "origin": label_origin,
                        "origin_name": "bad.wav",
                        "filename": "bad.wav",
                    },
                ]
            },
        }
        _write_detector(_detector_path(tm_name), tm_data)

        model_entry = register_detector(
            name="Test MT Detector",
            media_type="audio",
            num_training=2,
        )

        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)
        bad_emb = np.random.RandomState(200).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            return good_emb if "good" in Path(path).name else bad_emb

        with patch("vtscore.detectors.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find",
                data=json.dumps({"dataset_ids": [ds["id"]], "detector_ids": [model_entry["id"]]}),
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["media_type"] == "audio"

    def test_find_returns_negative_results(self, client, tmp_path):
        """The /api/find response should include negative_results for Bad verdicts."""
        import json
        import pickle
        import time

        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.registry import register_detector

        # Create a target dataset pkl
        target_medias = {}
        for i in range(5):
            emb = np.random.RandomState(i).randn(512).astype(np.float32)
            target_medias[i] = {
                "id": i,
                "type": "image",
                "embedding": emb,
                "md5": f"nr_md5_{i}",
                "filename": f"nr_{i}.jpg",
                "origin_name": f"nr_{i}.jpg",
                "origin": {"importer": "server_folder", "params": {"path": "/nr/folder"}},
            }

        pkl_path = tmp_path / "nr_target.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": target_medias}, f)

        ds = register_dataset(name="nr_ds", media_type="image", num_items=5, pkl_path=str(pkl_path))

        # Create a detector with labels
        label_folder = tmp_path / "nr_label"
        label_folder.mkdir()
        (label_folder / "good.jpg").write_bytes(b"good_content")
        (label_folder / "bad.jpg").write_bytes(b"bad_content")

        label_origin = {"importer": "server_folder", "params": {"path": str(label_folder)}}
        from vtscore.detectors.store import _detector_path, _write_detector

        tm_name = "Test NR Detector"
        tm_data = {
            "name": tm_name,
            "text_query": "",
            "media_type": "image",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {
                        "md5": "g_md5",
                        "label": "good",
                        "origin": label_origin,
                        "origin_name": "good.jpg",
                        "filename": "good.jpg",
                    },
                    {
                        "md5": "b_md5",
                        "label": "bad",
                        "origin": label_origin,
                        "origin_name": "bad.jpg",
                        "filename": "bad.jpg",
                    },
                ]
            },
        }
        _write_detector(_detector_path(tm_name), tm_data)

        model_entry = register_detector(
            name="Test NR Detector",
            media_type="image",
            num_training=2,
        )

        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)
        bad_emb = np.random.RandomState(200).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            return good_emb if "good" in Path(path).name else bad_emb

        with patch("vtscore.detectors.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find",
                data=json.dumps({"dataset_ids": [ds["id"]], "detector_ids": [model_entry["id"]]}),
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = resp.get_json()

        # Should have both results and negative_results
        assert "negative_results" in data
        assert isinstance(data["negative_results"], list)

        # Total of results + negative_results should equal the dataset size
        total = len(data["results"]) + len(data["negative_results"])
        assert total == 5

        # media_type should be set correctly from the dataset
        assert data["media_type"] == "image"


class TestFindCheckLabels:
    """Tests for /api/find/check-labels pre-flight endpoint."""

    def test_empty_params_returns_no_warnings(self, client):
        resp = client.post(
            "/api/find/check-labels",
            data="{}",
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["warnings"] == []

    def test_no_warnings_when_labels_match_dataset(self, client, tmp_path):
        """When labels match the target dataset directly, no warnings."""
        import json
        import pickle
        import time

        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.registry import register_detector
        from vtscore.detectors.store import _detector_path, _write_detector

        # Create a dataset where labels match by md5
        medias = {}
        for i in range(3):
            emb = np.random.RandomState(i).randn(512).astype(np.float32)
            medias[i] = {
                "id": i,
                "type": "audio",
                "embedding": emb,
                "md5": f"cl_match_{i}",
                "filename": f"clip_{i}.wav",
            }

        pkl_path = tmp_path / "cl_match.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": medias}, f)

        ds = register_dataset(name="cl_match_ds", media_type="audio", num_items=3, pkl_path=str(pkl_path))

        # Trainable model with labels that match by md5
        tm_name = "Match Model"
        tm_data = {
            "name": "Match Model",
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {"md5": "cl_match_0", "label": "good", "origin_name": "clip_0.wav", "filename": "clip_0.wav"},
                    {"md5": "cl_match_1", "label": "bad", "origin_name": "clip_1.wav", "filename": "clip_1.wav"},
                ]
            },
        }
        _write_detector(_detector_path(tm_name), tm_data)

        model_entry = register_detector(
            name="Match Model",
            media_type="audio",
            num_training=2,
        )

        resp = client.post(
            "/api/find/check-labels",
            data=json.dumps({"dataset_ids": [ds["id"]], "detector_ids": [model_entry["id"]]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["warnings"] == []

    def test_warnings_when_labels_fail_to_resolve(self, client, tmp_path):
        """When labels don't match and can't be resolved, return warnings."""
        import json
        import pickle
        import time

        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.registry import register_detector
        from vtscore.detectors.store import _detector_path, _write_detector

        # Create a target dataset (no overlap with labels)
        medias = {}
        for i in range(3):
            emb = np.random.RandomState(i).randn(512).astype(np.float32)
            medias[i] = {
                "id": i,
                "type": "audio",
                "embedding": emb,
                "md5": f"cl_diff_{i}",
                "filename": f"other_{i}.wav",
            }

        pkl_path = tmp_path / "cl_diff.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": medias}, f)

        ds = register_dataset(name="cl_diff_ds", media_type="audio", num_items=3, pkl_path=str(pkl_path))

        # Trainable model with labels from a nonexistent folder
        label_origin = {"importer": "server_folder", "params": {"path": "/nonexistent/folder"}}
        tm_name = "Diff Model"
        tm_data = {
            "name": tm_name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {
                        "md5": "no_match_g",
                        "label": "good",
                        "origin": label_origin,
                        "origin_name": "good.wav",
                        "filename": "good.wav",
                    },
                    {
                        "md5": "no_match_b1",
                        "label": "bad",
                        "origin": label_origin,
                        "origin_name": "bad1.wav",
                        "filename": "bad1.wav",
                    },
                    {
                        "md5": "no_match_b2",
                        "label": "bad",
                        "origin": label_origin,
                        "origin_name": "bad2.wav",
                        "filename": "bad2.wav",
                    },
                ]
            },
        }
        _write_detector(_detector_path(tm_name), tm_data)

        model_entry = register_detector(
            name=tm_name,
            media_type="audio",
            num_training=3,
        )

        resp = client.post(
            "/api/find/check-labels",
            data=json.dumps({"dataset_ids": [ds["id"]], "detector_ids": [model_entry["id"]]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["warnings"]) == 1

        w = data["warnings"][0]
        assert w["detector_name"] == "Diff Model"
        assert w["total_labels"] == 3
        assert w["failed_labels"] == 3
        assert w["resolved_labels"] == 0

    def test_partial_resolution_reports_correct_counts(self, client, tmp_path):
        """When some labels resolve and some don't, counts are accurate."""
        import json
        import pickle
        import time

        from vtscore.datasets.registry import register_dataset
        from vtscore.detectors.registry import register_detector
        from vtscore.detectors.store import _detector_path, _write_detector

        # Create a target dataset (no overlap with labels)
        medias = {}
        for i in range(3):
            emb = np.random.RandomState(i).randn(512).astype(np.float32)
            medias[i] = {
                "id": i,
                "type": "audio",
                "embedding": emb,
                "md5": f"cl_part_{i}",
                "filename": f"part_{i}.wav",
            }

        pkl_path = tmp_path / "cl_part.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": medias}, f)

        ds = register_dataset(name="cl_part_ds", media_type="audio", num_items=3, pkl_path=str(pkl_path))

        # Label folder — one file exists, one doesn't
        label_folder = tmp_path / "part_labels"
        label_folder.mkdir()
        (label_folder / "good.wav").write_bytes(b"good_audio")
        # bad.wav does NOT exist

        label_origin = {"importer": "server_folder", "params": {"path": str(label_folder), "media_type": "audio"}}
        tm_name = "Part Model"
        tm_data = {
            "name": tm_name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {
                        "md5": "no_match_g",
                        "label": "good",
                        "origin": label_origin,
                        "origin_name": "good.wav",
                        "filename": "good.wav",
                    },
                    {
                        "md5": "no_match_b",
                        "label": "bad",
                        "origin": label_origin,
                        "origin_name": "bad.wav",
                        "filename": "bad.wav",
                    },
                ]
            },
        }
        _write_detector(_detector_path(tm_name), tm_data)

        model_entry = register_detector(
            name=tm_name,
            media_type="audio",
            num_training=2,
        )

        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            return good_emb

        with patch("vtscore.detectors.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find/check-labels",
                data=json.dumps({"dataset_ids": [ds["id"]], "detector_ids": [model_entry["id"]]}),
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["warnings"]) == 1

        w = data["warnings"][0]
        assert w["detector_name"] == "Part Model"
        assert w["total_labels"] == 2
        assert w["resolved_labels"] == 1
        assert w["failed_labels"] == 1


# ---------------------------------------------------------------------------
# Logging on resolution failure
# ---------------------------------------------------------------------------


class TestResolutionWarningLogs:
    """Verify that resolve_label_embeddings logs warnings on resolution failure."""

    def test_logs_warning_on_total_failure(self, caplog):
        """Zero resolved labels out of N should emit a clear warning."""
        import logging

        labels = [
            {
                "label": "good",
                "origin": {"importer": "nonexistent", "params": {}},
                "origin_name": "a.wav",
                "filename": "a.wav",
            },
            {
                "label": "bad",
                "origin": {"importer": "nonexistent", "params": {}},
                "origin_name": "b.wav",
                "filename": "b.wav",
            },
        ]
        with caplog.at_level(logging.WARNING, logger="vtscore.detectors.resolver"):
            result = resolve_label_embeddings(labels, "audio")

        assert result.resolved_count == 0
        assert result.total_count == 2
        assert any("0 of 2 labels resolved" in m for m in caplog.messages)
        assert any("resolve_file()" in m or "origin_name=" in m for m in caplog.messages)

    def test_logs_warning_on_partial_failure(self, tmp_path, caplog):
        """Some resolved, some missing should log partial resolution warning."""
        import logging

        wav = tmp_path / "found.wav"
        wav.write_bytes(b"\x00" * 100)

        labels = [
            {"label": "good", "origin": None, "origin_name": "", "filename": ""},
            {
                "label": "bad",
                "origin": {"importer": "server_folder", "params": {"path": str(tmp_path)}},
                "origin_name": "found.wav",
                "filename": "found.wav",
            },
        ]

        dummy_emb = np.zeros(10)
        with (
            caplog.at_level(logging.WARNING, logger="vtscore.detectors.resolver"),
            patch("vtscore.detectors.resolver.embed_file", return_value=dummy_emb),
        ):
            result = resolve_label_embeddings(labels, "audio")

        assert result.resolved_count == 1
        assert result.total_count == 2
        assert any("1 of 2 labels resolved" in m for m in caplog.messages)


class TestResolveFileContextLifetime:
    """Regression: temp-backed MediaSources must stay alive across
    resolve + embed inside a single ``resolve_file_context`` block.

    Before the fix, ``_default_source_resolver`` created the source,
    asked for the path, and let the source go out of scope.  Sources
    that owned a ``tempfile.TemporaryDirectory`` (e.g. PullWrest) got
    finalised by GC and the path went stale before ``embed_file``
    opened it, raising ``FileNotFoundError`` deep inside the embedder.
    """

    def test_source_cleanup_deferred_to_context_exit(self, tmp_path, monkeypatch):
        from contextlib import ExitStack

        from vtscore.detectors import resolver as resolver_mod

        staging = tmp_path / "staging"
        staging.mkdir()
        media = staging / "thing.txt"
        media.write_bytes(b"payload")

        cleaned: list[bool] = []

        class _FakeSource:
            name = "fake"

            def resolve_path(self, origin_name: str = "", filename: str = ""):
                return media if media.exists() else None

            def cleanup(self) -> None:
                cleaned.append(True)
                if media.exists():
                    media.unlink()

        def _custom_source_resolver(
            stack: ExitStack,
            origin: dict,
            origin_name: str,
            filename: str,
        ):
            src = _FakeSource()
            stack.callback(src.cleanup)
            return src.resolve_path(origin_name, filename)

        monkeypatch.setattr(resolver_mod, "_source_resolver", _custom_source_resolver)
        monkeypatch.setattr(resolver_mod, "_auto_wired", True)

        origin = {"importer": "fake", "params": {}}
        with resolver_mod.resolve_file_context(origin, origin_name="thing.txt") as path:
            assert path is not None
            assert path.exists(), "file must be alive inside the with-block"
            assert path.read_bytes() == b"payload"
            assert cleaned == [], "cleanup() must not fire while context is open"

        assert cleaned == [True], "cleanup() must fire exactly once on exit"
        assert not media.exists(), "temp file gone after context exit"

    def test_legacy_wrapper_runs_cleanup_immediately(self, tmp_path, monkeypatch):
        """``resolve_file_from_origin`` is the non-CM wrapper — by design the
        source is dropped (and its temp dir cleaned) as soon as the call
        returns.  Callers that hold the returned path past that point are
        responsible for using ``resolve_file_context`` instead.
        """
        from contextlib import ExitStack

        from vtscore.detectors import resolver as resolver_mod

        cleaned: list[bool] = []

        class _FakeSource:
            name = "fake"

            def resolve_path(self, origin_name: str = "", filename: str = ""):
                return tmp_path / "x.txt"

            def cleanup(self) -> None:
                cleaned.append(True)

        def _custom_source_resolver(
            stack: ExitStack,
            origin: dict,
            origin_name: str,
            filename: str,
        ):
            src = _FakeSource()
            stack.callback(src.cleanup)
            return src.resolve_path(origin_name, filename)

        monkeypatch.setattr(resolver_mod, "_source_resolver", _custom_source_resolver)
        monkeypatch.setattr(resolver_mod, "_auto_wired", True)

        _ = resolver_mod.resolve_file_from_origin({"importer": "fake"}, "x.txt")
        assert cleaned == [True], "wrapper exits its context immediately"
