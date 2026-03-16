"""Tests for vtsearch.models.resolver — media file resolution from origin trails."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vtsearch.models.resolver import (
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

        origin = {"importer": "folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="clip.wav")
        assert result == folder / "clip.wav"

    def test_resolves_by_filename_fallback(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"fake_audio")

        origin = {"importer": "folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, filename="clip.wav")
        assert result == folder / "clip.wav"

    def test_resolves_nested_path(self, tmp_path):
        folder = tmp_path / "data"
        subfolder = folder / "category"
        subfolder.mkdir(parents=True)
        (subfolder / "item.wav").write_bytes(b"data")

        origin = {"importer": "folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="category/item.wav")
        assert result == subfolder / "item.wav"

    def test_returns_none_for_missing_file(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()

        origin = {"importer": "folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="nonexistent.wav")
        assert result is None

    def test_returns_none_for_missing_folder(self):
        origin = {"importer": "folder", "params": {"path": "/nonexistent/path"}}
        result = resolve_file_from_origin(origin, origin_name="clip.wav")
        assert result is None

    def test_returns_none_for_empty_path(self):
        origin = {"importer": "folder", "params": {"path": ""}}
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
                    "origin": {"importer": "folder", "params": {"path": "/gone"}},
                    "origin_name": "a.wav",
                    "filename": "a.wav",
                },
                {
                    "origin": {"importer": "folder", "params": {"path": str(folder)}},
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
                    "origin": {"importer": "folder", "params": {"path": "/gone"}},
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
                "parent_importer": "folder",
                "parent_path": str(folder),
            },
        }
        result = resolve_file_from_origin(origin)
        assert result == folder / "clip.mp4"


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
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

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
        from vtsearch.datasets.importers import _registry

        _registry._ensure_discovered()
        _registry._items[custom.name] = custom
        try:
            origin = {"importer": "test_custom", "params": {"path": str(tmp_path)}}
            result = resolve_file_from_origin(origin, origin_name="custom_media.wav")
            assert result == marker_file
        finally:
            _registry._items.pop(custom.name, None)


class TestResolveLabelEmbeddings:
    def test_resolves_folder_labels(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "good1.wav").write_bytes(b"good_audio")
        (folder / "bad1.wav").write_bytes(b"bad_audio")

        origin = {"importer": "folder", "params": {"path": str(folder), "media_type": "sounds"}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "good1.wav", "md5": "aaa", "filename": "good1.wav"},
            {"label": "bad", "origin": origin, "origin_name": "bad1.wav", "md5": "bbb", "filename": "bad1.wav"},
        ]

        fake_emb = np.zeros(512, dtype=np.float32)
        with patch("vtsearch.models.resolver.embed_file", return_value=fake_emb):
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

        origin = {"importer": "folder", "params": {"path": str(folder), "media_type": "sounds"}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "good1.wav", "md5": "aaa", "filename": "good1.wav"},
            {"label": "bad", "origin": origin, "origin_name": "missing.wav", "md5": "bbb", "filename": "missing.wav"},
        ]

        fake_emb = np.zeros(512, dtype=np.float32)
        with patch("vtsearch.models.resolver.embed_file", return_value=fake_emb):
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

        origin = {"importer": "folder", "params": {"path": str(folder)}}
        labels = [
            {"label": "good", "origin": origin, "origin_name": "clip.wav", "md5": "aaa", "filename": "clip.wav"},
        ]

        with patch("vtsearch.models.resolver.embed_file", return_value=None):
            result = resolve_label_embeddings(labels, "audio")

        assert result.resolved_count == 0
        assert result.total_count == 1
        assert len(result.missing_entries) == 1


class TestMultiFindCrossDatasetFallback:
    """Test that multi_find uses the resolver when labels don't match the target dataset."""

    def test_trainable_model_falls_back_to_resolver(self, client, tmp_path):
        """When a trainable model's labels don't match the target dataset,
        multi_find should fall back to resolving labels from their origins."""
        import json
        import pickle
        import time

        from vtsearch.datasets.registry import register_dataset
        from vtsearch.models.registry import register_model

        # Create a folder with media files for the trainable model's labels
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
                "origin": {"importer": "folder", "params": {"path": "/other/folder"}},
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

        # Create a trainable model with labels from label_folder
        label_origin = {"importer": "folder", "params": {"path": str(label_folder), "media_type": "sounds"}}
        from vtsearch.routes.trainable_models import _model_path, _write_model

        tm_name = "test_cross_detector"
        tm_path = _model_path(tm_name)
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
        _write_model(tm_path, tm_data)

        # Register the model
        model_entry = register_model(
            name="Test Cross Detector",
            media_type="audio",
            trainable=True,
            num_training=2,
            trainable_model_name=tm_name,
        )

        # Mock the embedder to return deterministic vectors
        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)
        bad_emb = np.random.RandomState(200).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            name = Path(path).name
            if "good" in name:
                return good_emb
            return bad_emb

        with patch("vtsearch.models.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find",
                data=json.dumps(
                    {
                        "dataset_ids": [ds["id"]],
                        "model_ids": [model_entry["id"]],
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
            verdicts = r["model_verdicts"]
            assert "Test Cross Detector" in verdicts
            assert verdicts["Test Cross Detector"]["verdict"] in ("Good", "Bad")

    def test_find_returns_media_type(self, client, tmp_path):
        """The /api/find response should include the media_type from the dataset."""
        import json
        import pickle
        import time

        from vtsearch.datasets.registry import register_dataset
        from vtsearch.models.registry import register_model

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
                "origin": {"importer": "folder", "params": {"path": "/mt/folder"}},
            }

        pkl_path = tmp_path / "mt_target.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": target_medias}, f)

        ds = register_dataset(name="mt_ds", media_type="audio", num_items=5, pkl_path=str(pkl_path))

        # Create a trainable model with labels
        label_folder = tmp_path / "mt_label_audio"
        label_folder.mkdir()
        (label_folder / "good.wav").write_bytes(b"good_content")
        (label_folder / "bad.wav").write_bytes(b"bad_content")

        label_origin = {"importer": "folder", "params": {"path": str(label_folder)}}
        from vtsearch.routes.trainable_models import _model_path, _write_model

        tm_name = "test_mt_detector"
        tm_data = {
            "name": "Test MT Detector",
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {"md5": "g_md5", "label": "good", "origin": label_origin, "origin_name": "good.wav", "filename": "good.wav"},
                    {"md5": "b_md5", "label": "bad", "origin": label_origin, "origin_name": "bad.wav", "filename": "bad.wav"},
                ]
            },
        }
        _write_model(_model_path(tm_name), tm_data)

        model_entry = register_model(
            name="Test MT Detector",
            media_type="audio",
            trainable=True,
            num_training=2,
            trainable_model_name=tm_name,
        )

        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)
        bad_emb = np.random.RandomState(200).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            return good_emb if "good" in Path(path).name else bad_emb

        with patch("vtsearch.models.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find",
                data=json.dumps({"dataset_ids": [ds["id"]], "model_ids": [model_entry["id"]]}),
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

        from vtsearch.datasets.registry import register_dataset
        from vtsearch.models.registry import register_model

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
                "origin": {"importer": "folder", "params": {"path": "/nr/folder"}},
            }

        pkl_path = tmp_path / "nr_target.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": target_medias}, f)

        ds = register_dataset(name="nr_ds", media_type="image", num_items=5, pkl_path=str(pkl_path))

        # Create a trainable model with labels
        label_folder = tmp_path / "nr_label"
        label_folder.mkdir()
        (label_folder / "good.jpg").write_bytes(b"good_content")
        (label_folder / "bad.jpg").write_bytes(b"bad_content")

        label_origin = {"importer": "folder", "params": {"path": str(label_folder)}}
        from vtsearch.routes.trainable_models import _model_path, _write_model

        tm_name = "test_nr_detector"
        tm_data = {
            "name": "Test NR Detector",
            "text_query": "",
            "media_type": "image",
            "examples": [],
            "created_at": time.time(),
            "labelset": {
                "labels": [
                    {"md5": "g_md5", "label": "good", "origin": label_origin, "origin_name": "good.jpg", "filename": "good.jpg"},
                    {"md5": "b_md5", "label": "bad", "origin": label_origin, "origin_name": "bad.jpg", "filename": "bad.jpg"},
                ]
            },
        }
        _write_model(_model_path(tm_name), tm_data)

        model_entry = register_model(
            name="Test NR Detector",
            media_type="image",
            trainable=True,
            num_training=2,
            trainable_model_name=tm_name,
        )

        good_emb = np.random.RandomState(100).randn(512).astype(np.float32)
        bad_emb = np.random.RandomState(200).randn(512).astype(np.float32)

        def fake_embed(path, media_type):
            return good_emb if "good" in Path(path).name else bad_emb

        with patch("vtsearch.models.resolver.embed_file", side_effect=fake_embed):
            resp = client.post(
                "/api/find",
                data=json.dumps({"dataset_ids": [ds["id"]], "model_ids": [model_entry["id"]]}),
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

    def test_pretrained_weights_cross_dataset_recalibrates_threshold(self, client, tmp_path):
        """A detector trained on Dataset A should produce Good hits when
        applied to Dataset B via /api/find, even when the stored threshold
        (calibrated on A) would reject everything on B.

        The fix: multi_find recalibrates the threshold using the target
        dataset's score distribution via calculate_safe_threshold.
        """
        import json
        import pickle

        from vtsearch.datasets.registry import register_dataset
        from vtsearch.models.registry import register_model
        from vtsearch.utils import add_autorun_detector

        rng = np.random.RandomState(42)

        # --- Build a detector trained on Dataset A ---
        # Simulate: 5 "good" embeddings cluster around [+1, +1, ...]
        # and 5 "bad" embeddings cluster around [-1, -1, ...]
        dim = 512
        good_embs_A = [rng.randn(dim).astype(np.float32) + 1.0 for _ in range(5)]
        bad_embs_A = [rng.randn(dim).astype(np.float32) - 1.0 for _ in range(5)]
        X_list = good_embs_A + bad_embs_A
        y_list = [1.0] * 5 + [0.0] * 5

        from vtsearch.routes.detectors_helpers import serialize_weights

        import torch

        from vtsearch.models.training import (
            calculate_cross_calibration_threshold,
            train_model,
        )

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        input_dim = X.shape[1]
        model = train_model(X, y, input_dim)
        weights = serialize_weights(model)

        # Compute a threshold on training data — likely very high (model
        # fits training data perfectly, scores are near 0 or 1).
        threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim)

        # Register the detector with weights
        det_name = "cross_ds_detector"
        add_autorun_detector(
            det_name, "image", weights, threshold,
            num_labels=10,
        )

        # Register in model registry
        model_entry = register_model(
            name="Cross DS Detector",
            media_type="image",
            trainable=False,
            num_training=10,
            detector_name=det_name,
        )

        # --- Build Dataset B with different embeddings ---
        # Dataset B has items that are shifted: "good-like" items are
        # around [+0.5, +0.5, ...] (weaker signal than training data).
        target_medias = {}
        # 3 items that should be "good" (positive direction, but weaker)
        for i in range(3):
            emb = rng.randn(dim).astype(np.float32) * 0.5 + 0.5
            target_medias[i] = {
                "id": i,
                "type": "image",
                "embedding": emb,
                "md5": f"good_target_{i}",
                "filename": f"good_{i}.jpg",
                "origin_name": f"good_{i}.jpg",
                "origin": {"importer": "folder", "params": {"path": "/target"}},
            }
        # 7 items that should be "bad" (negative direction)
        for i in range(3, 10):
            emb = rng.randn(dim).astype(np.float32) * 0.5 - 0.5
            target_medias[i] = {
                "id": i,
                "type": "image",
                "embedding": emb,
                "md5": f"bad_target_{i}",
                "filename": f"bad_{i}.jpg",
                "origin_name": f"bad_{i}.jpg",
                "origin": {"importer": "folder", "params": {"path": "/target"}},
            }

        pkl_path = tmp_path / "target_cross.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": target_medias}, f)

        ds = register_dataset(
            name="target_cross_ds",
            media_type="image",
            num_items=10,
            pkl_path=str(pkl_path),
        )

        # --- Run Find ---
        resp = client.post(
            "/api/find",
            data=json.dumps({
                "dataset_ids": [ds["id"]],
                "model_ids": [model_entry["id"]],
            }),
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()

        # The key assertion: there SHOULD be Good hits because the
        # threshold is recalibrated for Dataset B's score distribution.
        # Without recalibration, the training threshold is too high and
        # everything would be classified as Bad.
        assert data["total_hits"] > 0, (
            f"Expected Good hits but got 0. "
            f"Results: {len(data['results'])}, "
            f"Negative: {len(data['negative_results'])}"
        )
