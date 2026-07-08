"""Tests for the OpenLogo (QMUL-OpenLogo) logo dataset download + load_demo_source.

OpenLogo is the structural embedder's instance-matching *logo* demo.  It ships as
a FiftyOne dataset on HuggingFace (a flat ``data/`` media folder plus a
``samples.json`` of per-image ``ground_truth`` detections with normalized
``[x, y, w, h]`` boxes), pulled with ``huggingface_hub.snapshot_download`` and
parsed with the stdlib — no ``fiftyone`` dependency.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestDownloadOpenlogo:
    def test_returns_extract_directory(self, tmp_path):
        """download_openlogo pulls the snapshot and returns the openlogo/ directory."""
        from vtscore.datasets import downloader as dl_module

        def fake_snapshot(*, repo_id, repo_type, local_dir, ignore_patterns, token, tqdm_class=None, max_workers=8):
            d = Path(local_dir)
            (d / "data").mkdir(parents=True, exist_ok=True)
            (d / "data" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
            (d / "samples.json").write_text('{"samples": []}')

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            result = dl_module.download_openlogo(on_progress=lambda *a: None)

        assert result.name == "openlogo"
        assert (result / "data").is_dir()
        assert (result / "samples.json").exists()

    def test_forwards_widened_worker_count(self, tmp_path):
        """The download parallelizes across OpenLogo's ~27k tiny files.

        With no single archive to stream, wall-clock is dominated by per-file
        request latency, so we widen snapshot_download's default of 8 workers.
        Assert the configured count is forwarded rather than left at the default.
        """
        from vtscore.datasets import downloader as dl_module

        captured: dict = {}

        def fake_snapshot(*, repo_id, repo_type, local_dir, ignore_patterns, token, tqdm_class=None, max_workers=8):
            captured["max_workers"] = max_workers
            d = Path(local_dir)
            (d / "data").mkdir(parents=True, exist_ok=True)
            (d / "data" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
            (d / "samples.json").write_text('{"samples": []}')

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            dl_module.download_openlogo(on_progress=lambda *a: None)

        assert captured["max_workers"] == dl_module.core.OPENLOGO_DOWNLOAD_WORKERS
        assert captured["max_workers"] > 8

    def test_reports_measurable_file_progress(self, tmp_path):
        """The tqdm_class handed to snapshot_download forwards a live file count.

        Every other downloader streams byte-level progress so the UI leads with a
        count ("497/1.10GB ..."); OpenLogo is thousands of loose files, so it must
        drive snapshot_download's over-the-files bar instead. Simulate that bar and
        assert the callback sees a measurable (current, total), not just total=0.
        """
        from vtscore.datasets import downloader as dl_module

        events: list = []

        def fake_snapshot(*, repo_id, repo_type, local_dir, ignore_patterns, token, tqdm_class, max_workers=8):
            d = Path(local_dir)
            (d / "data").mkdir(parents=True, exist_ok=True)
            (d / "data" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
            (d / "samples.json").write_text('{"samples": []}')
            # Mimic how snapshot_download drives the over-the-files bar.
            bar = tqdm_class(total=3)
            for _ in range(3):
                bar.update(1)
            bar.close()

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch("huggingface_hub.snapshot_download", fake_snapshot),
        ):
            dl_module.download_openlogo(on_progress=lambda *a: events.append(a))

        measurable = [e for e in events if e[0] == "downloading" and e[3] > 0]
        assert measurable, f"expected a measurable download event, got {events}"
        assert measurable[-1] == ("downloading", "Downloading OpenLogo logos", 3, 3)

    def test_cached_snapshot_skips_download(self, tmp_path):
        """If samples.json and data/ already exist, snapshot_download is not called."""
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "openlogo"
        (extract_dir / "data").mkdir(parents=True)
        (extract_dir / "data" / "img1.jpg").write_bytes(b"\xff\xd8")
        (extract_dir / "samples.json").write_text('{"samples": []}')

        called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch("huggingface_hub.snapshot_download", lambda **kw: called.append(True)),
        ):
            result = dl_module.download_openlogo(on_progress=lambda *a: None)

        assert not called
        assert result.exists()


class TestLoadDemoSourceOpenlogo:
    """ImageMediaType.load_demo_source with source='openlogo'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "sift_vlad"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(8192, dtype=np.float32))
        return mock_emb

    def _prepare_dataset_dir(self, tmp_path: Path, samples: list) -> Path:
        from PIL import Image

        ds_dir = tmp_path / "openlogo"
        data_dir = ds_dir / "data"
        data_dir.mkdir(parents=True)
        for s in samples:
            Image.new("RGB", (100, 100), (12, 34, 56)).save(data_dir / Path(s["filepath"]).name)
        (ds_dir / "samples.json").write_text(json.dumps({"samples": samples}))
        return ds_dir

    def test_buckets_by_brand_with_normalized_matching_and_regions(self, tmp_path):
        """Brand labels match categories punctuation/case-insensitively; boxes become regions."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        samples = [
            # "coca-cola" must match the "Coca-Cola" display category.
            {
                "filepath": "/orig/img1.jpg",
                "ground_truth": {"detections": [{"label": "coca-cola", "bounding_box": [0.1, 0.2, 0.3, 0.4]}]},
            },
            # Multi-label image: two in-vocab brands, "stellaartois" -> "Stella Artois".
            {
                "filepath": "img2.jpg",
                "ground_truth": {
                    "detections": [
                        {"label": "stellaartois", "bounding_box": [0.0, 0.0, 0.5, 0.5]},
                        {"label": "pepsi", "bounding_box": [0.5, 0.5, 0.4, 0.4]},
                    ]
                },
            },
            # Out-of-vocab brand only -> image skipped entirely.
            {
                "filepath": "img3.jpg",
                "ground_truth": {"detections": [{"label": "notabrand", "bounding_box": [0.0, 0.0, 1.0, 1.0]}]},
            },
        ]
        ds_dir = self._prepare_dataset_dir(tmp_path, samples)

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_openlogo", return_value=ds_dir):
            mt.load_demo_source(
                source="openlogo",
                categories=["Coca-Cola", "Pepsi", "Stella Artois"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        # img3 (no in-vocab brand) is dropped; img1 + img2 remain.
        assert len(clips) == 2
        by_primary = {c["category"]: c for c in clips.values()}

        # img1: single brand, box [x,y,w,h]=[.1,.2,.3,.4] -> [x0,y0,x1,y1]=[.1,.2,.4,.6].
        coke = by_primary["Coca-Cola"]
        assert coke["categories"] == ["Coca-Cola"]
        assert coke["regions"] == [{"box": [0.1, 0.2, 0.4, 0.6], "label": "Coca-Cola"}]

        # img2: multi-label, both brands present with a region each.
        stella = by_primary["Stella Artois"]
        assert set(stella["categories"]) == {"Stella Artois", "Pepsi"}
        assert len(stella["regions"]) == 2

    def test_slice_is_applied(self, tmp_path):
        """Flat slicing limits how many images are loaded."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        samples = [
            {
                "filepath": f"img_{i:03d}.jpg",
                "ground_truth": {"detections": [{"label": "pepsi", "bounding_box": [0.1, 0.1, 0.2, 0.2]}]},
            }
            for i in range(10)
        ]
        ds_dir = self._prepare_dataset_dir(tmp_path, samples)

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_openlogo", return_value=ds_dir):
            mt.load_demo_source(
                source="openlogo",
                categories=["Pepsi"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
                slice_frac_start=0.0,
                slice_frac_end=0.3,
            )

        assert len(clips) == 3


class TestOpenlogoCategoriesList:
    def test_has_32_unique_entries(self):
        from vtscore.media.image._demo_categories import OPENLOGO_CATEGORIES

        cats = OPENLOGO_CATEGORIES
        assert len(cats) == 32
        assert len(set(cats)) == 32
        # The FlickrLogos-32 supervised core, in display form.
        assert "Coca-Cola" in cats
        assert "Stella Artois" in cats

    def test_normalize_helper_folds_punctuation_and_case(self):
        from vtscore.media.image._demo_sources import _openlogo_norm

        assert _openlogo_norm("Coca-Cola") == "cocacola"
        assert _openlogo_norm("Stella Artois") == "stellaartois"
        assert _openlogo_norm("Foster's") == "fosters"
        assert _openlogo_norm("UPS") == "ups"
