"""Tests for the RICO-Screen2Words mobile-UI screenshot demo.

RICO-Screen2Words ships on the Hub as parquet shards whose ``image`` column
holds screenshot bytes and whose ``category`` column holds the app's Google Play
genre.  ``download_rico_screen2words`` decodes each curated-category screenshot
into a ``<category>/<screenId>.jpg`` folder-per-class tree, after which it reuses
the same collect/embed path as Caltech/Food-101.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _jpeg_bytes(color=(20, 40, 60)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 60), color).save(buf, format="JPEG")
    return buf.getvalue()


class TestRicoNorm:
    def test_folds_case_and_punctuation(self):
        from vtscore.datasets.downloader.images import _rico_norm

        assert _rico_norm("Maps & Navigation") == "mapsnavigation"
        assert _rico_norm("News & Magazines") == "newsmagazines"
        assert _rico_norm("Finance") == "finance"
        assert _rico_norm(None) == ""


class TestExtractImagesToFolders:
    def test_buckets_by_category_and_skips_missing(self, tmp_path):
        from vtscore.datasets.downloader import _hf_parquet

        rows = [
            {"image": {"bytes": _jpeg_bytes(), "path": None}, "category": "Finance", "screenId": 1},
            {"image": {"bytes": _jpeg_bytes(), "path": None}, "category": "finance", "screenId": 2},  # folds
            {"image": {"bytes": _jpeg_bytes(), "path": None}, "category": "Weather", "screenId": 3},
            {"image": {"bytes": None, "path": None}, "category": "Finance", "screenId": 4},  # no bytes -> skip
            {"image": {"bytes": _jpeg_bytes(), "path": None}, "category": "Unknownzzz", "screenId": 5},  # skip
        ]
        display_by_norm = {
            _hf_parquet_norm("Finance"): "Finance",
            _hf_parquet_norm("Weather"): "Weather",
        }

        def category_of(row):
            return display_by_norm.get(_hf_parquet_norm(row.get("category")))

        out = tmp_path / "screenshots"
        with patch.object(_hf_parquet, "iter_parquet_rows", lambda shard, cols, batch_size=256: iter(rows)):
            _hf_parquet.extract_images_to_folders(
                [tmp_path / "fake.parquet"],
                image_col="image",
                out_dir=out,
                category_of=category_of,
                id_of=lambda row, idx: str(row.get("screenId") or idx),
                ext="jpg",
                dataset_name="RICO-Screen2Words",
                on_progress=lambda *a: None,
                columns=["image", "category", "screenId"],
            )

        assert sorted(p.name for p in (out / "Finance").glob("*.jpg")) == ["1.jpg", "2.jpg"]
        assert [p.name for p in (out / "Weather").glob("*.jpg")] == ["3.jpg"]
        assert not (out / "Unknownzzz").exists()


class TestDownloadRicoScreen2Words:
    def test_extracts_then_deletes_parquet(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets.downloader import _hf_parquet

        rows = [
            {"image": {"bytes": _jpeg_bytes(), "path": None}, "category": "Finance", "screenId": 10},
            {"image": {"bytes": _jpeg_bytes(), "path": None}, "category": "Weather", "screenId": 11},
        ]

        def fake_download_shards(repo_id, filenames, dest_dir, dataset_name, on_progress):
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            shard = Path(dest_dir) / "train-00000-of-00008.parquet"
            shard.write_bytes(b"parquet")
            return [shard]

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(_hf_parquet, "download_parquet_shards", fake_download_shards),
            patch.object(_hf_parquet, "iter_parquet_rows", lambda shard, cols, batch_size=256: iter(rows)),
        ):
            result = dl_module.download_rico_screen2words(on_progress=lambda *a: None)

        assert result == tmp_path / "rico_screen2words" / "screenshots"
        assert (result / "Finance" / "10.jpg").exists()
        assert (result / "Weather" / "11.jpg").exists()
        # parquet shards removed after extraction
        assert not (tmp_path / "rico_screen2words" / "parquet").exists()

    def test_cached_skips_download(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets.downloader import _hf_parquet

        shots = tmp_path / "rico_screen2words" / "screenshots" / "Finance"
        shots.mkdir(parents=True)
        (shots / "1.jpg").write_bytes(_jpeg_bytes())

        called = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(_hf_parquet, "download_parquet_shards", lambda *a, **k: called.append(True) or []),
        ):
            result = dl_module.download_rico_screen2words(on_progress=lambda *a: None)

        assert not called
        assert result == tmp_path / "rico_screen2words" / "screenshots"


class TestLoadDemoSourceRico:
    def _make_mock_embedder(self):
        emb = MagicMock()
        emb.name = "siglip"
        emb.media_type_id = "image"
        emb._model = True
        emb.embed_media = MagicMock(return_value=np.zeros(768, dtype=np.float32))
        return emb

    def test_folder_per_class_collect(self, tmp_path):
        from PIL import Image

        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        shots = tmp_path / "rico_screen2words" / "screenshots"
        for cat in ("Finance", "Weather"):
            (shots / cat).mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (20, 40), (1, 2, 3)).save(shots / cat / f"{cat}_{i}.jpg")

        clips: dict = {}
        with patch.object(dl_module, "download_rico_screen2words", return_value=shots):
            ImageMediaType().load_demo_source(
                source="rico_screen2words",
                categories=["Finance", "Weather"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
            )

        assert len(clips) == 6
        assert {c["category"] for c in clips.values()} == {"Finance", "Weather"}


class TestRicoCategoriesList:
    def test_has_16_unique(self):
        from vtscore.media.image._demo_categories import RICO_SCREEN2WORDS_CATEGORIES

        assert len(RICO_SCREEN2WORDS_CATEGORIES) == 16
        assert len(set(RICO_SCREEN2WORDS_CATEGORIES)) == 16
        assert "Finance" in RICO_SCREEN2WORDS_CATEGORIES
        assert "Maps & Navigation" in RICO_SCREEN2WORDS_CATEGORIES


def _hf_parquet_norm(s):
    """Local copy of the downloader's category normaliser for test bucketing."""
    import re

    return re.sub(r"[^a-z0-9]", "", (s or "").lower())
