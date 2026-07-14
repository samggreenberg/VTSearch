"""Tests for the RVL-CDIP document-image demo.

RVL-CDIP's canonical repo is a 38 GB tarball, so the demo pulls a demo-sized,
class-balanced 100-per-class parquet mirror (``jordyvl/...``) whose
``image``/``label`` columns decode into a ``<class>/<idx>.png`` folder-per-class
tree (the integer ``label`` indexes ``RVL_CDIP_CATEGORIES`` in canonical
RVL-CDIP order), reusing the Caltech/Food-101 collect path.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _png_bytes(color=(200, 200, 200)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (40, 50), color[0]).save(buf, format="PNG")
    return buf.getvalue()


class TestDownloadRvlCdip:
    def test_labels_map_to_class_folders(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets.downloader import _hf_parquet
        from vtscore.media.image._demo_categories import RVL_CDIP_CATEGORIES

        # canonical order: label 2 -> "email", 6 -> "scientific publication";
        # 99 is out of range -> skipped.
        rows = [
            {"image": {"bytes": _png_bytes(), "path": None}, "label": 2},
            {"image": {"bytes": _png_bytes(), "path": None}, "label": 6},
            {"image": {"bytes": _png_bytes(), "path": None}, "label": 99},
        ]

        def fake_download_shards(repo_id, filenames, dest_dir, dataset_name, on_progress):
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            shard = Path(dest_dir) / "train-00000-of-00001-abc.parquet"
            shard.write_bytes(b"parquet")
            return [shard]

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                _hf_parquet, "list_parquet_shards", lambda repo, prefix: ["data/train-00000-of-00001-abc.parquet"]
            ),
            patch.object(_hf_parquet, "download_parquet_shards", fake_download_shards),
            patch.object(_hf_parquet, "iter_parquet_rows", lambda shard, cols, batch_size=256: iter(rows)),
        ):
            result = dl_module.download_rvl_cdip(on_progress=lambda *a: None)

        assert result == tmp_path / "rvl_cdip" / "images"
        assert next((result / RVL_CDIP_CATEGORIES[2]).glob("*.png"), None) is not None  # email
        assert next((result / RVL_CDIP_CATEGORIES[6]).glob("*.png"), None) is not None  # scientific publication
        # the out-of-range label produced no folder
        assert sum(1 for _ in result.iterdir()) == 2
        assert not (tmp_path / "rvl_cdip" / "parquet").exists()

    def test_cached_skips_download_when_all_classes_present(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets.downloader import _hf_parquet
        from vtscore.media.image._demo_categories import RVL_CDIP_CATEGORIES

        # A *complete* tree has every one of the 16 class folders populated.
        images = tmp_path / "rvl_cdip" / "images"
        for cat in RVL_CDIP_CATEGORIES:
            (images / cat).mkdir(parents=True)
            (images / cat / "0.png").write_bytes(_png_bytes())

        called = []
        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(_hf_parquet, "list_parquet_shards", lambda *a, **k: called.append(True) or []),
        ):
            result = dl_module.download_rvl_cdip(on_progress=lambda *a: None)

        assert not called
        assert result == images

    def test_partial_decode_triggers_redownload(self, tmp_path):
        """An interrupted decode (only some class folders) must NOT look complete."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets.downloader import _hf_parquet
        from vtscore.media.image._demo_categories import RVL_CDIP_CATEGORIES

        # Only one class present -> the old any-png probe would false-positive.
        images = tmp_path / "rvl_cdip" / "images" / RVL_CDIP_CATEGORIES[11]  # invoice
        images.mkdir(parents=True)
        (images / "0.png").write_bytes(_png_bytes())

        # Full decode fills every class folder from the parquet rows.
        rows = [{"image": {"bytes": _png_bytes(), "path": None}, "label": i} for i in range(len(RVL_CDIP_CATEGORIES))]

        def fake_download_shards(repo_id, filenames, dest_dir, dataset_name, on_progress):
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            shard = Path(dest_dir) / "train-00000-of-00001-abc.parquet"
            shard.write_bytes(b"parquet")
            return [shard]

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(_hf_parquet, "list_parquet_shards", lambda repo, prefix: ["data/train-00000-of-00001-abc.parquet"]),
            patch.object(_hf_parquet, "download_parquet_shards", fake_download_shards),
            patch.object(_hf_parquet, "iter_parquet_rows", lambda shard, cols, batch_size=256: iter(rows)),
        ):
            result = dl_module.download_rvl_cdip(on_progress=lambda *a: None)

        # Re-download ran and every class folder is now populated.
        from vtscore.media.image._demo_categories import RVL_CDIP_CATEGORIES as CATS

        assert all(next((result / c).glob("*.png"), None) is not None for c in CATS)


class TestLoadDemoSourceRvl:
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

        images = tmp_path / "rvl_cdip" / "images"
        for cat in ("invoice", "resume"):
            (images / cat).mkdir(parents=True)
            for i in range(2):
                Image.new("L", (20, 30), 128).save(images / cat / f"{i}.png")

        clips: dict = {}
        with patch.object(dl_module, "download_rvl_cdip", return_value=images):
            ImageMediaType().load_demo_source(
                source="rvl_cdip",
                categories=["invoice", "resume"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=self._make_mock_embedder(),
            )

        assert len(clips) == 4
        assert {c["category"] for c in clips.values()} == {"invoice", "resume"}


class TestRvlCategoriesList:
    def test_has_16_unique_in_mirror_order(self):
        from vtscore.media.image._demo_categories import RVL_CDIP_CATEGORIES

        assert len(RVL_CDIP_CATEGORIES) == 16
        assert len(set(RVL_CDIP_CATEGORIES)) == 16
        # Order is the parquet mirror's canonical RVL-CDIP ClassLabel ordering
        # (index == label int), NOT alphabetical.
        assert RVL_CDIP_CATEGORIES[0] == "letter"
        assert RVL_CDIP_CATEGORIES[2] == "email"
        assert RVL_CDIP_CATEGORIES[11] == "invoice"
        assert RVL_CDIP_CATEGORIES[15] == "memo"
