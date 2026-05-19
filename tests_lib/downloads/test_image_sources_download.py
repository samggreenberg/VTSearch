"""Tests for new image dataset downloads and load_demo_source integration.

Covers: Oxford Flowers 102, Food-101, EuroSAT, Stanford Dogs, Places365.
"""

import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oxford_flowers_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal Oxford Flowers tgz and labels MAT fixtures."""
    import scipy.io  # noqa: PLC0415

    tgz_path = tmp_path / "102flowers.tgz"
    tree_root = tmp_path / "tgz_staging" / "jpg"
    tree_root.mkdir(parents=True)

    # Create 6 tiny JPEG-like files.
    for i in range(1, 7):
        (tree_root / f"image_{i:05d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    import tarfile as tf_mod

    with tf_mod.open(tgz_path, "w:gz") as tf:
        tf.add(tree_root, arcname="jpg")

    # Create labels MAT file: 6 images, labels cycle through 1-3.
    labels = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
    mat_path = tmp_path / "imagelabels.mat"
    scipy.io.savemat(str(mat_path), {"labels": labels.reshape(1, -1)})

    return tgz_path, mat_path


def _make_food101_tar(tmp_path: Path) -> Path:
    """Create a minimal Food-101 tar.gz fixture."""
    tar_path = tmp_path / "food-101.tar.gz"

    tree_root = tmp_path / "tar_staging" / "food-101" / "images"
    for cat in ("apple_pie", "sushi"):
        d = tree_root / cat
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(tmp_path / "tar_staging" / "food-101", arcname="food-101")

    return tar_path


def _make_eurosat_zip(tmp_path: Path) -> Path:
    """Create a minimal EuroSAT zip fixture."""
    zip_path = tmp_path / "EuroSAT_RGB.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        for cat in ("Forest", "Residential"):
            for i in range(3):
                zf.writestr(f"EuroSAT_RGB/{cat}/{cat}_{i:05d}.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    return zip_path


def _make_places365_tar(tmp_path: Path) -> Path:
    """Create a minimal Places365 val_256 tar fixture."""
    tar_path = tmp_path / "val_256.tar"

    tree_root = tmp_path / "tar_staging" / "val_256"
    tree_root.mkdir(parents=True)
    for i in range(1, 5):
        (tree_root / f"Places365_val_{i:08d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    with tarfile.open(tar_path, "w") as tf:
        tf.add(tree_root, arcname="val_256")

    return tar_path


def _make_places365_filelist_tar(tmp_path: Path, labels_bytes: bytes) -> Path:
    """Create a minimal Places365 filelist tar fixture.

    Mirrors the structure of MIT's ``filelist_places365-standard.tar``: a
    tarball whose root contains a ``places365_val.txt`` member (alongside
    other label files we don't use).  The production code only reads the
    val member, so that's the only one we need to fake.
    """
    tar_path = tmp_path / "filelist_places365-standard.tar"
    labels_src = tmp_path / "_filelist_staging_places365_val.txt"
    labels_src.write_bytes(labels_bytes)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(labels_src, arcname="places365_val.txt")
    labels_src.unlink()
    return tar_path


def _make_stanford_dogs_tar(tmp_path: Path) -> Path:
    """Create a minimal Stanford Dogs tar.gz fixture."""
    tar_path = tmp_path / "stanford_dogs_images.tar.gz"

    tree_root = tmp_path / "tar_staging" / "Images"
    for breed_dir_name in ("n02085620-Chihuahua", "n02099601-golden_retriever"):
        d = tree_root / breed_dir_name
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{breed_dir_name}_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(tree_root, arcname="Images")

    return tar_path


# ---------------------------------------------------------------------------
# download_oxford_flowers
# ---------------------------------------------------------------------------


class TestDownloadOxfordFlowers:
    def test_returns_extract_directory(self, tmp_path):
        """download_oxford_flowers returns the oxford_flowers/ directory."""
        from vtsearch.datasets import downloader as dl_module

        tgz_path, mat_path = _make_oxford_flowers_fixture(tmp_path)

        def fake_download(url, dest, size, cb):
            if "imagelabels" in url:
                import shutil

                shutil.copy(str(mat_path), str(dest))
            else:
                import shutil

                shutil.copy(str(tgz_path), str(dest))

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            result = dl_module.download_oxford_flowers(on_progress=lambda *a: None)

        assert result.name == "oxford_flowers"
        assert (result / "jpg").is_dir()
        assert (result / "imagelabels.mat").exists()

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract directory already exists, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        extract_dir = tmp_path / "oxford_flowers"
        jpg_dir = extract_dir / "jpg"
        jpg_dir.mkdir(parents=True)
        (jpg_dir / "image_00001.jpg").write_bytes(b"\xff\xd8")
        (extract_dir / "imagelabels.mat").write_bytes(b"mat")

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_oxford_flowers(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# download_food101
# ---------------------------------------------------------------------------


class TestDownloadFood101:
    def test_returns_images_directory(self, tmp_path):
        """download_food101 returns the food-101/images/ directory."""
        import shutil

        from vtsearch.datasets import downloader as dl_module

        tar_path = _make_food101_tar(tmp_path)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(tar_path), str(dest)),
            ),
        ):
            result = dl_module.download_food101(on_progress=lambda *a: None)

        assert result.name == "images"
        assert result.parent.name == "food-101"
        assert (result / "apple_pie").is_dir()
        assert (result / "sushi").is_dir()

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the images directory already exists, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        images_dir = tmp_path / "food-101" / "images" / "apple_pie"
        images_dir.mkdir(parents=True)
        (images_dir / "1.jpg").write_bytes(b"\xff\xd8")

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_food101(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# download_eurosat
# ---------------------------------------------------------------------------


class TestDownloadEurosat:
    def test_returns_extract_directory(self, tmp_path):
        """download_eurosat returns the EuroSAT_RGB/ directory."""
        import shutil

        from vtsearch.datasets import downloader as dl_module

        zip_path = _make_eurosat_zip(tmp_path)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(zip_path), str(dest)),
            ),
        ):
            result = dl_module.download_eurosat(on_progress=lambda *a: None)

        assert result.name == "EuroSAT_RGB"
        assert result.exists()
        assert (result / "Forest").is_dir()

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract directory already exists, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        extract_dir = tmp_path / "EuroSAT_RGB"
        (extract_dir / "Forest").mkdir(parents=True)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_eurosat(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# download_stanford_dogs
# ---------------------------------------------------------------------------


class TestDownloadStanfordDogs:
    def test_returns_images_directory(self, tmp_path):
        """download_stanford_dogs returns the stanford_dogs/Images/ directory."""
        import shutil

        from vtsearch.datasets import downloader as dl_module

        tar_path = _make_stanford_dogs_tar(tmp_path)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(tar_path), str(dest)),
            ),
        ):
            result = dl_module.download_stanford_dogs(on_progress=lambda *a: None)

        assert result.name == "Images"
        assert result.parent.name == "stanford_dogs"
        assert any(d.name.startswith("n02085620") for d in result.iterdir())

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the Images directory already exists, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        images_dir = tmp_path / "stanford_dogs" / "Images" / "n02085620-Chihuahua"
        images_dir.mkdir(parents=True)
        (images_dir / "img.jpg").write_bytes(b"\xff\xd8")

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_stanford_dogs(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# load_oxford_flowers_metadata
# ---------------------------------------------------------------------------


class TestLoadOxfordFlowersMetadata:
    def test_maps_labels_to_categories(self, tmp_path):
        """Reads imagelabels.mat and maps numeric labels to category names."""
        import scipy.io

        from vtsearch.datasets.loader import load_oxford_flowers_metadata

        jpg_dir = tmp_path / "jpg"
        jpg_dir.mkdir()
        for i in range(1, 5):
            (jpg_dir / f"image_{i:05d}.jpg").write_bytes(b"\xff\xd8")

        # Labels: 1-indexed. 4 images with labels [1, 2, 1, 2]
        labels = np.array([1, 2, 1, 2], dtype=np.int64)
        scipy.io.savemat(str(tmp_path / "imagelabels.mat"), {"labels": labels.reshape(1, -1)})

        categories = ["cat_a", "cat_b", "cat_c"]
        metadata = load_oxford_flowers_metadata(tmp_path, categories)

        assert len(metadata) == 4
        assert metadata["image_00001.jpg"]["category"] == "cat_a"
        assert metadata["image_00002.jpg"]["category"] == "cat_b"
        assert metadata["image_00003.jpg"]["category"] == "cat_a"


# ---------------------------------------------------------------------------
# load_demo_source — image (oxford_flowers_102, food101, eurosat, stanford_dogs)
# ---------------------------------------------------------------------------


class TestLoadDemoSourceOxfordFlowers:
    """ImageMediaType.load_demo_source with source='oxford_flowers_102'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
        return mock_emb

    def test_oxford_flowers_populates_clips(self, tmp_path):
        """load_demo_source with source='oxford_flowers_102' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        # Create stub image files.
        (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        fake_metadata = {
            "image_00001.jpg": {"category": "rose", "path": tmp_path / "img1.jpg"},
            "image_00002.jpg": {"category": "sunflower", "path": tmp_path / "img2.jpg"},
        }

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_oxford_flowers", return_value=tmp_path),
            patch.object(loader_module, "load_oxford_flowers_metadata", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="oxford_flowers_102",
                categories=["rose", "sunflower"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"rose", "sunflower"}

    def test_oxford_flowers_slice_is_applied(self, tmp_path):
        """slice_start/slice_end limits images per category."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        fake_metadata = {}
        for i in range(10):
            p = tmp_path / f"img_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
            fake_metadata[f"image_{i + 1:05d}.jpg"] = {"category": "rose", "path": p}

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_oxford_flowers", return_value=tmp_path),
            patch.object(loader_module, "load_oxford_flowers_metadata", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="oxford_flowers_102",
                categories=["rose"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 3


class TestLoadDemoSourceFood101:
    """ImageMediaType.load_demo_source with source='food101'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
        return mock_emb

    def test_food101_populates_clips(self, tmp_path):
        """load_demo_source with source='food101' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        fake_metadata = {
            "apple_pie/1.jpg": {"category": "apple_pie", "path": tmp_path / "img1.jpg"},
            "sushi/1.jpg": {"category": "sushi", "path": tmp_path / "img2.jpg"},
        }

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_food101", return_value=tmp_path),
            patch.object(loader_module, "load_image_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="food101",
                categories=["apple_pie", "sushi"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"apple_pie", "sushi"}


class TestLoadDemoSourceEurosat:
    """ImageMediaType.load_demo_source with source='eurosat'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
        return mock_emb

    def test_eurosat_populates_clips(self, tmp_path):
        """load_demo_source with source='eurosat' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        fake_metadata = {
            "Forest/Forest_00001.jpg": {"category": "Forest", "path": tmp_path / "img1.jpg"},
            "Residential/Residential_00001.jpg": {"category": "Residential", "path": tmp_path / "img2.jpg"},
        }

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_eurosat", return_value=tmp_path),
            patch.object(loader_module, "load_image_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="eurosat",
                categories=["Forest", "Residential"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"Forest", "Residential"}


class TestLoadDemoSourceStanfordDogs:
    """ImageMediaType.load_demo_source with source='stanford_dogs'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
        return mock_emb

    def test_stanford_dogs_populates_clips(self, tmp_path):
        """load_demo_source with source='stanford_dogs' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.media.image.media_type import ImageMediaType

        # Create folder structure: Images/n02085620-Chihuahua/img.jpg
        images_dir = tmp_path / "Images"
        breed_dir = images_dir / "n02085620-Chihuahua"
        breed_dir.mkdir(parents=True)
        for i in range(3):
            (breed_dir / f"n02085620_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_stanford_dogs", return_value=images_dir):
            mt.load_demo_source(
                source="stanford_dogs",
                categories=["Chihuahua"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 3
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"Chihuahua"}

    def test_stanford_dogs_slice_is_applied(self, tmp_path):
        """slice_start/slice_end limits images per breed."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.media.image.media_type import ImageMediaType

        images_dir = tmp_path / "Images"
        breed_dir = images_dir / "n02085620-Chihuahua"
        breed_dir.mkdir(parents=True)
        for i in range(10):
            (breed_dir / f"n02085620_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_stanford_dogs", return_value=images_dir):
            mt.load_demo_source(
                source="stanford_dogs",
                categories=["Chihuahua"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 3

    def test_unsupported_source_still_raises(self):
        """Non-existent sources still raise ValueError."""
        from vtsearch.media.image.media_type import ImageMediaType

        mt = ImageMediaType()
        with pytest.raises(ValueError, match="Unsupported image source"):
            mt.load_demo_source(
                source="unknown_source",
                categories=[],
                slice_start=0,
                slice_end=10,
                clips={},
                on_progress=lambda *a: None,
            )


# ---------------------------------------------------------------------------
# download_places365
# ---------------------------------------------------------------------------


class TestDownloadPlaces365:
    def test_returns_extract_directory(self, tmp_path):
        """download_places365 returns the places365/ directory with val_256/ and labels."""
        from vtsearch.datasets import downloader as dl_module

        val_tar_path = _make_places365_tar(tmp_path)
        labels_bytes = b"Places365_val_00000001.jpg 0\nPlaces365_val_00000002.jpg 1\n"
        filelist_tar_path = _make_places365_filelist_tar(tmp_path, labels_bytes)

        def fake_download(url, dest, size, cb):
            import shutil

            src = filelist_tar_path if "filelist" in url else val_tar_path
            shutil.copy(str(src), str(dest))

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            result = dl_module.download_places365(on_progress=lambda *a: None)

        assert result.name == "places365"
        assert (result / "val_256").is_dir()
        assert (result / "places365_val.txt").read_bytes() == labels_bytes
        assert any(p.name.startswith("Places365_val_") for p in (result / "val_256").iterdir())

    def test_cached_extraction_skips_download(self, tmp_path):
        """If val_256/ and the labels file already exist, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        extract_dir = tmp_path / "places365"
        val_dir = extract_dir / "val_256"
        val_dir.mkdir(parents=True)
        (val_dir / "Places365_val_00000001.jpg").write_bytes(b"\xff\xd8")
        (extract_dir / "places365_val.txt").write_text("Places365_val_00000001.jpg 0\n")

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_places365(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# load_places365_metadata
# ---------------------------------------------------------------------------


class TestLoadPlaces365Metadata:
    def test_maps_indices_to_categories(self, tmp_path):
        """Reads places365_val.txt and maps numeric indices to category names."""
        from vtsearch.datasets.loader import load_places365_metadata

        val_dir = tmp_path / "val_256"
        val_dir.mkdir()
        for i in range(1, 4):
            (val_dir / f"Places365_val_{i:08d}.jpg").write_bytes(b"\xff\xd8")

        labels_text = "Places365_val_00000001.jpg 0\nPlaces365_val_00000002.jpg 2\nPlaces365_val_00000003.jpg 1\n"
        (tmp_path / "places365_val.txt").write_text(labels_text)

        categories = ["airfield", "alley", "amphitheater"]
        metadata = load_places365_metadata(tmp_path, categories)

        assert len(metadata) == 3
        assert metadata["Places365_val_00000001.jpg"]["category"] == "airfield"
        assert metadata["Places365_val_00000002.jpg"]["category"] == "amphitheater"
        assert metadata["Places365_val_00000003.jpg"]["category"] == "alley"
        assert metadata["Places365_val_00000001.jpg"]["path"] == val_dir / "Places365_val_00000001.jpg"

    def test_out_of_range_indices_are_skipped(self, tmp_path):
        """Label indices outside the categories list are silently ignored."""
        from vtsearch.datasets.loader import load_places365_metadata

        (tmp_path / "val_256").mkdir()
        (tmp_path / "places365_val.txt").write_text(
            "Places365_val_00000001.jpg 0\n"
            "Places365_val_00000002.jpg 99\n"  # out of range
            "Places365_val_00000003.jpg -1\n"  # out of range
            "Places365_val_00000004.jpg bogus\n"  # unparseable
        )

        metadata = load_places365_metadata(tmp_path, ["airfield"])

        assert list(metadata) == ["Places365_val_00000001.jpg"]


# ---------------------------------------------------------------------------
# load_demo_source — places365
# ---------------------------------------------------------------------------


class TestLoadDemoSourcePlaces365:
    """ImageMediaType.load_demo_source with source='places365'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "siglip"
        mock_emb.media_type_id = "image"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
        return mock_emb

    def test_places365_populates_clips(self, tmp_path):
        """load_demo_source with source='places365' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        fake_metadata = {
            "Places365_val_00000001.jpg": {"category": "airfield", "path": tmp_path / "img1.jpg"},
            "Places365_val_00000002.jpg": {"category": "beach", "path": tmp_path / "img2.jpg"},
        }

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_places365", return_value=tmp_path),
            patch.object(loader_module, "load_places365_metadata", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="places365",
                categories=["airfield", "beach"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"airfield", "beach"}

    def test_places365_filters_by_requested_categories(self, tmp_path):
        """Metadata entries whose category is not requested are dropped."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        fake_metadata = {
            "Places365_val_00000001.jpg": {"category": "airfield", "path": tmp_path / "img1.jpg"},
            "Places365_val_00000002.jpg": {"category": "beach", "path": tmp_path / "img2.jpg"},
        }

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_places365", return_value=tmp_path),
            patch.object(loader_module, "load_places365_metadata", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="places365",
                categories=["airfield"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 1
        assert next(iter(clips.values()))["category"] == "airfield"

    def test_places365_slice_is_applied(self, tmp_path):
        """slice_start/slice_end limits images per category."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.image.media_type import ImageMediaType

        fake_metadata = {}
        for i in range(10):
            p = tmp_path / f"img_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
            fake_metadata[f"Places365_val_{i + 1:08d}.jpg"] = {"category": "airfield", "path": p}

        mt = ImageMediaType()
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_places365", return_value=tmp_path),
            patch.object(loader_module, "load_places365_metadata", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="places365",
                categories=["airfield"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 3


class TestPlaces365CategoriesList:
    """The hardcoded ``PLACES365_CATEGORIES`` matches the upstream label space."""

    def test_has_365_unique_entries(self):
        from vtsearch.media.image._demo_categories import PLACES365_CATEGORIES

        cats = PLACES365_CATEGORIES
        assert len(cats) == 365
        assert len(set(cats)) == 365
        # Spot-check entries near the index that previously had an off-by-one
        # error (swimming_pool/indoor vs swimming_pool/outdoor).
        assert cats[324] == "swimming_hole"
        assert cats[325] == "swimming_pool_indoor"
        assert cats[326] == "swimming_pool_outdoor"
        assert cats[0] == "airfield"
        assert cats[364] == "zen_garden"
