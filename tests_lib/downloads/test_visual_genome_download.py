"""Tests for the Visual Genome demo dataset: download, multi-label loading,
and ground-truth region storage.

Visual Genome is the only image demo that is multi-label (one image positive
for several categories) and that carries ground-truth bounding boxes, so these
tests cover the bits that differ from the folder-per-class image sources.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _make_vg_fixture(tmp_path: Path) -> Path:
    """Create a minimal extracted Visual Genome tree under ``tmp_path``.

    Layout mirrors the real download: ``VG_100K/<id>.jpg`` image files plus an
    ``objects.json`` with per-object names and pixel boxes.  Images are 100x100
    so pixel boxes normalize to clean fractions.
    """
    vg_dir = tmp_path / "visual_genome"
    img_dir = vg_dir / "VG_100K"
    img_dir.mkdir(parents=True)

    for image_id in (1, 2, 3, 4):
        Image.new("RGB", (100, 100), color=(image_id, image_id, image_id)).save(img_dir / f"{image_id}.jpg")

    objects = [
        {
            "image_id": 1,
            "objects": [
                {"object_id": 1, "x": 10, "y": 20, "w": 30, "h": 40, "names": ["man"]},
                {"object_id": 2, "x": 50, "y": 50, "w": 10, "h": 10, "names": ["dog"]},
            ],
        },
        {
            "image_id": 2,
            "objects": [{"object_id": 3, "x": 0, "y": 0, "w": 50, "h": 50, "names": ["banana"]}],
        },
        {
            "image_id": 3,
            "objects": [
                {"object_id": 4, "x": 5, "y": 5, "w": 20, "h": 20, "names": ["cars"]},  # plural -> car
                {"object_id": 5, "x": 0, "y": 0, "w": 90, "h": 90, "names": ["tree"]},
                {"object_id": 6, "x": 1, "y": 1, "w": 2, "h": 2, "names": ["unicorn"]},  # out of vocab
            ],
        },
        {
            "image_id": 4,
            "objects": [{"object_id": 7, "x": 0, "y": 0, "w": 10, "h": 10, "names": ["men"]}],  # irregular -> man
        },
    ]
    (vg_dir / "objects.json").write_text(json.dumps(objects))
    return vg_dir


def _make_mock_embedder():
    mock_emb = MagicMock()
    mock_emb.name = "siglip"
    mock_emb.media_type_id = "image"
    mock_emb._model = True
    mock_emb.embed_media = MagicMock(return_value=np.zeros(768))
    return mock_emb


_VOCAB = ["man", "dog", "banana", "car", "tree"]


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


class TestVgCategoryFor:
    def test_exact_match(self):
        from vtscore.media.image._demo_sources import _vg_category_for

        assert _vg_category_for("dog", frozenset(_VOCAB)) == "dog"

    def test_case_and_whitespace(self):
        from vtscore.media.image._demo_sources import _vg_category_for

        assert _vg_category_for("  Dog ", frozenset(_VOCAB)) == "dog"

    def test_regular_plural(self):
        from vtscore.media.image._demo_sources import _vg_category_for

        assert _vg_category_for("cars", frozenset(_VOCAB)) == "car"

    def test_irregular_plural(self):
        from vtscore.media.image._demo_sources import _vg_category_for

        assert _vg_category_for("men", frozenset(_VOCAB)) == "man"

    def test_out_of_vocab(self):
        from vtscore.media.image._demo_sources import _vg_category_for

        assert _vg_category_for("unicorn", frozenset(_VOCAB)) is None


# ---------------------------------------------------------------------------
# Collection (objects.json -> per-image records)
# ---------------------------------------------------------------------------


class TestCollectVisualGenome:
    def test_builds_multilabel_records(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image._demo_sources import _collect_visual_genome_files

        vg_dir = _make_vg_fixture(tmp_path)
        with patch.object(dl_module, "download_visual_genome", return_value=vg_dir):
            records = _collect_visual_genome_files(_VOCAB, (0, None, 0.0, None), lambda *a: None)

        # 4 images, all have at least one in-vocab object.
        assert len(records) == 4
        by_name = {path.name: (cats, regions) for path, cats, regions in records}

        # Image 1 is multi-label: man AND dog.
        assert set(by_name["1.jpg"][0]) == {"man", "dog"}
        # Image 3: plural "cars" maps to car, tree kept, unicorn dropped.
        assert set(by_name["3.jpg"][0]) == {"car", "tree"}
        # Image 4: irregular plural "men" maps to man.
        assert by_name["4.jpg"][0] == ["man"]

    def test_out_of_vocab_object_excluded_from_regions(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image._demo_sources import _collect_visual_genome_files

        vg_dir = _make_vg_fixture(tmp_path)
        with patch.object(dl_module, "download_visual_genome", return_value=vg_dir):
            records = _collect_visual_genome_files(_VOCAB, (0, None, 0.0, None), lambda *a: None)

        by_name = {path.name: regions for path, _cats, regions in records}
        # Image 3 has 3 objects but only 2 are in vocab -> 2 pixel regions.
        labels = {label for *_box, label in by_name["3.jpg"]}
        assert labels == {"car", "tree"}

    def test_category_filter_restricts_vocab(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image._demo_sources import _collect_visual_genome_files

        vg_dir = _make_vg_fixture(tmp_path)
        # Only "banana" in the active vocab -> only image 2 qualifies.
        with patch.object(dl_module, "download_visual_genome", return_value=vg_dir):
            records = _collect_visual_genome_files(["banana"], (0, None, 0.0, None), lambda *a: None)
        assert len(records) == 1
        assert records[0][0].name == "2.jpg"

    def test_flat_slice_is_applied(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image._demo_sources import _collect_visual_genome_files

        vg_dir = _make_vg_fixture(tmp_path)
        # Records are sorted by image id; a flat [1:3) slice yields images 2 and 3.
        with patch.object(dl_module, "download_visual_genome", return_value=vg_dir):
            records = _collect_visual_genome_files(_VOCAB, (1, 3, None, None), lambda *a: None)
        assert [path.name for path, _c, _r in records] == ["2.jpg", "3.jpg"]


# ---------------------------------------------------------------------------
# load_demo_source integration
# ---------------------------------------------------------------------------


class TestLoadDemoSourceVisualGenome:
    def test_populates_multilabel_clips_with_regions(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        vg_dir = _make_vg_fixture(tmp_path)
        mt = ImageMediaType()
        clips: dict = {}

        with patch.object(dl_module, "download_visual_genome", return_value=vg_dir):
            mt.load_demo_source(
                source="visual_genome",
                categories=_VOCAB,
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=_make_mock_embedder(),
            )

        assert len(clips) == 4
        by_name = {c["filename"]: c for c in clips.values()}

        man_dog = by_name["1.jpg"]
        # Multi-label list present; primary "category" is the first positive.
        assert set(man_dog["categories"]) == {"man", "dog"}
        assert man_dog["category"] == man_dog["categories"][0]

        # Region boxes are normalized to the 100x100 image and labelled.
        regions = {r["label"]: r["box"] for r in man_dog["regions"]}
        assert regions["man"] == [0.1, 0.2, 0.4, 0.6]
        assert regions["dog"] == [0.5, 0.5, 0.6, 0.6]

    def test_clips_carry_embeddings_and_origin(self, tmp_path):
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.image.media_type import ImageMediaType

        vg_dir = _make_vg_fixture(tmp_path)
        mt = ImageMediaType()
        clips: dict = {}

        with patch.object(dl_module, "download_visual_genome", return_value=vg_dir):
            mt.load_demo_source(
                source="visual_genome",
                categories=_VOCAB,
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=_make_mock_embedder(),
            )

        for c in clips.values():
            assert "siglip" in c["embeddings"]
            assert c["origin"] == {"importer": "demo", "params": {}}
            assert c["media_type"] == "image"


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class TestDownloadVisualGenome:
    def test_fetches_three_archives(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "IMAGE_DIR", tmp_path / "images"),
            patch.object(dl_module.core, "_download_and_extract") as mock_extract,
        ):
            result = dl_module.download_visual_genome(on_progress=lambda *a: None)

        assert result == tmp_path / "visual_genome"
        # One call each for images.zip, images2.zip, objects.json.zip.
        assert mock_extract.call_count == 3
        archive_names = {kwargs["archive_name"] for _args, kwargs in mock_extract.call_args_list}
        assert archive_names == {"vg_images.zip", "vg_images2.zip", "vg_objects.json.zip"}
