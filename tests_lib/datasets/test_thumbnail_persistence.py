"""Library-tier tests for precomputed thumbnails through ingest and pickle.

Images precompute their grid/list thumbnail at ingest (matching the existing
audio/video behavior) so the request path never re-decodes the full-resolution
original on a cold tile fetch.  These tests lock two things:

1. ``ImageMediaType.load_media_data`` produces ``thumbnail_bytes`` and declares
   it in ``pickle_extra_fields``.
2. ``export_dataset_to_file`` writes every media type's ``pickle_extra_fields``
   (not just a hardcoded list), so ``thumbnail_bytes`` survives a pickle
   round-trip.  This previously silently dropped for audio/video too.
3. A **thin** load (the "Reference files in place" option / CLI ``--thin``)
   precomputes the same thumbnail as a full load while still leaving
   ``media_bytes`` empty, so referenced datasets browse at the same speed as
   copied ones.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vtscore.datasets.loader import export_dataset_to_file, load_dataset_from_folder
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.media.document.media_type import DocumentMediaType
from vtscore.media.image.media_type import ImageMediaType


def _png_bytes(size: tuple[int, int] = (1200, 800), color: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class TestImageLoadMediaDataThumbnail:
    def test_load_media_data_includes_thumbnail_bytes(self, tmp_path: Path):
        src = _png_bytes()
        path = tmp_path / "img.png"
        path.write_bytes(src)

        data = ImageMediaType().load_media_data(path, media_bytes=src)

        assert data["thumbnail_bytes"] is not None
        with Image.open(io.BytesIO(data["thumbnail_bytes"])) as thumb:
            assert max(thumb.size) <= 384
        # The precomputed thumbnail is smaller than the source it was bounded from.
        assert len(data["thumbnail_bytes"]) < len(src)

    def test_thumbnail_bytes_is_declared_for_pickling(self):
        assert "thumbnail_bytes" in ImageMediaType().pickle_extra_fields

    def test_undecodable_source_yields_no_thumbnail(self, tmp_path: Path):
        path = tmp_path / "bad.png"
        path.write_bytes(b"not an image")
        data = ImageMediaType().load_media_data(path, media_bytes=b"not an image")
        assert data["thumbnail_bytes"] is None


class TestThumbnailSurvivesPickleRoundTrip:
    def test_image_thumbnail_persists(self, tmp_path: Path):
        src = _png_bytes()
        thumb = ImageMediaType().load_media_data(tmp_path / "i.png", media_bytes=src)["thumbnail_bytes"]
        assert thumb is not None

        rng = np.random.default_rng(7)
        medias = {
            1: {
                "id": 1,
                "media_type": "image",
                "duration": 0,
                "file_size": len(src),
                "md5": "abc",
                "embedder": "siglip",
                "embeddings": {"siglip": rng.standard_normal(512).astype(np.float32)},
                "filename": "i.png",
                "category": "test",
                "media_bytes": src,
                "width": 1200,
                "height": 800,
                "thumbnail_bytes": thumb,
            }
        }

        container = export_dataset_to_file(medias, embedder="siglip", media_type="image")
        pkl = tmp_path / "ds.pkl"
        pkl.write_bytes(container)

        loaded: dict = {}
        load_dataset_from_pickle(pkl, loaded)

        assert loaded[1]["thumbnail_bytes"] == thumb


def _thin_load_images(folder: Path) -> dict[int, dict[str, Any]]:
    medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_folder(folder, "image", medias, thin=True)
    return medias


class TestThinLoadPrecomputesThumbnails:
    """A thin load keeps the payload out of memory but not the preview."""

    def test_thin_image_load_has_thumbnail_but_no_bytes(self, tmp_path: Path):
        (tmp_path / "a.png").write_bytes(_png_bytes())
        (tmp_path / "b.png").write_bytes(_png_bytes(color=(200, 30, 30)))

        medias = _thin_load_images(tmp_path)

        assert len(medias) == 2
        for media in medias.values():
            # Still a pure path reference: the point of a thin load.
            assert media["media_bytes"] is None
            assert media["media_string"] is None
            assert Path(media["media_path"]).exists()
            # ...but the browse/grid tile no longer decodes the original.
            assert media["thumbnail_bytes"]
            with Image.open(io.BytesIO(media["thumbnail_bytes"])) as thumb:
                assert max(thumb.size) <= 384

    def test_thin_load_records_dimensions(self, tmp_path: Path):
        (tmp_path / "a.png").write_bytes(_png_bytes(size=(1200, 800)))

        media = _thin_load_images(tmp_path)[1]

        assert (media["width"], media["height"]) == (1200, 800)

    def test_thin_and_full_loads_produce_the_same_thumbnail(self, tmp_path: Path):
        (tmp_path / "a.png").write_bytes(_png_bytes())

        thin = _thin_load_images(tmp_path)[1]
        full: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "image", full, thin=False)

        assert thin["thumbnail_bytes"] == full[1]["thumbnail_bytes"]

    def test_thin_thumbnail_survives_pickle_round_trip(self, tmp_path: Path):
        (tmp_path / "a.png").write_bytes(_png_bytes())
        medias = _thin_load_images(tmp_path)
        rng = np.random.default_rng(11)
        medias[1]["embeddings"] = {"siglip": rng.standard_normal(512).astype(np.float32)}

        pkl = tmp_path / "ds.pkl"
        pkl.write_bytes(export_dataset_to_file(medias, embedder="siglip", media_type="image"))
        loaded: dict = {}
        load_dataset_from_pickle(pkl, loaded)

        # Without an ingest-time thumbnail the save-side backfill can't help a
        # thin media (it needs ``media_bytes``), so this would be None before.
        assert loaded[1]["thumbnail_bytes"] == medias[1]["thumbnail_bytes"]

    def test_undecodable_source_thins_without_raising(self, tmp_path: Path):
        (tmp_path / "bad.png").write_bytes(b"not an image")

        media = _thin_load_images(tmp_path)[1]

        assert media["media_bytes"] is None
        assert media["thumbnail_bytes"] is None


class TestThinLoadSkipsTypesWithNoIngestArtifact:
    def test_document_thin_load_reads_nothing(self, tmp_path: Path):
        """Documents rasterise their preview on request, so a thin load stays pure."""
        assert DocumentMediaType().load_thin_media_data(tmp_path / "missing.pdf") == {}
