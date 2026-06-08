"""Library-tier tests for precomputed thumbnails through ingest and pickle.

Images precompute their grid/list thumbnail at ingest (matching the existing
audio/video behavior) so the request path never re-decodes the full-resolution
original on a cold tile fetch.  These tests lock two things:

1. ``ImageMediaType.load_media_data`` produces ``thumbnail_bytes`` and declares
   it in ``pickle_extra_fields``.
2. ``export_dataset_to_file`` writes every media type's ``pickle_extra_fields``
   (not just a hardcoded list), so ``thumbnail_bytes`` survives a pickle
   round-trip.  This previously silently dropped for audio/video too.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
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
                "embedding": rng.standard_normal(512).astype(np.float32),
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
