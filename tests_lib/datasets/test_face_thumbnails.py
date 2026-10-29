"""Face crops carry a small grid thumbnail, however they arrived (#5465).

``image2face`` thumbnails each crop it makes (tested beside the converter).
Faces converted before it did have none, so every tile of the example picker
decoded and resized its crop on the request thread.  Two backfills close that:
the face type's ``ensure_thumbnail_bytes``, and the dataset save, which already
thumbnailed image media missing one.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.media.face.media_type import FaceMediaType
from vtscore.media.image.thumbnail import DEFAULT_MAX_DIM, make_image_thumbnail


def _crop_png(side: int = 900) -> bytes:
    rng = np.random.default_rng(3)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _face(**extra: Any) -> dict[str, Any]:
    return {
        "id": 1,
        "media_type": "face",
        "duration": 0,
        "file_size": 64,
        "md5": "abc",
        "embedder": "face",
        "embeddings": {"face": np.random.default_rng(7).standard_normal(16).astype(np.float32)},
        "filename": "group.png→group_face_1.png",
        "category": "test",
        "media_bytes": _crop_png(),
        "media_string": None,
        "media_path": None,
        "width": 900,
        "height": 900,
        "origin_name": "group.png→group_face_1.png",
        **extra,
    }


class TestEnsureThumbnailBytes:
    def test_builds_and_keeps_the_tile_the_route_would(self):
        media = _face()
        thumb = FaceMediaType().ensure_thumbnail_bytes(media)
        assert thumb == make_image_thumbnail(media["media_bytes"])[0]
        assert media["thumbnail_bytes"] == thumb
        with Image.open(io.BytesIO(thumb)) as img:
            assert max(img.size) == DEFAULT_MAX_DIM

    def test_an_existing_thumbnail_is_kept(self):
        media = _face(thumbnail_bytes=b"already")
        assert FaceMediaType().ensure_thumbnail_bytes(media) == b"already"

    def test_no_bytes_no_thumbnail(self):
        media = _face(media_bytes=None)
        assert FaceMediaType().ensure_thumbnail_bytes(media) is None
        assert "thumbnail_bytes" not in media


class TestSaveBackfillsTheThumbnail:
    def test_a_face_saved_without_one_reloads_with_one(self, tmp_path: Path):
        media = _face()
        pkl = tmp_path / "faces.pkl"
        pkl.write_bytes(export_dataset_to_file({1: media}, embedder="face", media_type="face"))
        loaded: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl, loaded)
        assert loaded[1]["thumbnail_bytes"] == make_image_thumbnail(media["media_bytes"])[0]
