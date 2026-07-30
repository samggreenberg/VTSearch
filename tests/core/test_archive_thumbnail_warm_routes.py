"""App-tier coverage for the archive-member thumbnail warm-up (issue #2738).

The library-tier pass itself is covered in
``tests_lib/datasets/test_archive_thumbnail_warm.py``.  These tests pin the two
seams that only exist in the app: the thumbnail route serves a warmed thumbnail
without any request-time decode, and the load pipeline actually kicks the pass.
"""

from __future__ import annotations

import io
import tarfile
import threading

import numpy as np
from PIL import Image

from vtscore.datasets.thumbnail_warm import warm_archive_thumbnails
from vtscore.state.core import get_active_context, get_context
from vtsearch.state import medias

_MEDIA_ID = 91001


def _jpeg_bytes(size=(800, 600), color=(20, 140, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_shard(tmp_path, payload: bytes):
    archive = tmp_path / "shard_000000.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("photo.jpg")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return archive


def _inject(archive, payload: bytes):
    """Inject a media shaped exactly as ``local_archive_member`` leaves one."""
    medias[_MEDIA_ID] = {
        "id": _MEDIA_ID,
        "media_type": "image",
        "embedder": "siglip",
        "embeddings": {"siglip": np.zeros(8, dtype=np.float32)},
        "md5": "0" * 32,
        "filename": "photo.jpg",
        "category": "custom",
        "file_size": len(payload),
        "media_bytes": None,
        "media_string": None,
        "duration": 0,
        "origin": {
            "importer": "local_archive_member",
            "params": {"archive_path": str(archive), "member": "photo.jpg", "media_type": "image"},
        },
        "archive_member": {"path": str(archive), "member": "photo.jpg"},
    }


class TestThumbnailRouteAfterWarm:
    def test_warmed_thumbnail_is_served_without_a_request_time_decode(self, client, tmp_path, monkeypatch):
        payload = _jpeg_bytes()
        _inject(_make_shard(tmp_path, payload), payload)

        assert warm_archive_thumbnails(get_active_context()) == 1

        # With the thumbnail warmed, the route must take its streaming branch:
        # no full-resolution decode, no resize, no re-encode on the request
        # thread.  That is the whole point of the background pass.
        import vtscore.media.image.thumbnail as thumb_mod

        def fail(*_a, **_k):
            raise AssertionError("request-time thumbnail generation on a warmed media")

        monkeypatch.setattr(thumb_mod, "make_image_thumbnail", fail)

        resp = client.get(f"/api/medias/{_MEDIA_ID}/thumbnail")
        assert resp.status_code == 200
        assert resp.data == medias[_MEDIA_ID]["thumbnail_bytes"]
        with Image.open(io.BytesIO(resp.data)) as img:
            assert max(img.size) <= 384

    def test_unwarmed_media_still_serves_a_thumbnail(self, client, tmp_path):
        payload = _jpeg_bytes()
        _inject(_make_shard(tmp_path, payload), payload)

        # Partial coverage must degrade gracefully: a media the pass has not
        # reached keeps the per-request fallback it always had.
        resp = client.get(f"/api/medias/{_MEDIA_ID}/thumbnail")
        assert resp.status_code == 200
        with Image.open(io.BytesIO(resp.data)) as img:
            assert max(img.size) <= 384


class TestLoadPipelineKick:
    def test_load_pipeline_kicks_the_warm_up(self, tmp_path, monkeypatch):
        """The kick is wired into the shared load path, so import *and* reload get it."""
        from vtscore.datasets import load_pipeline

        seen: list[str] = []
        kicked = threading.Event()

        def record(ctx):
            seen.append(ctx.dataset_id)
            kicked.set()

        monkeypatch.setattr(load_pipeline, "start_archive_thumbnail_warm", record)

        payload = _jpeg_bytes()
        archive = _make_shard(tmp_path, payload)

        def load_fn(target):
            target[1] = {
                "id": 1,
                "media_type": "image",
                "embedder": "siglip",
                "embeddings": {"siglip": np.zeros(8, dtype=np.float32)},
                "md5": "1" * 32,
                "filename": "photo.jpg",
                "category": "custom",
                "file_size": len(payload),
                "media_bytes": None,
                "media_string": None,
                "duration": 0,
                "archive_member": {"path": str(archive), "member": "photo.jpg"},
            }

        load_pipeline._run_origin_load_in_background(
            load_fn,
            origin={"importer": "local_archive_member", "params": {"archive_path": str(archive)}},
            name="warm-kick-test",
            media_type="image",
            embedder="siglip",
        )
        assert kicked.wait(timeout=60), "load completed without kicking the thumbnail warm-up"
        # Kicked exactly once, and *after* the registry migration renamed the
        # context off its ``_loading_…`` task id -- so the id the pass records
        # (and re-resolves through ``get_context`` to detect an unload) is the
        # durable dataset id, not one that stops resolving a moment later.
        assert len(seen) == 1
        assert seen[0] and not seen[0].startswith("_loading_")
        assert get_context(seen[0]) is not None
