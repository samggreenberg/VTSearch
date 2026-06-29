"""End-to-end: serve archive-member media bytes via the playback routes.

Exercises the no-extraction path -- a media whose bytes live inside a tar shard
is streamed (with HTTP Range) straight out of the archive by the video/audio
routes, never extracting or fully buffering the member.
"""

from __future__ import annotations

import io
import tarfile

import numpy as np

from vtsearch.state import medias

VIDEO_BYTES = b"FAKE-MP4-VIDEO-MEMBER-CONTENT-" * 32
AUDIO_BYTES = b"FAKE-AAC-AUDIO-MEMBER-CONTENT-" * 16
_MEDIA_ID = 90001


def _make_shard(tmp_path):
    archive = tmp_path / "shard_000000.tar"
    with tarfile.open(archive, "w") as tf:
        for name, payload in (("clip.mp4", VIDEO_BYTES), ("clip.aac", AUDIO_BYTES)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return archive


def _inject(archive, member, media_type, filename):
    medias[_MEDIA_ID] = {
        "id": _MEDIA_ID,
        "media_type": media_type,
        "embedder": "clap",
        "embeddings": {"clap": np.zeros(512, dtype=np.float32)},
        "md5": "0" * 32,
        "filename": filename,
        "category": "custom",
        "media_bytes": None,
        "media_string": None,
        "duration": 0,
        "origin": {
            "importer": "local_archive_member",
            "params": {"archive_path": str(archive), "member": member, "media_type": media_type},
        },
        "archive_member": {"path": str(archive), "member": member},
    }


class TestArchiveMemberVideo:
    def test_full_video_streams_member_bytes(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.mp4", "video", "clip.mp4")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video")
        assert resp.status_code == 200
        assert resp.data == VIDEO_BYTES
        assert resp.headers["Accept-Ranges"] == "bytes"
        assert resp.headers["Content-Type"] == "video/mp4"

    def test_range_request_returns_partial(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.mp4", "video", "clip.mp4")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video", headers={"Range": "bytes=5-19"})
        assert resp.status_code == 206
        assert resp.data == VIDEO_BYTES[5:20]
        assert resp.headers["Content-Range"] == f"bytes 5-19/{len(VIDEO_BYTES)}"
        assert resp.headers["Content-Length"] == "15"

    def test_unsatisfiable_range_returns_416(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.mp4", "video", "clip.mp4")
        resp = client.get(
            f"/api/medias/{_MEDIA_ID}/video", headers={"Range": f"bytes={len(VIDEO_BYTES) + 10}-"}
        )
        assert resp.status_code == 416
        assert resp.headers["Content-Range"] == f"bytes */{len(VIDEO_BYTES)}"

    def test_missing_member_falls_through_to_404(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "not_present.mp4", "video", "not_present.mp4")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video")
        assert resp.status_code == 404


class TestArchiveMemberAudio:
    def test_audio_streams_member_bytes(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.aac", "audio", "clip.aac")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/audio")
        assert resp.status_code == 200
        assert resp.data == AUDIO_BYTES
        assert resp.headers["Accept-Ranges"] == "bytes"

    def test_audio_range_request_returns_partial(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.aac", "audio", "clip.aac")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/audio", headers={"Range": "bytes=3-10"})
        assert resp.status_code == 206
        assert resp.data == AUDIO_BYTES[3:11]
