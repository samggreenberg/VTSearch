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
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video", headers={"Range": f"bytes={len(VIDEO_BYTES) + 10}-"})
        assert resp.status_code == 416
        assert resp.headers["Content-Range"] == f"bytes */{len(VIDEO_BYTES)}"

    def test_suffix_range_serves_only_the_tail(self, client, tmp_path):
        # "bytes=-N" asks for the last N bytes (how some players find an MP4
        # moov atom); serving the whole member would defeat the streamed read.
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.mp4", "video", "clip.mp4")
        total = len(VIDEO_BYTES)
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video", headers={"Range": "bytes=-40"})
        assert resp.status_code == 206
        assert resp.data == VIDEO_BYTES[-40:]
        assert resp.headers["Content-Range"] == f"bytes {total - 40}-{total - 1}/{total}"
        assert resp.headers["Content-Length"] == "40"

    def test_oversized_suffix_range_serves_whole_member(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.mp4", "video", "clip.mp4")
        total = len(VIDEO_BYTES)
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video", headers={"Range": f"bytes=-{total + 100}"})
        assert resp.status_code == 206
        assert resp.data == VIDEO_BYTES
        assert resp.headers["Content-Range"] == f"bytes 0-{total - 1}/{total}"

    def test_zero_length_suffix_range_returns_416(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.mp4", "video", "clip.mp4")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/video", headers={"Range": "bytes=-0"})
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


class TestArchiveMemberExampleSortOrigin:
    """Cross-dataset Find re-supplies the member's precomputed vector (no path)."""

    def _make_manifest(self, tmp_path, archive, member):
        rng = np.random.default_rng(5)
        manifest = tmp_path / "manifest.npz"
        np.savez(
            manifest,
            vectors=rng.standard_normal((1, 512)).astype(np.float32),
            members=np.array([member]),
            archives=np.array(str(archive)),
            embedder_name=np.array("clap"),
        )
        return manifest

    def _origin(self, tmp_path):
        from vtscore.datasets.importers.local_archive_member import IMPORTER

        archive = _make_shard(tmp_path)
        manifest = self._make_manifest(tmp_path, archive, "clip.aac")
        scratch: dict[int, dict] = {}
        IMPORTER.run({"manifest": str(manifest), "media_type": "audio"}, scratch)
        return next(iter(scratch.values()))["origin"]

    def test_origin_sort_uses_precomputed_vector(self, client, tmp_path):
        origin = self._origin(tmp_path)
        resp = client.post(
            "/api/example-sort-origin",
            json={"origin": origin, "key": "clip.aac"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data

    def test_crop_on_archive_member_returns_400(self, client, tmp_path):
        origin = self._origin(tmp_path)
        resp = client.post(
            "/api/example-sort-origin",
            json={"origin": origin, "key": "clip.aac", "crop_params": {"start": 0, "end": 1}},
        )
        assert resp.status_code == 400


class TestArchiveMemberAudioMimetype:
    """The /audio route picks an audio Content-Type per the member's container."""

    def test_aac_member_serves_audio_aac(self, client, tmp_path):
        archive = _make_shard(tmp_path)
        _inject(archive, "clip.aac", "audio", "clip.aac")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/audio")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/aac"

    def test_mp4_audio_member_serves_audio_mp4(self, client, tmp_path):
        # multivent-raw audio rides in an MP4 container; mimetypes would call
        # that video/mp4, but the <audio> element needs an audio/* type.
        archive = _make_shard(tmp_path)
        with tarfile.open(archive, "a") as tf:
            info = tarfile.TarInfo("clip_audio.mp4")
            info.size = len(AUDIO_BYTES)
            tf.addfile(info, io.BytesIO(AUDIO_BYTES))
        _inject(archive, "clip_audio.mp4", "audio", "clip_audio.mp4")
        resp = client.get(f"/api/medias/{_MEDIA_ID}/audio")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/mp4"
