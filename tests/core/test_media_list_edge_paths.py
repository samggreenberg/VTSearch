"""Edge-path coverage for ``vtsearch/routes/media/list.py``.

The byte-serving, batch-metadata, vote, and add-to-pile routes carry a long
tail of branches that the happy-path suites in ``test_medias.py`` /
``test_votes.py`` never reach: unregistered media types, lazy path-backed
resolution, video transcoding, range-request corner cases, and the
persistence-failure fallbacks on the vote routes.

These tests drive those branches through the public API (``client``),
injecting purpose-built media dicts straight into the active context's
``medias`` the way ``test_archive_member_routes.py`` does.  A fresh context
is created per test by ``conftest.reset_state`` (it swaps in a brand-new
``DatasetContext`` and drops the old one), so injected high-id medias never
leak into later tests and need no explicit cleanup.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import app as app_module
from vtsearch.routes.media import list as media_list
from vtsearch.state import medias

# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

_ID = 70000


def _next_id() -> int:
    global _ID
    _ID += 1
    return _ID


def _inject(**overrides) -> int:
    """Insert a minimal media dict into the active context and return its id."""
    mid = _next_id()
    media = {
        "id": mid,
        "media_type": "audio",
        "embedder": "clap",
        "embeddings": {"clap": np.zeros(512, dtype=np.float32)},
        "md5": f"{mid:032x}",
        "filename": f"media_{mid}.wav",
        "category": "custom",
        "media_bytes": b"",
        "media_string": None,
        "duration": 0,
        "origin": {"importer": "test", "params": {}},
    }
    media.update(overrides)
    medias[mid] = media
    return mid


def _png_bytes(color=(200, 40, 40), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _real_avi_bytes() -> bytes:
    """Encode a tiny, genuinely-decodable AVI so the transcode path succeeds."""
    import os
    import tempfile

    import cv2

    d = tempfile.mkdtemp()
    path = os.path.join(d, "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"MJPG"), 5.0, (32, 32))
    try:
        for i in range(5):
            writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
    finally:
        writer.release()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    os.rmdir(d)
    return data


def _inject_archive_member(tmp_path, payload, member, media_type, filename):
    """Inject a media whose bytes live inside a tar shard; return its id."""
    import tarfile

    mid = _next_id()
    archive = tmp_path / f"shard_{mid}.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    medias[mid] = {
        "id": mid,
        "media_type": media_type,
        "embedder": "clap",
        "embeddings": {"clap": np.zeros(512, dtype=np.float32)},
        "md5": f"{mid:032x}",
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
    return mid


# ---------------------------------------------------------------------------
# _parse_region_box (unit) -- the schema coerces region_box to floats before
# the route sees it, so the coercion-failure branch is only reachable directly.
# ---------------------------------------------------------------------------


class TestParseRegionBox:
    def test_none_returns_none(self):
        assert media_list._parse_region_box(None) is None

    def test_valid_four_tuple(self):
        assert media_list._parse_region_box([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="4-element"):
            media_list._parse_region_box([0.1, 0.2, 0.3])

    def test_non_list_rejected(self):
        with pytest.raises(ValueError, match="4-element"):
            media_list._parse_region_box("0,0,1,1")

    def test_non_numeric_entries_rejected(self):
        # Hits the ``float(v)`` TypeError/ValueError branch that the API's
        # marshmallow schema forecloses (it coerces to floats first).
        with pytest.raises(ValueError, match="must be numbers"):
            media_list._parse_region_box(["a", "b", "c", "d"])

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            media_list._parse_region_box([0.0, 0.0, 1.5, 0.0])


# ---------------------------------------------------------------------------
# Thumbnail region query parsing
# ---------------------------------------------------------------------------


class TestThumbnailRegionQuery:
    def test_four_non_numeric_parts_fall_back_to_full(self, client):
        # Four comma parts that aren't floats exercise the float() ValueError
        # branch in _parse_region_query (distinct from the wrong-arity path,
        # which the existing ``region=not,a,box`` test already covers).
        full = client.get("/api/medias/1/thumbnail")
        bad = client.get("/api/medias/1/thumbnail?region=a,b,c,d")
        assert bad.status_code == 200
        assert bad.data == full.data


# ---------------------------------------------------------------------------
# Byte resolution fallbacks (_resolve_bytes) via /audio
# ---------------------------------------------------------------------------


class TestResolveBytesFallbacks:
    def test_unregistered_type_serves_inline_bytes(self, client):
        # media_type isn't a registered MediaType -> _resolve_bytes falls back
        # to the inline media_bytes path.
        mid = _inject(media_type="mystery_type", media_bytes=b"RIFFinline-audio")
        resp = client.get(f"/api/medias/{mid}/audio")
        assert resp.status_code == 200
        assert resp.data == b"RIFFinline-audio"

    def test_unregistered_type_reads_local_path(self, client, tmp_path):
        p = tmp_path / "clip.bin"
        p.write_bytes(b"ON-DISK-AUDIO-BYTES")
        mid = _inject(media_type="mystery_type", media_bytes=None, media_path=str(p))
        resp = client.get(f"/api/medias/{mid}/audio")
        assert resp.status_code == 200
        assert resp.data == b"ON-DISK-AUDIO-BYTES"

    def test_missing_bytes_returns_404(self, client):
        # Registered audio type but no bytes / path / archive member available.
        mid = _inject(media_type="audio", media_bytes=None)
        resp = client.get(f"/api/medias/{mid}/audio")
        assert resp.status_code == 404

    def test_unregistered_type_no_bytes_returns_404(self, client):
        # Unregistered type, no inline bytes, no path -> the _resolve_bytes
        # fallback exhausts every branch and returns None -> 404.
        mid = _inject(media_type="mystery_type", media_bytes=None)
        resp = client.get(f"/api/medias/{mid}/audio")
        assert resp.status_code == 404

    def test_unregistered_type_nonexistent_path_returns_404(self, client, tmp_path):
        # media_path is set but the file is gone -> the exists() branch is
        # false and _resolve_bytes returns None -> 404.
        mid = _inject(media_type="mystery_type", media_bytes=None, media_path=str(tmp_path / "gone.bin"))
        resp = client.get(f"/api/medias/{mid}/audio")
        assert resp.status_code == 404

    def test_unknown_extension_defaults_audio_mimetype(self, client):
        # A filename mimetypes can't classify drives _audio_member_mimetype's
        # "no guess -> audio/wav" default.
        mid = _inject(filename="soundfile_no_extension", media_bytes=b"RIFFxyz")
        resp = client.get(f"/api/medias/{mid}/audio")
        assert resp.status_code == 200
        assert resp.content_type == "audio/wav"


# ---------------------------------------------------------------------------
# Video route: mimetype selection, range handling, transcoding
# ---------------------------------------------------------------------------


class TestVideoRoute:
    def test_not_a_video_returns_400(self, client):
        resp = client.get("/api/medias/1/video")  # media 1 is audio
        assert resp.status_code == 400

    def test_mp4_full_response(self, client):
        data = b"MP4-BODY-" * 20
        mid = _inject(media_type="video", filename="clip.mp4", media_bytes=data)
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 200
        assert resp.data == data
        assert resp.content_type == "video/mp4"
        assert resp.headers["Accept-Ranges"] == "bytes"

    def test_webm_mimetype(self, client):
        mid = _inject(media_type="video", filename="clip.webm", media_bytes=b"WEBMDATA123")
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 200
        assert resp.content_type == "video/webm"

    def test_ogg_mimetype(self, client):
        mid = _inject(media_type="video", filename="clip.ogv", media_bytes=b"OGGDATA123")
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 200
        assert resp.content_type == "video/ogg"

    def test_range_request_partial(self, client):
        data = b"0123456789ABCDEF" * 4
        mid = _inject(media_type="video", filename="clip.mp4", media_bytes=data)
        resp = client.get(f"/api/medias/{mid}/video", headers={"Range": "bytes=5-10"})
        assert resp.status_code == 206
        assert resp.data == data[5:11]
        assert resp.headers["Content-Range"] == f"bytes 5-10/{len(data)}"

    def test_malformed_range_serves_whole(self, client):
        data = b"MP4-BODY-" * 8
        mid = _inject(media_type="video", filename="clip.mp4", media_bytes=data)
        resp = client.get(f"/api/medias/{mid}/video", headers={"Range": "bytes=abc-def"})
        # An unparseable Range degrades to the whole payload, served as a 206
        # spanning the full 0..len-1 window (the except-branch default).
        assert resp.status_code == 206
        assert resp.data == data
        assert resp.headers["Content-Range"] == f"bytes 0-{len(data) - 1}/{len(data)}"

    def test_unsatisfiable_range_returns_416(self, client):
        data = b"short"
        mid = _inject(media_type="video", filename="clip.mp4", media_bytes=data)
        resp = client.get(f"/api/medias/{mid}/video", headers={"Range": f"bytes={len(data) + 5}-"})
        assert resp.status_code == 416
        assert resp.headers["Content-Range"] == f"bytes */{len(data)}"

    def test_missing_bytes_returns_404(self, client):
        mid = _inject(media_type="video", filename="clip.mp4", media_bytes=None)
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 404

    def test_cached_transcode_is_reused(self, client):
        # A media carrying a memoised _transcoded_mp4 is served straight from
        # the cache without touching ffmpeg/opencv again.
        mp4 = b"CACHED-TRANSCODED-MP4-BODY"
        mid = _inject(
            media_type="video",
            filename="clip.avi",
            media_bytes=b"raw-avi",
            _transcoded_mp4=mp4,
        )
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 200
        assert resp.data == mp4
        assert resp.content_type == "video/mp4"

    def test_transcode_missing_bytes_returns_404(self, client):
        # Non-browser ext with no resolvable bytes -> 404 from the transcode path.
        mid = _inject(media_type="video", filename="clip.avi", media_bytes=None)
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 404

    def test_untranscodable_avi_returns_415(self, client):
        # Garbage .avi bytes: ffmpeg and OpenCV both fail to decode, so the
        # route reports 415 rather than a broken stream.
        mid = _inject(media_type="video", filename="clip.avi", media_bytes=b"not-a-real-avi" * 8)
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 415

    def test_real_avi_is_transcoded_to_mp4(self, client):
        # A genuinely-decodable non-browser video is transcoded to MP4 and
        # served (exercising the ffmpeg success path), then memoised so the
        # second request comes from the cache.
        mid = _inject(media_type="video", filename="clip.avi", media_bytes=_real_avi_bytes())
        resp = client.get(f"/api/medias/{mid}/video")
        assert resp.status_code == 200
        assert resp.content_type == "video/mp4"
        assert resp.data[4:8] == b"ftyp"  # MP4 container signature
        assert medias[mid].get("_transcoded_mp4")  # memoised for reuse

    def test_archive_member_malformed_range_serves_whole(self, client, tmp_path):
        # A non-numeric Range against an archive-streamed member degrades to the
        # full payload (the _send_streamed_range except-branch).
        payload = b"ARCHIVE-VIDEO-MEMBER-" * 8
        mid = _inject_archive_member(tmp_path, payload, "clip.mp4", "video", "clip.mp4")
        resp = client.get(f"/api/medias/{mid}/video", headers={"Range": "bytes=xx-yy"})
        assert resp.status_code == 206
        assert resp.data == payload
        assert resp.headers["Content-Range"] == f"bytes 0-{len(payload) - 1}/{len(payload)}"


# ---------------------------------------------------------------------------
# Image route + display-image resolution
# ---------------------------------------------------------------------------


class TestImageRoute:
    def test_png_image_served_with_png_mimetype(self, client):
        png = _png_bytes()
        mid = _inject(media_type="image", filename="pic.png", media_bytes=png)
        resp = client.get(f"/api/medias/{mid}/image")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        assert resp.data == png

    def test_unknown_extension_defaults_jpeg(self, client):
        png = _png_bytes(color=(10, 120, 30))
        mid = _inject(media_type="image", filename="pic.qqq", media_bytes=png)
        resp = client.get(f"/api/medias/{mid}/image")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"

    def test_no_filename_defaults_jpeg(self, client):
        png = _png_bytes(color=(30, 30, 120))
        mid = _inject(media_type="image", filename="", media_bytes=png)
        resp = client.get(f"/api/medias/{mid}/image")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"

    def test_unregistered_non_image_type_returns_400(self, client):
        # A non-image type with no registered MediaType (so no image_response
        # delegate) has no image to serve.
        mid = _inject(media_type="mystery_type", media_bytes=b"whatever")
        resp = client.get(f"/api/medias/{mid}/image")
        assert resp.status_code == 400

    def test_image_missing_bytes_returns_404(self, client):
        mid = _inject(media_type="image", filename="pic.png", media_bytes=None)
        resp = client.get(f"/api/medias/{mid}/image")
        assert resp.status_code == 404

    def test_non_image_type_with_no_extractable_frame_returns_400(self, client):
        # A registered non-image type (video) whose image_response delegate
        # yields nothing for undecodable bytes -> "no image available" 400.
        mid = _inject(media_type="video", filename="clip.mp4", media_bytes=b"not-a-real-video" * 4)
        resp = client.get(f"/api/medias/{mid}/image")
        assert resp.status_code == 400

    def test_thumbnail_falls_back_when_undecodable(self, client):
        # An "image" whose bytes aren't a real image: make_image_thumbnail
        # returns None, so the route falls back to image_thumbnail_response,
        # which serves the original bytes rather than erroring.
        mid = _inject(media_type="image", filename="pic.png", media_bytes=b"NOT-AN-IMAGE" * 4)
        resp = client.get(f"/api/medias/{mid}/thumbnail")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Text / paragraph route + string resolution
# ---------------------------------------------------------------------------


class TestParagraphRoute:
    def test_text_media_returns_content_and_counts(self, client):
        mid = _inject(media_type="text", media_string="alpha beta gamma delta", filename="a.txt")
        resp = client.get(f"/api/medias/{mid}/paragraph")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["content"] == "alpha beta gamma delta"
        assert body["word_count"] == 4
        assert body["character_count"] == len("alpha beta gamma delta")

    def test_text_content_lazy_loaded_from_path(self, client, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("  lazily loaded text  ", encoding="utf-8")
        mid = _inject(media_type="text", media_string=None, media_path=str(p), filename="doc.txt")
        resp = client.get(f"/api/medias/{mid}/paragraph")
        assert resp.status_code == 200
        assert resp.get_json()["content"] == "lazily loaded text"

    def test_non_text_media_returns_400(self, client):
        resp = client.get("/api/medias/1/paragraph")  # media 1 is audio
        assert resp.status_code == 400

    def test_text_media_without_content_returns_404(self, client):
        mid = _inject(media_type="text", media_string=None, filename="empty.txt")
        resp = client.get(f"/api/medias/{mid}/paragraph")
        assert resp.status_code == 404

    def test_text_media_path_missing_file_returns_404(self, client, tmp_path):
        # media_string is absent and media_path points at a file that isn't
        # there -> _resolve_string exhausts its branches and returns None.
        missing = tmp_path / "gone.txt"
        mid = _inject(media_type="text", media_string=None, media_path=str(missing), filename="gone.txt")
        resp = client.get(f"/api/medias/{mid}/paragraph")
        assert resp.status_code == 404

    def test_text_route_alias(self, client):
        mid = _inject(media_type="text", media_string="via the /text alias", filename="a.txt")
        resp = client.get(f"/api/medias/{mid}/text")
        assert resp.status_code == 200
        assert resp.get_json()["content"] == "via the /text alias"


# ---------------------------------------------------------------------------
# Generic /media route
# ---------------------------------------------------------------------------


class TestGenericMediaRoute:
    def test_audio_serves_binary(self, client):
        resp = client.get("/api/medias/1/media")
        assert resp.status_code == 200
        assert resp.data  # binary body

    def test_text_serves_json(self, client):
        mid = _inject(media_type="text", media_string="generic text body", filename="a.txt")
        resp = client.get(f"/api/medias/{mid}/media")
        assert resp.status_code == 200
        assert resp.content_type.startswith("application/json")

    def test_unsupported_type_returns_400(self, client):
        mid = _inject(media_type="mystery_type", media_bytes=b"x")
        resp = client.get(f"/api/medias/{mid}/media")
        assert resp.status_code == 400

    def test_missing_media_returns_404(self, client):
        resp = client.get("/api/medias/999999/media")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Batch metadata edge branches
# ---------------------------------------------------------------------------


class TestBatchMetadataBranches:
    def test_unregistered_type_custom_metadata_and_optional_fields(self, client):
        # A single media exercises four otherwise-missed batch branches:
        #  - unregistered media_type -> display_metadata KeyError -> custom={}
        #  - importer custom_metadata merged onto the payload
        #  - origin_name ABSENT (the false side of the origin_name branch)
        #  - description present
        mid = _inject(
            media_type="mystery_type",
            custom_metadata={"source": "unit-test", "n": 3},
            description="a described item",
        )
        assert "origin_name" not in medias[mid]
        resp = client.post("/api/medias/batch", json={"ids": [mid]})
        assert resp.status_code == 200
        (item,) = resp.get_json()
        assert item["custom_metadata"]["source"] == "unit-test"
        assert item["description"] == "a described item"
        assert "origin_name" not in item

    def test_sliced_audio_suppresses_playback_window(self, client):
        # The audio clipper serves the already-sliced clip bytes (a short WAV in
        # its own 0-based timeline) but stamps the original absolute offsets.
        # The top-level clip_start/clip_end are a *seek window into the whole
        # served file*, so emitting the absolute offsets here would make the
        # player seek past the end of the short clip and play silence. They must
        # be suppressed for byte-sliced audio; clip_index (provenance) stays.
        mid = _inject(clip_start=120.0, clip_end=125.0, clip_index=2)
        resp = client.post("/api/medias/batch", json={"ids": [mid]})
        assert resp.status_code == 200
        (item,) = resp.get_json()
        assert "clip_start" not in item
        assert "clip_end" not in item
        assert item["clip_index"] == 2
        # The offsets still reach the UI as provenance via custom_metadata.
        assert item["custom_metadata"]["Clip Start"] == 120.0
        assert item["custom_metadata"]["Clip End"] == 125.0

    def test_video_keeps_playback_window(self, client):
        # Video clips share the parent's bytes and the player seeks within
        # [clip_start, clip_end], so the window must survive for video.
        mid = _inject(media_type="video", filename="clip.mp4", clip_start=1.5, clip_end=3.0, clip_index=2)
        resp = client.post("/api/medias/batch", json={"ids": [mid]})
        assert resp.status_code == 200
        (item,) = resp.get_json()
        assert item["clip_start"] == 1.5
        assert item["clip_end"] == 3.0
        assert item["clip_index"] == 2

    def test_archive_member_audio_keeps_playback_window(self, client, tmp_path):
        # A windowed archive-member (AAC/MP4) import serves the *whole* member,
        # so the player must seek to clip_start and loop within the window; the
        # window has to survive for these.
        mid = _inject_archive_member(tmp_path, b"RIFFfake", "a.wav", "audio", "a.wav")
        medias[mid].update({"clip_start": 4.0, "clip_end": 9.0})
        resp = client.post("/api/medias/batch", json={"ids": [mid]})
        assert resp.status_code == 200
        (item,) = resp.get_json()
        assert item["clip_start"] == 4.0
        assert item["clip_end"] == 9.0


# ---------------------------------------------------------------------------
# Vote route edge branches
# ---------------------------------------------------------------------------


class TestVoteEdgePaths:
    def test_out_of_range_region_box_rejected(self, client):
        # region_box passes the schema (floats) but fails the [0,1] range check
        # in _parse_region_box -> 400.
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "region_box": [0.0, 0.0, 1.5, 0.0]},
        )
        assert resp.status_code == 400

    def test_region_box_on_non_good_target_rejected(self, client):
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "bad", "region_box": [0.1, 0.1, 0.5, 0.5]},
        )
        assert resp.status_code == 400
        assert "good" in resp.get_json()["message"]

    def test_labelset_source_failure_is_swallowed(self, client, monkeypatch):
        # A failure scheduling the debounced labelset-source push is logged but
        # must not fail the vote (the detector-store write already succeeded).
        import vtscore.labels.sync as label_sync

        def _boom():
            raise RuntimeError("scheduling failed")

        monkeypatch.setattr(label_sync, "sync_to_labelset_source", _boom)
        resp = client.post("/api/medias/1/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert resp.get_json()["state"] == "good"


class TestVoteBulkEdgePaths:
    def test_detector_sync_failure_returns_500(self, client, monkeypatch):
        import vtscore.detectors.label_sync as label_sync

        def _boom():
            raise RuntimeError("detector write failed")

        monkeypatch.setattr(label_sync, "sync_labels_to_loaded_detector", _boom)
        resp = client.post("/api/medias/vote-bulk", json={"target": "good", "ids": [1, 2]})
        assert resp.status_code == 500

    def test_labelset_source_failure_is_swallowed(self, client, monkeypatch):
        import vtscore.labels.sync as label_sync

        def _boom():
            raise RuntimeError("scheduling failed")

        monkeypatch.setattr(label_sync, "sync_to_labelset_source", _boom)
        resp = client.post("/api/medias/vote-bulk", json={"target": "good", "ids": [1, 2]})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True


# ---------------------------------------------------------------------------
# add-to-pile embedder resolution / thumbnail / failure branches
# ---------------------------------------------------------------------------


class TestAddToPileEdgePaths:
    def _single_audio_dataset(self, embedder_name):
        """Replace the loaded dataset with one audio media using *embedder_name*."""
        saved = dict(medias)
        medias.clear()
        medias[1] = {
            "id": 1,
            "media_type": "audio",
            "embedder": embedder_name,
            "embeddings": {"clap": np.zeros(512, dtype=np.float32)},
            "md5": "f" * 32,
            "filename": "seed.wav",
            "category": "test",
            "media_bytes": b"seed-bytes",
            "origin": {"importer": "test", "params": {}},
            "origin_name": "seed.wav",
        }
        return saved

    def test_empty_embedder_name_falls_back_to_type_default(self, client):
        # dataset_embedder_name is "" -> _resolve_embedder skips get_embedder and
        # picks the first embedder registered for the media type.
        saved = self._single_audio_dataset("")
        try:
            wav = app_module.generate_wav(4321, 0.2)
            resp = client.post(
                "/api/medias/add-to-pile",
                data={"label": "good", "file": (io.BytesIO(wav), "fresh.wav")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 201
            assert resp.get_json()["is_new"] is True
        finally:
            medias.clear()
            medias.update(saved)

    def test_unknown_embedder_name_falls_back_to_type_default(self, client):
        # dataset_embedder_name resolves to a KeyError in get_embedder -> the
        # handler still recovers via the media-type default embedder.
        saved = self._single_audio_dataset("no_such_embedder")
        try:
            wav = app_module.generate_wav(4322, 0.2)
            resp = client.post(
                "/api/medias/add-to-pile",
                data={"label": "good", "file": (io.BytesIO(wav), "fresh2.wav")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 201
        finally:
            medias.clear()
            medias.update(saved)

    def test_no_embedder_available_returns_400(self, client):
        # A dataset whose media type has no registered embedder and no usable
        # recorded embedder -> the upload of a fresh file 400s.
        saved = dict(medias)
        medias.clear()
        medias[1] = {
            "id": 1,
            "media_type": "mystery_type",
            "embedder": "",
            "embeddings": {},
            "md5": "e" * 32,
            "filename": "seed.bin",
            "category": "test",
            "media_bytes": b"seed",
            "origin": {"importer": "test", "params": {}},
        }
        try:
            resp = client.post(
                "/api/medias/add-to-pile",
                data={"label": "good", "file": (io.BytesIO(b"brand-new-bytes"), "fresh.bin")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 400
            assert "embedder" in resp.get_json()["message"]
        finally:
            medias.clear()
            medias.update(saved)

    def test_embed_failure_returns_400(self, client, monkeypatch):
        # The resolved embedder returns None for the upload -> 400.
        from vtscore.media import embedders_for_type

        audio_embedder = embedders_for_type("audio")[0]
        monkeypatch.setattr(audio_embedder, "embed_media", lambda *_a, **_k: None)
        wav = app_module.generate_wav(4323, 0.2)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(wav), "fresh3.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "embed" in resp.get_json()["message"].lower()

    def test_empty_filename_rejected(self, client):
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(b"data"), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "file" in resp.get_json()["message"].lower()

    def test_insert_without_thumbnail(self, client, monkeypatch):
        # When the thumbnail generator yields nothing, the media is still
        # inserted -- just without thumbnail_bytes (the thumb-None branch).
        monkeypatch.setattr(media_list, "_make_pile_thumbnail", lambda *_a, **_k: None)
        wav = app_module.generate_wav(4324, 0.2)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(wav), "nothumb.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        new_id = resp.get_json()["media_id"]
        assert "thumbnail_bytes" not in medias[new_id]


# ---------------------------------------------------------------------------
# _make_pile_thumbnail dispatch (direct: video/image branches need real media
# bytes that the API path would only reach with heavyweight image/video
# embedders loaded).
# ---------------------------------------------------------------------------


class TestMakePileThumbnail:
    def test_image_branch(self):
        thumb = media_list._make_pile_thumbnail("image", _png_bytes(), "pic.png")
        # make_image_thumbnail re-encodes to JPEG; assert a decodable image.
        assert isinstance(thumb, bytes) and thumb
        with Image.open(io.BytesIO(thumb)) as img:
            assert img.size[0] > 0 and img.size[1] > 0

    def test_video_branch_returns_none_for_garbage(self):
        # No decodable frame -> generate_video_thumbnail yields None; the point
        # is that the video dispatch arm executes.
        assert media_list._make_pile_thumbnail("video", b"not-a-video" * 8) is None

    def test_unknown_type_returns_none(self):
        assert media_list._make_pile_thumbnail("mystery_type", b"data") is None
