"""Library-tier tests for :mod:`vtscore.media.video.media_type`.

Brings video's media-type coverage up to the parity the audio tests hold
(``tests_lib/core/test_audio.py``): thumbnailing (middle-frame and
seek-to-time, from bytes and from a file), the frame-seek math and its
byte/format edge cases, the ``VideoMediaType`` display/serving surface, and
the error paths on every branch.

Frame decoding goes through :mod:`vtscore.media.video.decode`, which shells
out to ffmpeg and so cannot decode the tiny synthetic clips used in the rest
of the suite.  The happy paths therefore install a **fake decoder** (a real
ndarray frame, then the real PIL/PNG pipeline runs); the error paths use
genuinely undecodable input and the real decoder so the ``return None``
guards are exercised for real.  The two together cover every branch without
depending on a particular codec being present in the container.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from PIL import Image

from vtscore.media.video import _demo_sources, decode
from vtscore.media.video.media_type import (
    _VIDEO_MIME_TYPES,
    VideoMediaType,
    _seek_time,
    generate_video_thumbnail,
    generate_video_thumbnail_at,
    generate_video_thumbnail_from_file,
    generate_video_thumbnail_from_file_at,
)

if TYPE_CHECKING:
    from vtscore.media.embedder import MediaEmbedder

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Fake decoder
# ---------------------------------------------------------------------------


class _FakeDecoder:
    """Stand-in for the ffmpeg decode layer with configurable metadata.

    ``frame_at`` returns a solid-colour frame whose red channel encodes the
    requested timestamp's frame index, so two different seeks yield two
    visibly different frames (and thus different thumbnail bytes).  Every
    requested timestamp is recorded in :attr:`requested_times`.
    """

    def __init__(self, *, probe_ok=True, duration=6.0, fps=30.0, read_ok=True, dim=256):
        self._probe_ok = probe_ok
        self._duration = duration
        self._fps = fps
        self._read_ok = read_ok
        self._dim = dim
        self.requested_times: list[float] = []
        self.probed_paths: list[object] = []

    def probe(self, path):
        self.probed_paths.append(path)
        if not self._probe_ok:
            return None
        return decode.VideoInfo(duration=self._duration, fps=self._fps, width=self._dim, height=self._dim)

    def frame_at(self, _path, time_seconds):
        self.requested_times.append(float(time_seconds))
        if not self._read_ok:
            return None
        fill = int(round(time_seconds * self._fps)) % 256
        frame = np.zeros((self._dim, self._dim, 3), dtype=np.uint8)
        frame[..., 0] = fill
        return frame


@pytest.fixture
def install_fake_decoder(monkeypatch):
    """Return an installer that swaps the decode layer for a configured fake."""

    def _install(**cfg) -> _FakeDecoder:
        fake = _FakeDecoder(**cfg)
        monkeypatch.setattr(decode, "probe", fake.probe)
        monkeypatch.setattr(decode, "frame_at", fake.frame_at)
        return fake

    return _install


def _thumb_size(png_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png_bytes)) as img:
        return img.size


# ---------------------------------------------------------------------------
# Middle-frame thumbnails (from bytes and from a file)
# ---------------------------------------------------------------------------


class TestMiddleFrameThumbnail:
    def test_from_bytes_returns_png(self, install_fake_decoder):
        install_fake_decoder()
        out = generate_video_thumbnail(b"ignored-by-fake")
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_default_thumb_size(self, install_fake_decoder):
        install_fake_decoder(dim=256)
        out = generate_video_thumbnail(b"x")
        assert out is not None
        assert _thumb_size(out) == (128, 128)

    def test_from_bytes_custom_size(self, install_fake_decoder):
        install_fake_decoder(dim=256)
        out = generate_video_thumbnail(b"x", size=64)
        assert out is not None
        assert _thumb_size(out) == (64, 64)

    def test_from_bytes_seeks_to_middle(self, install_fake_decoder):
        fake = install_fake_decoder(duration=10.0, fps=30.0, dim=8)
        out = generate_video_thumbnail(b"x")
        assert out is not None
        assert fake.requested_times == [5.0]
        with Image.open(io.BytesIO(out)) as img:
            # The fill encodes the seeked frame index (5.0s * 30fps = 150).
            pixel = img.convert("RGB").getpixel((0, 0))
            assert isinstance(pixel, tuple)
            assert pixel[0] == 150

    def test_from_file_returns_png(self, install_fake_decoder, tmp_path):
        install_fake_decoder()
        out = generate_video_thumbnail_from_file(tmp_path / "clip.mp4")
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_empty_input_returns_none(self, install_fake_decoder):
        install_fake_decoder()
        assert generate_video_thumbnail(b"") is None

    def test_from_bytes_probe_failure_returns_none(self, install_fake_decoder):
        install_fake_decoder(probe_ok=False)
        assert generate_video_thumbnail(b"x") is None

    def test_from_bytes_read_failure_returns_none(self, install_fake_decoder):
        install_fake_decoder(read_ok=False)
        assert generate_video_thumbnail(b"x") is None

    def test_from_file_probe_failure_returns_none(self, install_fake_decoder, tmp_path):
        install_fake_decoder(probe_ok=False)
        assert generate_video_thumbnail_from_file(tmp_path / "clip.mp4") is None

    def test_from_file_read_failure_returns_none(self, install_fake_decoder, tmp_path):
        install_fake_decoder(read_ok=False)
        assert generate_video_thumbnail_from_file(tmp_path / "clip.mp4") is None


# ---------------------------------------------------------------------------
# Error paths with the real decoder
# ---------------------------------------------------------------------------


class TestThumbnailErrorPaths:
    def test_from_bytes_undecodable_input_returns_none(self):
        # The real decoder cannot decode this; every path yields None.
        assert generate_video_thumbnail(b"not a video at all") is None
        assert generate_video_thumbnail_at(b"not a video at all", 1.0) is None

    def test_from_file_missing_path_returns_none(self, tmp_path):
        # Probing a nonexistent file finds no video stream -> None.
        assert generate_video_thumbnail_from_file(tmp_path / "nope.mp4") is None
        assert generate_video_thumbnail_from_file_at(tmp_path / "nope.mp4", 1.0) is None


# ---------------------------------------------------------------------------
# Seek-time clamping (_seek_time)
# ---------------------------------------------------------------------------


class TestSeekTime:
    def _info(self, duration=10.0, fps=30.0):
        return decode.VideoInfo(duration=duration, fps=fps, width=64, height=64)

    def test_none_seeks_to_middle(self):
        assert _seek_time(self._info(), None) == pytest.approx(5.0)

    def test_passes_through_in_range_time(self):
        assert _seek_time(self._info(), 2.0) == pytest.approx(2.0)

    def test_clamps_beyond_end_to_last_frame(self):
        # One frame of headroom so the seek still lands on decodable video.
        assert _seek_time(self._info(), 9999.0) == pytest.approx(10.0 - 1 / 30)

    def test_clamps_negative_time_to_zero(self):
        assert _seek_time(self._info(), -5.0) == 0.0

    def test_unknown_fps_uses_default_frame_headroom(self):
        assert _seek_time(self._info(fps=0.0), 9999.0) == pytest.approx(10.0 - 0.04)

    def test_unknown_duration_passes_time_through(self):
        assert _seek_time(self._info(duration=0.0), 3.0) == pytest.approx(3.0)

    def test_unknown_duration_seeks_to_start_for_middle(self):
        assert _seek_time(self._info(duration=0.0), None) == 0.0


# ---------------------------------------------------------------------------
# Seek-to-time thumbnails (from bytes and from a file)
# ---------------------------------------------------------------------------


class TestThumbnailAtTime:
    def test_from_bytes_returns_png(self, install_fake_decoder):
        fake = install_fake_decoder(duration=10.0, fps=30.0)
        out = generate_video_thumbnail_at(b"x", 2.0)
        assert out is not None
        assert out[:8] == _PNG_MAGIC
        assert fake.requested_times == [2.0]

    def test_from_bytes_custom_size(self, install_fake_decoder):
        install_fake_decoder(dim=256)
        out = generate_video_thumbnail_at(b"x", 1.0, size=32)
        assert out is not None
        assert _thumb_size(out) == (32, 32)

    def test_different_times_produce_different_thumbnails(self, install_fake_decoder):
        install_fake_decoder(duration=10.0, fps=30.0, dim=8)
        early = generate_video_thumbnail_at(b"x", 1.0)
        late = generate_video_thumbnail_at(b"x", 5.0)
        assert early is not None and late is not None
        assert early != late

    def test_from_file_returns_png(self, install_fake_decoder, tmp_path):
        install_fake_decoder(duration=5.0, fps=24.0)
        out = generate_video_thumbnail_from_file_at(tmp_path / "clip.mp4", 1.5)
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_probe_failure_returns_none(self, install_fake_decoder):
        install_fake_decoder(probe_ok=False)
        assert generate_video_thumbnail_at(b"x", 1.0) is None

    def test_from_bytes_read_failure_returns_none(self, install_fake_decoder):
        install_fake_decoder(read_ok=False)
        assert generate_video_thumbnail_at(b"x", 1.0) is None

    def test_from_file_read_failure_returns_none(self, install_fake_decoder, tmp_path):
        install_fake_decoder(read_ok=False)
        assert generate_video_thumbnail_from_file_at(tmp_path / "clip.mp4", 1.0) is None


# ---------------------------------------------------------------------------
# VideoMediaType: identity + static metadata
# ---------------------------------------------------------------------------


class TestVideoMediaTypeIdentity:
    def setup_method(self):
        self.mt = VideoMediaType()

    def test_identity(self):
        assert self.mt.type_id == "video"
        assert self.mt.name == "Video"
        assert self.mt.icon == "video"

    def test_thumbnail_and_loop_flags(self):
        assert self.mt.has_thumbnail is True
        assert self.mt.loops is True

    def test_file_extensions(self):
        assert self.mt.file_extensions == ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]

    def test_import_keys(self):
        assert self.mt.folder_import_name == "video"
        assert self.mt.dir_key == "video_dir"

    def test_pickle_extra_fields(self):
        assert self.mt.pickle_extra_fields == ["thumbnail_bytes"]


# ---------------------------------------------------------------------------
# VideoMediaType.display_metadata
# ---------------------------------------------------------------------------


class TestVideoDisplayMetadata:
    def setup_method(self):
        self.mt = VideoMediaType()

    def test_full_metadata(self):
        md = self.mt.display_metadata({"category": "Archery", "duration": 3.5, "file_size": 2048, "id": 1})
        assert md["Category"] == "Archery"
        assert md["Duration"] == 3.5
        assert md["File Size"] == 2048

    def test_placeholder_category_hidden(self):
        for cat in ("unknown", "custom"):
            md = self.mt.display_metadata({"category": cat})
            assert "Category" not in md

    def test_zero_and_missing_fields_hidden(self):
        md = self.mt.display_metadata({"duration": 0, "file_size": 0})
        assert "Duration" not in md
        assert "File Size" not in md
        assert self.mt.display_metadata({}) == {}

    def test_merges_base_clip_fields(self):
        md = self.mt.display_metadata({"category": "Archery", "clip_start": 1.0, "clip_end": 2.0})
        # Type-specific field is present, and base-class clip fields are merged in.
        assert md["Category"] == "Archery"
        assert md["Clip Start"] == 1.0
        assert md["Clip End"] == 2.0


# ---------------------------------------------------------------------------
# VideoMediaType.demo_datasets
# ---------------------------------------------------------------------------


class TestVideoDemoDatasets:
    def setup_method(self):
        self.mt = VideoMediaType()
        self.demos = self.mt.demo_datasets

    def test_ids_unique_and_expected(self):
        ids = [d.id for d in self.demos]
        assert len(ids) == len(set(ids))
        for family in ("ucf101", "hmdb51", "ucf101_full", "kth"):
            for suffix in ("s", "m", "l", "a"):
                assert f"{family}_{suffix}" in ids

    def test_sources_are_downloadable(self):
        supported = set(_demo_sources._VIDEO_SOURCE_DOWNLOADERS)
        assert {d.source for d in self.demos} <= supported

    def test_categories_match_family(self):
        by_id = {d.id: d for d in self.demos}
        assert by_id["ucf101_s"].categories == _demo_sources._DEMO_CATEGORIES
        assert by_id["hmdb51_s"].categories == _demo_sources._HMDB51_CATEGORIES
        assert by_id["kth_s"].categories == _demo_sources._KTH_CATEGORIES
        assert by_id["ucf101_full_s"].categories == _demo_sources._UCF101_FULL_CATEGORIES

    def test_all_slice_covers_full_range(self):
        by_id = {d.id: d for d in self.demos}
        whole = by_id["kth_a"]
        assert whole.slice_frac_start == 0.0
        assert whole.slice_frac_end is None


# ---------------------------------------------------------------------------
# VideoMediaType.load_media_data
# ---------------------------------------------------------------------------


class TestVideoLoadMediaData:
    def setup_method(self):
        self.mt = VideoMediaType()

    def test_computes_duration_and_thumbnail(self, install_fake_decoder, tmp_path):
        install_fake_decoder(duration=3.0, fps=30.0)
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"raw-video-bytes")
        data = self.mt.load_media_data(path)
        assert data["media_bytes"] == b"raw-video-bytes"  # read from disk
        assert data["duration"] == pytest.approx(3.0)
        assert data["thumbnail_bytes"][:8] == _PNG_MAGIC

    def test_uses_provided_bytes_without_reading_file(self, install_fake_decoder, tmp_path):
        install_fake_decoder(duration=2.0, fps=30.0)
        data = self.mt.load_media_data(tmp_path / "does-not-exist.mp4", media_bytes=b"inline")
        assert data["media_bytes"] == b"inline"
        assert data["duration"] == pytest.approx(2.0)

    def test_probes_the_path_once(self, install_fake_decoder, tmp_path):
        """The thumbnail reuses the duration's probe instead of re-probing."""
        fake = install_fake_decoder(duration=4.0, fps=25.0)
        data = self.mt.load_media_data(tmp_path / "x.mp4", media_bytes=b"inline")
        assert data["duration"] == pytest.approx(4.0)
        assert data["thumbnail_bytes"][:8] == _PNG_MAGIC
        assert len(fake.probed_paths) == 1

    def test_decode_failure_yields_zero_duration(self, install_fake_decoder, tmp_path):
        install_fake_decoder(probe_ok=False)
        data = self.mt.load_media_data(tmp_path / "x.mp4", media_bytes=b"b")
        assert data["duration"] == 0.0
        assert data["thumbnail_bytes"] is None


# ---------------------------------------------------------------------------
# VideoMediaType.image_response
# ---------------------------------------------------------------------------


class TestVideoImageResponse:
    def setup_method(self):
        self.mt = VideoMediaType()

    def test_returns_stored_thumbnail(self):
        resp = self.mt.image_response({"id": 7, "thumbnail_bytes": b"PNGDATA"})
        assert resp is not None
        assert resp.data == b"PNGDATA"
        assert resp.mimetype == "image/png"
        assert resp.download_name == "media_7_thumb.png"

    def test_generates_and_memoises_from_media_bytes(self, install_fake_decoder):
        install_fake_decoder()
        media = {"id": 3, "media_bytes": b"raw"}
        resp = self.mt.image_response(media)
        assert resp is not None
        assert resp.data[:8] == _PNG_MAGIC
        # Generated thumbnail is cached back onto the media dict.
        assert media["thumbnail_bytes"] == resp.data

    def test_returns_none_when_no_bytes_resolvable(self):
        assert self.mt.image_response({"id": 9}) is None

    def test_returns_none_when_generation_fails(self, install_fake_decoder):
        install_fake_decoder(read_ok=False)
        media = {"id": 4, "media_bytes": b"undecodable"}
        assert self.mt.image_response(media) is None
        assert "thumbnail_bytes" not in media


# ---------------------------------------------------------------------------
# VideoMediaType.media_response
# ---------------------------------------------------------------------------


class TestVideoMediaResponse:
    def setup_method(self):
        self.mt = VideoMediaType()

    @pytest.mark.parametrize(
        ("filename", "expected_mime", "expected_ext"),
        [
            ("clip.mp4", "video/mp4", ".mp4"),
            ("clip.webm", "video/webm", ".webm"),
            ("clip.mov", "video/quicktime", ".mov"),
            ("clip.avi", "video/x-msvideo", ".avi"),
            ("clip.mkv", "video/x-matroska", ".mkv"),
            ("clip.MP4", "video/mp4", ".mp4"),  # extension lowercased
            ("clip.xyz", "video/mp4", ".xyz"),  # unknown ext -> mp4 mimetype
        ],
    )
    def test_mimetype_by_extension(self, filename, expected_mime, expected_ext):
        resp = self.mt.media_response({"id": 5, "filename": filename, "media_bytes": b"data"})
        assert resp.mimetype == expected_mime
        assert resp.data == b"data"
        assert resp.download_name == f"media_5{expected_ext}"

    def test_missing_filename_defaults_to_mp4(self):
        resp = self.mt.media_response({"id": 6, "media_bytes": b"data"})
        assert resp.mimetype == "video/mp4"
        assert resp.download_name == "media_6.mp4"

    def test_unresolvable_bytes_yield_empty_payload(self):
        resp = self.mt.media_response({"id": 8, "filename": "clip.webm"})
        assert resp.data == b""
        assert resp.mimetype == "video/webm"
        assert resp.download_name == "media_8.webm"

    def test_mime_table_matches_constant(self):
        # Guards against drift between the served mimetypes and the table.
        assert _VIDEO_MIME_TYPES[".webm"] == "video/webm"
        assert _VIDEO_MIME_TYPES[".mkv"] == "video/x-matroska"


# ---------------------------------------------------------------------------
# VideoMediaType.load_demo_source error handling
# ---------------------------------------------------------------------------


class TestVideoLoadDemoSourceErrors:
    def setup_method(self):
        self.mt = VideoMediaType()

    def test_unsupported_source_raises(self):
        with pytest.raises(ValueError, match="Unsupported video source"):
            self.mt.load_demo_source(
                "bogus-source",
                categories=[],
                slice_start=0,
                slice_end=None,
                clips={},
                on_progress=lambda *a, **k: None,
                # Non-None so no embedder registry lookup happens; the source
                # check rejects it before anything touches the embedder.
                embedder=cast("MediaEmbedder", object()),
            )
