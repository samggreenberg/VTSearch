"""Library-tier tests for :mod:`vtscore.media.video.media_type`.

Brings video's media-type coverage up to the parity the audio tests hold
(``tests_lib/core/test_audio.py``): thumbnailing (middle-frame and
seek-to-time, from bytes and from a file), the frame-seek math and its
byte/format edge cases, the ``VideoMediaType`` display/serving surface, and
the error paths on every branch.

Frame decoding goes through OpenCV (``cv2``), which cannot decode the tiny
synthetic clips used in the rest of the suite.  So the happy paths inject a
**fake ``cv2``** (a real ndarray frame, then the real PIL/PNG pipeline runs)
via ``sys.modules``; the error paths use genuinely undecodable input and the
real (or absent) ``cv2`` so the ``return None`` guards are exercised for
real.  The two together cover every branch without depending on a working
video codec being present in the container.
"""

from __future__ import annotations

import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from vtscore.media.video.media_type import (
    _VIDEO_MIME_TYPES,
    VideoMediaType,
    _frame_at_time,
    generate_video_thumbnail,
    generate_video_thumbnail_at,
    generate_video_thumbnail_from_file,
    generate_video_thumbnail_from_file_at,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Fake cv2 decoder
# ---------------------------------------------------------------------------

# Sentinel values for the CAP_PROP_* / COLOR_* constants.  The fake reads them
# by identity, so their concrete values are irrelevant.
_PROP_FRAME_COUNT = "cap_prop_frame_count"
_PROP_FPS = "cap_prop_fps"
_PROP_POS_FRAMES = "cap_prop_pos_frames"
_COLOR_BGR2RGB = "color_bgr2rgb"


class _FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` with configurable metadata.

    ``read`` returns a solid-colour frame whose fill byte encodes the last
    frame index requested via ``set(POS_FRAMES, ...)``, so seeking to two
    different frames yields two visibly different frames (and thus different
    thumbnail bytes).
    """

    def __init__(self, *, opened=True, frame_count=200, fps=30.0, read_ok=True, dim=256):
        self._opened = opened
        self._frame_count = frame_count
        self._fps = fps
        self._read_ok = read_ok
        self._dim = dim
        self.requested_frame: int | None = None
        self.released = False

    def isOpened(self):  # noqa: N802 - cv2 API name
        return self._opened

    def get(self, prop):
        if prop == _PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == _PROP_FPS:
            return float(self._fps)
        return 0.0

    def set(self, prop, value):
        if prop == _PROP_POS_FRAMES:
            self.requested_frame = int(value)

    def read(self):
        if not self._read_ok:
            return False, None
        fill = (self.requested_frame or 0) % 256
        # A BGR frame (channel 0 carries the fill so BGR->RGB is observable).
        frame = np.zeros((self._dim, self._dim, 3), dtype=np.uint8)
        frame[..., 0] = fill
        return True, frame

    def release(self):
        self.released = True


def _fake_cv2(**capture_cfg) -> types.ModuleType:
    """Build a fake ``cv2`` module whose ``VideoCapture`` yields ``_FakeCapture``."""
    mod = types.ModuleType("cv2")
    mod.CAP_PROP_FRAME_COUNT = _PROP_FRAME_COUNT
    mod.CAP_PROP_FPS = _PROP_FPS
    mod.CAP_PROP_POS_FRAMES = _PROP_POS_FRAMES
    mod.COLOR_BGR2RGB = _COLOR_BGR2RGB
    mod.VideoCapture = lambda _path: _FakeCapture(**capture_cfg)

    def cvt_color(frame, code):
        assert code == _COLOR_BGR2RGB
        return frame[..., ::-1].copy()

    mod.cvtColor = cvt_color
    return mod


@pytest.fixture
def install_fake_cv2(monkeypatch):
    """Return an installer that swaps ``cv2`` for a configured fake in ``sys.modules``."""

    def _install(**capture_cfg):
        monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(**capture_cfg))

    return _install


def _thumb_size(png_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png_bytes)) as img:
        return img.size


# ---------------------------------------------------------------------------
# Middle-frame thumbnails (from bytes and from a file)
# ---------------------------------------------------------------------------


class TestMiddleFrameThumbnail:
    def test_from_bytes_returns_png(self, install_fake_cv2):
        install_fake_cv2(frame_count=100, fps=30.0)
        out = generate_video_thumbnail(b"ignored-by-fake")
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_default_thumb_size(self, install_fake_cv2):
        install_fake_cv2(dim=256)
        out = generate_video_thumbnail(b"x")
        assert out is not None
        assert _thumb_size(out) == (128, 128)

    def test_from_bytes_custom_size(self, install_fake_cv2):
        install_fake_cv2(dim=256)
        out = generate_video_thumbnail(b"x", size=64)
        assert out is not None
        assert _thumb_size(out) == (64, 64)

    def test_from_bytes_seeks_to_middle_frame(self, install_fake_cv2):
        # frame_count=100 -> mid index 50; the fill byte encodes the frame.
        install_fake_cv2(frame_count=100, fps=30.0, dim=8)
        out = generate_video_thumbnail(b"x")
        assert out is not None
        with Image.open(io.BytesIO(out)) as img:
            # The fill byte encodes the seeked frame index (50); BGR->RGB moves
            # the fake's channel-0 fill into the blue channel.
            assert img.convert("RGB").getpixel((0, 0))[2] == 50

    def test_from_file_returns_png(self, install_fake_cv2, tmp_path):
        install_fake_cv2(frame_count=40)
        out = generate_video_thumbnail_from_file(tmp_path / "clip.mp4")
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_not_opened_returns_none(self, install_fake_cv2):
        install_fake_cv2(opened=False)
        assert generate_video_thumbnail(b"x") is None

    def test_from_bytes_zero_frames_returns_none(self, install_fake_cv2):
        install_fake_cv2(frame_count=0)
        assert generate_video_thumbnail(b"x") is None

    def test_from_bytes_read_failure_returns_none(self, install_fake_cv2):
        install_fake_cv2(read_ok=False)
        assert generate_video_thumbnail(b"x") is None

    def test_from_file_not_opened_returns_none(self, install_fake_cv2, tmp_path):
        install_fake_cv2(opened=False)
        assert generate_video_thumbnail_from_file(tmp_path / "clip.mp4") is None

    def test_from_file_zero_frames_returns_none(self, install_fake_cv2, tmp_path):
        install_fake_cv2(frame_count=0)
        assert generate_video_thumbnail_from_file(tmp_path / "clip.mp4") is None

    def test_from_file_read_failure_returns_none(self, install_fake_cv2, tmp_path):
        install_fake_cv2(read_ok=False)
        assert generate_video_thumbnail_from_file(tmp_path / "clip.mp4") is None


# ---------------------------------------------------------------------------
# Error paths without a working decoder
# ---------------------------------------------------------------------------


class TestThumbnailErrorPaths:
    def test_from_bytes_cv2_unavailable_returns_none(self, monkeypatch):
        # `import cv2` raises when the module object is None -> caught -> None.
        monkeypatch.setitem(sys.modules, "cv2", None)
        assert generate_video_thumbnail(b"anything") is None
        assert generate_video_thumbnail_at(b"anything", 1.0) is None

    def test_from_file_cv2_unavailable_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "cv2", None)
        assert generate_video_thumbnail_from_file(tmp_path / "x.mp4") is None
        assert generate_video_thumbnail_from_file_at(tmp_path / "x.mp4", 1.0) is None

    def test_from_bytes_undecodable_input_returns_none(self):
        # Real (or absent) cv2 cannot decode this; every path yields None.
        assert generate_video_thumbnail(b"not a video at all") is None

    def test_from_file_missing_path_returns_none(self, tmp_path):
        # VideoCapture on a nonexistent file never opens -> None (and the
        # from-file guard also swallows any decode exception).
        assert generate_video_thumbnail_from_file(tmp_path / "nope.mp4") is None


# ---------------------------------------------------------------------------
# Seek-to-time frame math (_frame_at_time)
# ---------------------------------------------------------------------------


class TestFrameAtTime:
    def _cap_with_fake_cv2(self, install_fake_cv2, **cfg):
        install_fake_cv2()  # only the constants matter here
        return _FakeCapture(**cfg)

    def test_seeks_by_fps(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=300, fps=30.0)
        frame = _frame_at_time(cap, 2.0)
        assert frame is not None
        assert cap.requested_frame == 60  # round(2.0 * 30)

    def test_rounds_to_nearest_frame(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=300, fps=30.0)
        _frame_at_time(cap, 1.017)  # 30.51 -> 31
        assert cap.requested_frame == 31

    def test_clamps_beyond_end_to_last_frame(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=100, fps=30.0)
        _frame_at_time(cap, 9999.0)
        assert cap.requested_frame == 99  # frame_count - 1

    def test_clamps_negative_time_to_zero(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=100, fps=30.0)
        _frame_at_time(cap, -5.0)
        assert cap.requested_frame == 0

    def test_falls_back_to_middle_when_fps_zero(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=80, fps=0.0)
        _frame_at_time(cap, 3.0)
        assert cap.requested_frame == 40  # frame_count // 2

    def test_zero_frame_count_returns_none(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=0, fps=30.0)
        assert _frame_at_time(cap, 1.0) is None

    def test_read_failure_returns_none(self, install_fake_cv2):
        cap = self._cap_with_fake_cv2(install_fake_cv2, frame_count=100, fps=30.0, read_ok=False)
        assert _frame_at_time(cap, 1.0) is None


# ---------------------------------------------------------------------------
# Seek-to-time thumbnails (from bytes and from a file)
# ---------------------------------------------------------------------------


class TestThumbnailAtTime:
    def test_from_bytes_returns_png(self, install_fake_cv2):
        install_fake_cv2(frame_count=300, fps=30.0)
        out = generate_video_thumbnail_at(b"x", 2.0)
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_custom_size(self, install_fake_cv2):
        install_fake_cv2(dim=256)
        out = generate_video_thumbnail_at(b"x", 1.0, size=32)
        assert out is not None
        assert _thumb_size(out) == (32, 32)

    def test_different_times_produce_different_thumbnails(self, install_fake_cv2):
        install_fake_cv2(frame_count=300, fps=30.0, dim=8)
        early = generate_video_thumbnail_at(b"x", 1.0)  # frame 30
        late = generate_video_thumbnail_at(b"x", 5.0)  # frame 150
        assert early is not None and late is not None
        assert early != late

    def test_from_file_returns_png(self, install_fake_cv2, tmp_path):
        install_fake_cv2(frame_count=120, fps=24.0)
        out = generate_video_thumbnail_from_file_at(tmp_path / "clip.mp4", 1.5)
        assert out is not None
        assert out[:8] == _PNG_MAGIC

    def test_from_bytes_not_opened_returns_none(self, install_fake_cv2):
        install_fake_cv2(opened=False)
        assert generate_video_thumbnail_at(b"x", 1.0) is None

    def test_from_bytes_zero_frames_returns_none(self, install_fake_cv2):
        install_fake_cv2(frame_count=0)
        assert generate_video_thumbnail_at(b"x", 1.0) is None

    def test_from_file_read_failure_returns_none(self, install_fake_cv2, tmp_path):
        install_fake_cv2(read_ok=False)
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
        md = self.mt.display_metadata(
            {"category": "Archery", "duration": 3.5, "file_size": 2048, "id": 1}
        )
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
        md = self.mt.display_metadata(
            {"category": "Archery", "clip_start": 1.0, "clip_end": 2.0}
        )
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
        supported = set(self.mt._VIDEO_SOURCE_DOWNLOADERS)
        assert {d.source for d in self.demos} <= supported

    def test_categories_match_family(self):
        by_id = {d.id: d for d in self.demos}
        assert by_id["ucf101_s"].categories == self.mt._DEMO_CATEGORIES
        assert by_id["hmdb51_s"].categories == self.mt._HMDB51_CATEGORIES
        assert by_id["kth_s"].categories == self.mt._KTH_CATEGORIES
        assert by_id["ucf101_full_s"].categories == self.mt._UCF101_FULL_CATEGORIES

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

    def test_computes_duration_and_thumbnail(self, install_fake_cv2, tmp_path):
        install_fake_cv2(frame_count=90, fps=30.0)
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"raw-video-bytes")
        data = self.mt.load_media_data(path)
        assert data["media_bytes"] == b"raw-video-bytes"  # read from disk
        assert data["duration"] == pytest.approx(3.0)  # 90 / 30
        assert data["thumbnail_bytes"][:8] == _PNG_MAGIC

    def test_uses_provided_bytes_without_reading_file(self, install_fake_cv2, tmp_path):
        install_fake_cv2(frame_count=60, fps=30.0)
        data = self.mt.load_media_data(tmp_path / "does-not-exist.mp4", media_bytes=b"inline")
        assert data["media_bytes"] == b"inline"
        assert data["duration"] == pytest.approx(2.0)

    def test_zero_fps_yields_zero_duration(self, install_fake_cv2, tmp_path):
        install_fake_cv2(frame_count=90, fps=0.0)
        data = self.mt.load_media_data(tmp_path / "x.mp4", media_bytes=b"b")
        assert data["duration"] == 0.0

    def test_decode_failure_yields_zero_duration(self, monkeypatch, tmp_path):
        # cv2 unavailable -> the try/except sets duration 0.0 and thumbnail None.
        monkeypatch.setitem(sys.modules, "cv2", None)
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

    def test_generates_and_memoises_from_media_bytes(self, install_fake_cv2):
        install_fake_cv2(frame_count=50, fps=30.0)
        media = {"id": 3, "media_bytes": b"raw"}
        resp = self.mt.image_response(media)
        assert resp is not None
        assert resp.data[:8] == _PNG_MAGIC
        # Generated thumbnail is cached back onto the media dict.
        assert media["thumbnail_bytes"] == resp.data

    def test_returns_none_when_no_bytes_resolvable(self):
        assert self.mt.image_response({"id": 9}) is None

    def test_returns_none_when_generation_fails(self, install_fake_cv2):
        install_fake_cv2(read_ok=False)
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
                embedder=object(),  # non-None so no embedder registry lookup happens
            )
