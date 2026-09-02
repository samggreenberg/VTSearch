"""Video media type - MP4/AVI/MOV/WEBM/MKV files."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)
from vtscore.media.video import decode
from vtscore.media.video._demo_sources import build_demo_datasets, load_demo_source
from vtscore.media.video.crop import crop_frame

if TYPE_CHECKING:
    import numpy as np

_THUMB_SIZE = 128


def _seek_time(info: decode.VideoInfo, time_seconds: float | None) -> float:
    """Clamp *time_seconds* to a readable position; ``None`` means the middle."""
    if time_seconds is None:
        return info.duration / 2 if info.duration > 0 else 0.0
    if info.duration <= 0:
        return max(0.0, time_seconds)
    return max(0.0, min(time_seconds, max(0.0, info.duration - info.frame_seconds)))


def _encode_thumbnail(frame_rgb: "np.ndarray", size: int) -> bytes | None:
    """Shrink an RGB frame to fit *size* and encode it as PNG bytes."""
    try:
        from PIL import Image  # noqa: PLC0415

        img = Image.fromarray(frame_rgb)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return None


def _thumbnail_from_path(
    file_path: Path,
    time_seconds: float | None,
    size: int,
    info: decode.VideoInfo | None = None,
    crop_box: Any = None,
) -> bytes | None:
    """Grab one frame from *file_path* and return it as a PNG thumbnail.

    *info* lets a caller that already probed the container hand the result in
    rather than paying for a second probe.  *crop_box* is a unit's ``clip_box``
    (see :mod:`vtscore.media.video.crop`); passing it keeps a cropped unit's
    preview framed like the frames its embedder sees.
    """
    if info is None:
        info = decode.probe(file_path)
    if info is None:
        return None
    frame = decode.frame_at(file_path, _seek_time(info, time_seconds))
    if frame is None:
        return None
    return _encode_thumbnail(crop_frame(frame, crop_box), size)


def _thumbnail_from_bytes(
    video_bytes: bytes, time_seconds: float | None, size: int, crop_box: Any = None
) -> bytes | None:
    """Spill *video_bytes* to a temp file (decoders need a seekable path) and thumbnail it."""
    if not video_bytes:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()
        return _thumbnail_from_path(Path(tmp.name), time_seconds, size, crop_box=crop_box)
    finally:
        import os  # noqa: PLC0415

        os.unlink(tmp.name)


def generate_video_thumbnail(video_bytes: bytes, *, size: int = _THUMB_SIZE, crop_box: Any = None) -> bytes | None:
    """Extract the middle frame of a video and return it as a PNG thumbnail.

    Decoding runs out-of-process through ffmpeg (see
    :mod:`vtscore.media.video.decode`) and PIL produces the PNG.  Returns
    ``None`` if the video cannot be decoded or has no frames.
    """
    return _thumbnail_from_bytes(video_bytes, None, size, crop_box)


def generate_video_thumbnail_from_file(
    file_path: Path, *, size: int = _THUMB_SIZE, crop_box: Any = None
) -> bytes | None:
    """Extract the middle frame of a video file and return it as a PNG thumbnail."""
    return _thumbnail_from_path(file_path, None, size, crop_box=crop_box)


def generate_video_thumbnail_at(
    video_bytes: bytes, time_seconds: float, *, size: int = _THUMB_SIZE, crop_box: Any = None
) -> bytes | None:
    """Extract a frame at *time_seconds* from *video_bytes* and return it as a PNG thumbnail."""
    return _thumbnail_from_bytes(video_bytes, time_seconds, size, crop_box)


def generate_video_thumbnail_from_file_at(
    file_path: Path, time_seconds: float, *, size: int = _THUMB_SIZE, crop_box: Any = None
) -> bytes | None:
    """Extract a frame at *time_seconds* from a video file and return it as a PNG thumbnail."""
    return _thumbnail_from_path(file_path, time_seconds, size, crop_box=crop_box)


_VIDEO_MIME_TYPES: dict[str, str] = {
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


class VideoMediaType(MediaType):
    """Handles video medias - file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtscore.media.video.embedder_xclip.VideoXClipEmbedder`.
    """

    #: Video renders a poster-frame thumbnail: a browsable-thumbnail type.
    has_thumbnail = True

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "video"

    @property
    def name(self) -> str:
        return "Video"

    @property
    def icon(self) -> str:
        return "video"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]

    @property
    def folder_import_name(self) -> str:
        return "video"

    @property
    def dir_key(self) -> str:
        return "video_dir"

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        dur = media.get("duration")
        if dur and dur > 0:
            result["Duration"] = dur
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        return build_demo_datasets()

    def load_demo_source(
        self,
        source,
        categories,
        slice_start,
        slice_end,
        clips,
        on_progress=None,
        embedder=None,
        slice_frac_start=None,
        slice_frac_end=None,
        skip_embedding=False,
        **kwargs,
    ):
        return load_demo_source(
            self,
            source,
            categories,
            slice_start,
            slice_end,
            clips,
            on_progress=on_progress,
            embedder=embedder,
            slice_frac_start=slice_frac_start,
            slice_frac_end=slice_frac_end,
            skip_embedding=skip_embedding,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["thumbnail_bytes"]

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        if media_bytes is None:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
        return {"media_bytes": media_bytes, **self.load_thin_media_data(file_path)}

    def load_thin_media_data(self, file_path: Path) -> dict:
        """Duration + mid-frame thumbnail, both read straight from the path.

        Overrides the base default (which routes through
        :meth:`load_media_data` and strips the payload) because neither
        artifact needs the file's bytes in memory: ffmpeg probes the container
        by path and the thumbnail grabs a single decoded frame.  A thin video
        load therefore never buffers a multi-GB file just to get a preview
        frame.
        """
        # One probe serves both the duration and the thumbnail's mid-frame
        # seek, so a load costs one container read rather than two.
        info = decode.probe(file_path)
        duration = info.duration if info is not None else 0.0
        thumbnail = _thumbnail_from_path(file_path, None, _THUMB_SIZE, info=info)
        return {"duration": duration, "thumbnail_bytes": thumbnail}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def ensure_thumbnail_bytes(self, media: dict) -> bytes | None:
        """Return the mid-frame PNG, extracting it from the media's bytes if absent.

        The extraction is the same one the ingest-time path performs, so a
        thumbnail warmed here (archive members, which had no file to read at
        import) is indistinguishable from one computed by the loader.  The
        resolved payload is dropped as soon as the frame is grabbed, keeping a
        warm-up over multi-GB shards to one member at a time.

        A unit carrying a ``clip_box`` (letterbox-cropped by a cleaner) is
        framed to it, so a preview warmed here matches one the load stage
        rendered.
        """
        thumb = media.get("thumbnail_bytes")
        if thumb:
            return thumb
        raw = self._resolve_media_bytes(media)
        if not raw:
            return None
        thumb = generate_video_thumbnail(raw, crop_box=media.get("clip_box"))
        if thumb:
            media["thumbnail_bytes"] = thumb
        return thumb

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the video thumbnail as a PNG image, or *None*."""
        thumb = self.ensure_thumbnail_bytes(media)
        if not thumb:
            return None
        return MediaResponse(
            data=thumb,
            mimetype="image/png",
            download_name=f"media_{media['id']}_thumb.png",
        )

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".mp4"
        mimetype = _VIDEO_MIME_TYPES.get(ext, "video/mp4")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
