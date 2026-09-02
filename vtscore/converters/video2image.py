"""Extract frames from a video as images."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from vtscore.converters.base import MediaConverter
from vtscore.media.video import decode
from vtscore.plugins import PluginField

logger = logging.getLogger(__name__)


def _compute_n_frames(frame_count: int, fps: float, n_clips: int, seconds_per_frame: float) -> int:
    """Choose how many frames to extract from a video.

    When ``seconds_per_frame`` is positive and a real ``fps`` is known, the
    count is derived from the video's duration so longer videos yield more
    frames; otherwise it falls back to the fixed ``n_clips``. The result is
    capped at ``frame_count`` so we never ask for more frames than exist.
    """
    if seconds_per_frame > 0 and fps > 0:
        duration = frame_count / fps
        n = max(1, round(duration / seconds_per_frame))
    else:
        n = n_clips
    return min(n, frame_count)


class Video2ImageMediaConverter(MediaConverter):
    """Cut a video into evenly-spaced segments and extract the middle frame
    from each segment as a PNG image.

    Frames are decoded out-of-process by ffmpeg (see
    :mod:`vtscore.media.video.decode`).

    The number of frames is driven by one of two mutually-exclusive,
    user-configurable parameters — supply one *or* the other, never both:

    ``n_clips`` (Frames per video)
        Fixed number of evenly-spaced frames to extract from each video,
        regardless of its length.  Defaults to 10.

    ``seconds_per_frame`` (Seconds per frame)
        Sampling cadence: extract one frame per this many seconds of video,
        so longer videos yield more frames.  When set (non-empty and
        positive) it takes precedence over ``n_clips``.  Left blank by
        default, meaning ``n_clips`` is used.

    The two fields :attr:`clear <vtscore.plugins.PluginField.clears>` each
    other in the UI, so only one is ever active at a time; the backend
    likewise honours ``seconds_per_frame`` when present and falls back to
    ``n_clips`` otherwise.
    """

    display_name = "Video → Images"
    description = "Extract frames from video files"
    summary_template = "Sample {n_clips} frames per video, or one frame every {seconds_per_frame}s."
    fields = [
        PluginField(
            key="n_clips",
            label="Frames per video",
            field_type="number",
            description="Fixed number of evenly-spaced frames to extract from each video.",
            default="10",
            required=False,
            min="1",
            max="1000",
            step="1",
            clears=["seconds_per_frame"],
        ),
        PluginField(
            key="seconds_per_frame",
            label="Seconds per frame",
            field_type="number",
            description="Extract one frame per this many seconds of video (longer videos yield more frames). "
            "Leave blank to use 'Frames per video' instead.",
            default="",
            required=False,
            min="0.1",
            max="3600",
            step="0.5",
            clears=["n_clips"],
        ),
    ]

    @property
    def source_type(self) -> str:
        return "video"

    @property
    def target_type(self) -> str:
        return "image"

    def _resolve_frame_params(self, params: dict[str, Any] | None) -> tuple[int, float]:
        """Parse the two frame-count knobs into ``(n_clips, seconds_per_frame)``.

        ``n_clips`` is coerced to an integer of at least 1 (defaulting to 10 on
        a missing/invalid value); ``seconds_per_frame`` is coerced to a float,
        defaulting to ``0.0`` (meaning "use ``n_clips``") when blank or invalid.
        """
        try:
            n_clips = int(self.get_param(params, "n_clips") or 10)
        except (TypeError, ValueError):
            n_clips = 10
        if n_clips < 1:
            n_clips = 1

        seconds_raw = self.get_param(params, "seconds_per_frame")
        try:
            seconds_per_frame = float(seconds_raw) if seconds_raw not in (None, "") else 0.0
        except (TypeError, ValueError):
            seconds_per_frame = 0.0

        return n_clips, seconds_per_frame

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        import tempfile  # noqa: PLC0415

        n_clips, seconds_per_frame = self._resolve_frame_params(params)

        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "video.mp4")
        stem = Path(filename).stem

        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            logger.warning("video2image requires Pillow; producing no images")
            return []

        # The decoder needs a seekable file path, not a buffer.
        tmp_file = None
        if media_path and Path(media_path).exists():
            video_path = str(media_path)
        elif media_bytes:
            ext = Path(filename).suffix or ".mp4"
            tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            try:
                tmp_file.write(media_bytes)
                tmp_file.close()
            except BaseException:
                # A failed write (e.g. ENOSPC) happens before the try/finally
                # below owns the temp file; clean it up here or it leaks.
                tmp_file.close()
                Path(tmp_file.name).unlink(missing_ok=True)
                raise
            video_path = tmp_file.name
        else:
            return []

        results: list[dict[str, Any]] = []
        try:
            info = decode.probe(video_path)
            if info is None or info.duration <= 0:
                return []

            # "Seconds per frame" wins when supplied: derive the frame count
            # from the video's duration so longer videos yield more frames.
            # Otherwise fall back to the fixed "frames per video".
            frame_count = max(1, info.frame_count or 1)
            fps = info.fps if seconds_per_frame > 0 else 0.0
            n = _compute_n_frames(frame_count, fps, n_clips, seconds_per_frame)
            # Divide the video into n equal segments and take the middle of each,
            # stopping a frame short of the end where nothing decodes.
            last = max(0.0, info.duration - info.frame_seconds)
            segment = info.duration / n
            mid_times = [min((i + 0.5) * segment, last) for i in range(n)]

            for clip_num, frame_rgb in enumerate(decode.frames_at(video_path, mid_times)):
                img = Image.fromarray(frame_rgb)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()

                results.append(
                    {
                        "filename": f"{stem}_clip_{clip_num + 1}.png",
                        "media_bytes": png_bytes,
                        "duration": 0,
                        "width": img.width,
                        "height": img.height,
                    }
                )
        finally:
            if tmp_file is not None:
                import os  # noqa: PLC0415

                os.unlink(tmp_file.name)

        return results


CONVERTER = Video2ImageMediaConverter()
