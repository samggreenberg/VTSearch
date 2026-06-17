"""Extract frames from a video as images."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from vtscore.converters.base import MediaConverter
from vtscore.plugins import PluginField


class Video2ImageMediaConverter(MediaConverter):
    """Cut a video into evenly-spaced segments and extract the middle frame
    from each segment as a PNG image.

    Uses OpenCV (``cv2``) to read the video and sample frames.

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

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:  # noqa: C901
        import tempfile  # noqa: PLC0415

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

        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "video.mp4")
        stem = Path(filename).stem

        try:
            import cv2  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("Video2ImageMediaConverter requires opencv-python and Pillow")
            return []

        # We need a file path for cv2.VideoCapture
        tmp_file = None
        if media_path and Path(media_path).exists():
            video_path = str(media_path)
        elif media_bytes:
            ext = Path(filename).suffix or ".mp4"
            tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp_file.write(media_bytes)
            tmp_file.close()
            video_path = tmp_file.name
        else:
            return []

        results: list[dict[str, Any]] = []
        try:
            cap = cv2.VideoCapture(video_path)
            try:
                if not cap.isOpened():
                    return []
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    return []

                # "Seconds per frame" wins when supplied: derive the frame
                # count from the video's duration so longer videos yield more
                # frames.  Otherwise fall back to the fixed "frames per video".
                if seconds_per_frame > 0:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if fps > 0:
                        duration = frame_count / fps
                        n = max(1, round(duration / seconds_per_frame))
                    else:
                        n = n_clips
                else:
                    n = n_clips
                n = min(n, frame_count)
                # Divide video into n equal segments, pick middle frame of each
                segment_size = frame_count / n
                mid_indices = [int((i + 0.5) * segment_size) for i in range(n)]
                # Clamp to valid range
                mid_indices = [min(idx, frame_count - 1) for idx in mid_indices]

                for clip_num, frame_idx in enumerate(mid_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                cap.release()
        finally:
            if tmp_file is not None:
                import os  # noqa: PLC0415

                os.unlink(tmp_file.name)

        return results


CONVERTER = Video2ImageMediaConverter()
