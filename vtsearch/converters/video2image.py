"""Extract frames from a video as images."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from vtsearch.converters.base import MediaConverter


class Video2ImageMediaConverter(MediaConverter):
    """Cut a video into *n_clips* segments and extract the middle frame
    from each segment as a PNG image.

    Uses OpenCV (``cv2``) to read the video and sample frames.

    Parameters
    ----------
    n_clips : int
        Number of clips to divide the video into.  One frame is
        extracted from the temporal centre of each clip.  Defaults to 10.
    """

    def __init__(self, n_clips: int = 10) -> None:
        self.n_clips = n_clips

    @property
    def source_type(self) -> str:
        return "video"

    @property
    def target_type(self) -> str:
        return "image"

    def convert(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        import tempfile  # noqa: PLC0415

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
            if not cap.isOpened():
                return []

            try:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    return []

                n = min(self.n_clips, frame_count)
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

                    results.append({
                        "filename": f"{stem}_clip_{clip_num + 1}.png",
                        "media_bytes": png_bytes,
                        "duration": 0,
                        "width": img.width,
                        "height": img.height,
                    })
            finally:
                cap.release()
        finally:
            if tmp_file is not None:
                import os  # noqa: PLC0415

                os.unlink(tmp_file.name)

        return results
