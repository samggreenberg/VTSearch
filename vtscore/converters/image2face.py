"""Localise faces in an image and emit one face crop per detected face.

This is the *convert-in* path for the :class:`~vtscore.media.face.media_type.FaceMediaType`
half type: images are the only source of faces (faces have no native import),
so every face in the app is produced here. MediaPipe Face Detection finds each
face above the configured confidence, and each is cropped (with optional
padding) into its own ``face``-type media, ready to be embedded in FaceNet
identity space.

``mediapipe`` is an opt-in dependency (declared in pyproject's ``DEP001``
ignore-list, same pattern as ``facenet-pytorch``); when it is missing the
converter degrades gracefully to producing no faces.
"""

from __future__ import annotations

import io
from typing import Any, Optional

from vtscore.converters.base import MediaConverter
from vtscore.plugins import PluginField


class Image2FaceMediaConverter(MediaConverter):
    """Detect faces with MediaPipe and emit one crop per face as ``face`` media.

    Images with **no** detected faces yield zero outputs and drop out of the
    dataset - the intended semantic for a face-only collection. The four
    tunables mirror the (now-removed) image face clipper: confidence
    ``threshold``, crop ``padding``, ``min_size``, and detector ``model_selection``.
    """

    display_name = "Images → Faces"
    description = "Detect faces in images and crop one face per detection"
    summary_template = "Detect faces (confidence ≥ {threshold}, min size {min_size}px) and crop one face per detection."
    fields = [
        PluginField(
            key="threshold",
            label="Min detection confidence",
            field_type="number",
            description="Skip detections below this confidence (0–1).",
            default="0.5",
            required=False,
            min="0",
            max="1",
            step="0.05",
        ),
        PluginField(
            key="padding",
            label="Crop padding (fraction of face size)",
            field_type="number",
            description="Expand each crop by this fraction on every side so the embedder gets some context.",
            default="0.25",
            required=False,
            min="0",
            max="2",
            step="0.05",
        ),
        PluginField(
            key="min_size",
            label="Minimum face size (pixels)",
            field_type="number",
            description="Drop detections whose padded crop is smaller than this on either axis.",
            default="32",
            required=False,
            min="1",
            step="1",
        ),
        PluginField(
            key="model_selection",
            label="Detector range",
            field_type="number",
            description="0 = short-range (within ~2m), 1 = full-range (within ~5m).",
            default="1",
            required=False,
            min="0",
            max="1",
            step="1",
        ),
    ]

    @property
    def source_type(self) -> str:
        return "image"

    @property
    def target_type(self) -> str:
        return "face"

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _make_detector(self, threshold: float, model_selection: int) -> Optional[Any]:
        try:
            import mediapipe as mp  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        except ImportError:
            print("Image2FaceMediaConverter requires mediapipe; producing no faces")
            return None
        return mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=threshold,
            model_selection=model_selection,
        )

    def _box_from_detection(
        self, det: Any, w: int, h: int, threshold: float, padding: float, min_size: int
    ) -> tuple[int, int, int, int, float] | None:
        """Turn one MediaPipe detection into a padded, clamped pixel box.

        Returns ``(x1, y1, x2, y2, conf)`` in pixel coordinates, or ``None``
        when the detection is unusable, below *threshold*, or smaller than
        *min_size* on either axis after padding.
        """
        try:
            conf = float(det.score[0])
        except (AttributeError, IndexError, TypeError):
            return None
        if conf < threshold:
            return None
        rel = det.location_data.relative_bounding_box
        fx1 = rel.xmin * w
        fy1 = rel.ymin * h
        fx2 = (rel.xmin + rel.width) * w
        fy2 = (rel.ymin + rel.height) * h
        pad_x = (fx2 - fx1) * padding
        pad_y = (fy2 - fy1) * padding
        x1 = max(0, int(round(fx1 - pad_x)))
        y1 = max(0, int(round(fy1 - pad_y)))
        x2 = min(w, int(round(fx2 + pad_x)))
        y2 = min(h, int(round(fy2 + pad_y)))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            return None
        return (x1, y1, x2, y2, conf)

    def _resolve_params(self, params: dict[str, Any] | None) -> tuple[float, float, int, int]:
        threshold = float(self.get_param(params, "threshold") or 0.5)
        padding = float(self.get_param(params, "padding") or 0.0)
        min_size = int(self.get_param(params, "min_size") or 32)
        model_selection = int(self.get_param(params, "model_selection") or 1)
        if model_selection not in (0, 1):
            model_selection = 1
        return threshold, max(0.0, padding), max(1, min_size), model_selection

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from vtscore.media.embedder import media_from_path  # noqa: PLC0415

        threshold, padding, min_size, model_selection = self._resolve_params(params)

        media_bytes = media.get("media_bytes")
        if media_bytes is None:
            media_path = media.get("media_path")
            if media_path:
                media_bytes = media_from_path(media_path).get("media_bytes")
        if not media_bytes:
            return []
        try:
            img = Image.open(io.BytesIO(media_bytes)).convert("RGB")
        except Exception:
            return []
        img_w, img_h = img.size
        fmt = img.format or "PNG"

        detector = self._make_detector(threshold, model_selection)
        if detector is None:
            return []
        try:
            results = detector.process(np.array(img))
        except Exception:
            return []
        detections = getattr(results, "detections", None) or []

        boxes: list[tuple[int, int, int, int, float]] = []
        for det in detections:
            box = self._box_from_detection(det, img_w, img_h, threshold, padding, min_size)
            if box is not None:
                boxes.append(box)
        # Highest-confidence face first so face index 0 is the "primary" face.
        boxes.sort(key=lambda b: b[4], reverse=True)

        stem = (media.get("filename") or "image").rsplit(".", 1)[0]
        out_ext = "png" if fmt.upper() == "PNG" else fmt.lower()
        outputs: list[dict[str, Any]] = []
        for idx, (x1, y1, x2, y2, conf) in enumerate(boxes):
            cropped = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
            crop_bytes = buf.getvalue()
            outputs.append(
                {
                    "filename": f"{stem}_face_{idx + 1}.{out_ext}",
                    "media_bytes": crop_bytes,
                    "duration": 0,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "custom_metadata": {"Detection Confidence": round(conf, 4)},
                }
            )
        return outputs


CONVERTER = Image2FaceMediaConverter()
