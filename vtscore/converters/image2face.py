"""Localise faces in an image and emit one face crop per detected face.

This is the *convert-in* path for the :class:`~vtscore.media.face.media_type.FaceMediaType`
half type: images are the only source of faces (faces have no native import),
so every face in the app is produced here. facenet-pytorch's MTCNN detector
finds each face above the configured confidence, and each is cropped (with
optional padding) into its own ``face``-type media, ready to be embedded in
FaceNet identity space by the same ``facenet-pytorch`` install.

``facenet-pytorch`` is an opt-in dependency (declared in pyproject's ``DEP001``
ignore-list, installed ``--no-deps`` by the install scripts because its pins
would downgrade the app's torch/numpy stack); when it is missing the converter
degrades gracefully to producing no faces.
"""

from __future__ import annotations

import io
from typing import Any, Optional

from vtscore.converters.base import MediaConverter, resolve_media_bytes
from vtscore.plugins import PluginField


class Image2FaceMediaConverter(MediaConverter):
    """Detect faces with MTCNN and emit one crop per face as ``face`` media.

    Images with **no** detected faces yield zero outputs and drop out of the
    dataset - the intended semantic for a face-only collection. The three
    tunables mirror the (now-removed) image face clipper: confidence
    ``threshold``, crop ``padding``, and ``min_size``.
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
    ]

    def __init__(self) -> None:
        super().__init__()
        # Lazily built, then reused across every image in a conversion run.
        self._detector: Optional[Any] = None

    @property
    def source_type(self) -> str:
        return "image"

    @property
    def target_type(self) -> str:
        return "face"

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _make_detector(self) -> Optional[Any]:
        """Return a cached, keep-all MTCNN detector, or ``None`` if unavailable.

        MTCNN ships inside ``facenet-pytorch`` (the same install that provides
        the FaceNet embedder) and carries its own bundled weights, so there is
        no separate model download. The detector is confidence-independent — the
        per-face ``threshold`` is applied as a post-filter in
        :meth:`_box_from_detection` — so a single instance is reused across every
        image in a conversion run.
        """
        if self._detector is not None:
            return self._detector
        try:
            from facenet_pytorch import MTCNN  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        except ImportError:
            print("Image2FaceMediaConverter requires facenet-pytorch; producing no faces")
            return None
        from vtscore.config import resolve_device  # noqa: PLC0415

        self._detector = MTCNN(
            keep_all=True,
            select_largest=False,
            post_process=False,
            device=resolve_device(),
        )
        return self._detector

    def _box_from_detection(
        self, box: Any, prob: Any, w: int, h: int, threshold: float, padding: float, min_size: int
    ) -> tuple[int, int, int, int, float] | None:
        """Turn one MTCNN ``(box, prob)`` pair into a padded, clamped pixel box.

        MTCNN returns boxes as ``[x1, y1, x2, y2]`` already in pixel
        coordinates. Returns ``(x1, y1, x2, y2, conf)``, or ``None`` when the
        detection is unusable, below *threshold*, or smaller than *min_size* on
        either axis after padding.
        """
        try:
            conf = float(prob)
        except (TypeError, ValueError):
            return None
        if conf < threshold:
            return None
        try:
            fx1, fy1, fx2, fy2 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        pad_x = (fx2 - fx1) * padding
        pad_y = (fy2 - fy1) * padding
        x1 = max(0, int(round(fx1 - pad_x)))
        y1 = max(0, int(round(fy1 - pad_y)))
        x2 = min(w, int(round(fx2 + pad_x)))
        y2 = min(h, int(round(fy2 + pad_y)))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            return None
        return (x1, y1, x2, y2, conf)

    def _resolve_params(self, params: dict[str, Any] | None) -> tuple[float, float, int]:
        threshold = float(self.get_param(params, "threshold") or 0.5)
        padding = float(self.get_param(params, "padding") or 0.0)
        min_size = int(self.get_param(params, "min_size") or 32)
        return threshold, max(0.0, padding), max(1, min_size)

    def _detect_boxes(
        self, img: Any, w: int, h: int, threshold: float, padding: float, min_size: int
    ) -> list[tuple[int, int, int, int, float]]:
        """Run MTCNN on *img* and return usable pixel boxes, highest-confidence first."""
        detector = self._make_detector()
        if detector is None:
            return []
        try:
            det_boxes, det_probs = detector.detect(img)
        except Exception:
            return []
        if det_boxes is None:
            return []

        boxes: list[tuple[int, int, int, int, float]] = []
        for box, prob in zip(det_boxes, det_probs):
            resolved = self._box_from_detection(box, prob, w, h, threshold, padding, min_size)
            if resolved is not None:
                boxes.append(resolved)
        # Highest-confidence face first so face index 0 is the "primary" face.
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from vtscore.media.image.decode import decode_bounded_rgb  # noqa: PLC0415

        threshold, padding, min_size = self._resolve_params(params)

        media_bytes = resolve_media_bytes(media)
        if not media_bytes:
            return []
        try:
            # Bounded decode: detection *and* the face crops below both come off
            # this one image, so they stay in a single consistent coordinate
            # space and a gigapixel group photo never has to fit in memory.
            img, _scale = decode_bounded_rgb(media_bytes)
        except Exception:
            return []
        img_w, img_h = img.size
        fmt = img.format or "PNG"

        boxes = self._detect_boxes(img, img_w, img_h, threshold, padding, min_size)

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
