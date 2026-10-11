"""Localise faces in an image and emit one face crop per detected face.

This is the *convert-in* path for the :class:`~vtscore.media.face.media_type.FaceMediaType`
half type: images are the only source of faces (faces have no native import),
so every face in the app is produced here. facenet-pytorch's MTCNN detector
finds each face above the configured confidence, and each is cropped (with
optional padding) into its own ``face``-type media, ready to be embedded in
FaceNet identity space by the same ``facenet-pytorch`` install.

**Alignment** (``align``, #4777). MTCNN also returns five landmarks per face
(eyes, nose, mouth corners), and a least-squares similarity fit of them to
:data:`FACE_TEMPLATE` says how the face sits in the photo. ``roll`` turns the
crop so the face is upright, about the box's centre and at the box's own size;
``similarity`` also re-centres and re-scales it to the template. ``none`` is the
plain box crop. Alignment changes a crop's pixels only: which faces are found
and kept, and the ``source_box`` each records, do not depend on it.

``facenet-pytorch`` is an opt-in dependency (declared in pyproject's ``DEP001``
ignore-list, installed ``--no-deps`` by the install scripts because its pins
would downgrade the app's torch/numpy stack); when it is missing the converter
degrades gracefully to producing no faces.
"""

from __future__ import annotations

import io
import logging
import math
from typing import Any, NamedTuple, Optional

import numpy as np

from vtscore.converters.base import MediaConverter, resolve_media_bytes
from vtscore.media.provenance import SOURCE_BOX_FIELD
from vtscore.plugins import PluginField

logger = logging.getLogger(__name__)

#: How a crop is turned before it is embedded: not at all, upright about its own
#: box, or upright and re-framed on the template (module docstring).
ALIGN_MODES = ("none", "roll", "similarity")

#: Where an upright face's five MTCNN landmarks sit in MTCNN's own box, as
#: fractions of the box (x right, y down): the image-left eye, the other eye, the
#: nose, the image-left mouth corner, the other. The medians over the 7,186
#: subject faces of FHIBE's one-person photos at 1024 px whose eye line is within
#: 10 degrees of level (#4777; ``scripts/experiments/fhibe/align_landmarks_4777.py
#: template``; 640 px agrees to 0.003). MTCNN's box sits a little left of the
#: eyes' centre, and the template keeps that, so an upright face's ``similarity``
#: crop frames about what the plain crop does.
FACE_TEMPLATE: tuple[tuple[float, float], ...] = (
    (0.282, 0.386),
    (0.750, 0.383),
    (0.513, 0.583),
    (0.311, 0.764),
    (0.720, 0.763),
)


class FaceDetection(NamedTuple):
    """One kept MTCNN detection."""

    box: tuple[int, int, int, int]  # the padded crop box, clamped to the image, in pixels
    conf: float
    raw: tuple[float, float, float, float]  # MTCNN's own box, unpadded
    landmarks: Optional[np.ndarray]  # (5, 2) pixels, or None when MTCNN gave none


def fit_similarity(src: Any, dst: Any) -> tuple[float, float, float, float]:
    """The least-squares similarity taking points *src* onto *dst*.

    Returns ``(a, b, tx, ty)`` with ``x' = a*x - b*y + tx`` and
    ``y' = b*x + a*y + ty``: a scale of ``hypot(a, b)`` and, with y pointing
    down, a clockwise turn of ``atan2(b, a)``.
    """
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    m = np.zeros((2 * len(s), 4))
    m[0::2] = np.column_stack([s[:, 0], -s[:, 1], np.ones(len(s)), np.zeros(len(s))])
    m[1::2] = np.column_stack([s[:, 1], s[:, 0], np.zeros(len(s)), np.ones(len(s))])
    sol, *_ = np.linalg.lstsq(m, d.reshape(-1), rcond=None)
    a, b, tx, ty = (float(v) for v in sol)
    return a, b, tx, ty


def _usable_landmarks(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float64).reshape(5, 2)
    except (TypeError, ValueError):
        return None
    return arr if np.isfinite(arr).all() else None


def crop_face(img: Any, det: FaceDetection, align: str, padding: float) -> Any:
    """The crop *det* yields from *img*, turned as *align* says (:data:`ALIGN_MODES`).

    ``none``, or a detection without landmarks, is the plain box crop. The other
    two fit :data:`FACE_TEMPLATE` (in units of a box side) to the landmarks.
    ``roll`` is the same box with the photo turned about the box's centre by the
    fit's angle, so at no turn it IS the plain crop. ``similarity`` is a square at
    the fit's own scale that frames the template's unit box padded by *padding*
    on every side, so it depends on the landmarks alone. Anything the turned
    frame takes from outside the photo is black.
    """
    from PIL import Image  # noqa: PLC0415

    x1, y1, x2, y2 = det.box
    pts = det.landmarks
    if align == "none" or pts is None:
        return img.crop((x1, y1, x2, y2))
    if align not in ALIGN_MODES:
        raise ValueError(f"unknown align mode {align!r}; expected one of {ALIGN_MODES}")
    # Template unit (u, v) -> photo pixel (a*u - b*v + tx, b*u + a*v + ty).
    a, b, tx, ty = fit_similarity(FACE_TEMPLATE, pts)
    resample = Image.Resampling.BICUBIC
    if align == "roll":
        angle = math.atan2(b, a)
        if angle == 0.0:
            return img.crop((x1, y1, x2, y2))
        c, s = math.cos(angle), math.sin(angle)
        fx1, fy1, fx2, fy2 = det.raw
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        # Output pixel (u, v) is the turned photo's (x1 + u, y1 + v): the photo's
        # point at that offset from the centre, turned back by the angle.
        qx, qy = x1 - cx, y1 - cy
        coeffs = (c, -s, c * qx - s * qy + cx, s, c, s * qx + c * qy + cy)
        return img.transform((x2 - x1, y2 - y1), Image.Transform.AFFINE, coeffs, resample, fillcolor=(0, 0, 0))
    # similarity: output pixel (i, j) is template unit (-padding + i*k, -padding + j*k).
    span = 1 + 2 * padding
    side = max(1, int(round(math.hypot(a, b) * span)))
    k = span / side
    coeffs = (a * k, -b * k, -padding * (a - b) + tx, b * k, a * k, -padding * (a + b) + ty)
    return img.transform((side, side), Image.Transform.AFFINE, coeffs, resample, fillcolor=(0, 0, 0))


class Image2FaceMediaConverter(MediaConverter):
    """Detect faces with MTCNN and emit one crop per face as ``face`` media.

    Images with **no** detected faces yield zero outputs and drop out of the
    dataset - the intended semantic for a face-only collection. Three
    tunables mirror the (now-removed) image face clipper: confidence
    ``threshold``, crop ``padding``, and ``min_size``; the fourth, ``align``,
    says how each crop is turned (module docstring).

    Each crop records where it sat in its photo as ``source_box``, the padded
    crop box normalised to the upright source frame, so the face can be
    outlined on the photo it came from.
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
            key="align",
            label="Align faces",
            field_type="select",
            description=(
                "Turn each crop upright using the face's landmarks: none keeps the plain box; "
                "roll turns the box so the eyes are level; similarity also centres and scales the face."
            ),
            options=list(ALIGN_MODES),
            default="none",
            required=False,
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
            logger.warning("image2face requires facenet-pytorch; producing no faces")
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

    def _resolve_params(self, params: dict[str, Any] | None) -> tuple[float, float, int, str]:
        threshold = float(self.get_param(params, "threshold") or 0.5)
        padding = float(self.get_param(params, "padding") or 0.0)
        min_size = int(self.get_param(params, "min_size") or 32)
        align = str(self.get_param(params, "align") or "none")
        if align not in ALIGN_MODES:
            raise ValueError(f"image2face: align must be one of {ALIGN_MODES}, not {align!r}")
        return threshold, max(0.0, padding), max(1, min_size), align

    def _detect_faces(self, img: Any, threshold: float, padding: float, min_size: int) -> list[FaceDetection]:
        """Run MTCNN on *img* and return the kept detections, highest-confidence first.

        The landmarks come from the same pass as the boxes (MTCNN computes them
        either way), so asking for them changes no box.
        """
        detector = self._make_detector()
        if detector is None:
            return []
        try:
            detected = detector.detect(img, landmarks=True)
        except Exception:
            return []
        w, h = img.size
        return self._kept_faces(detected, w, h, threshold, padding, min_size)

    def _kept_faces(
        self, detected: Any, w: int, h: int, threshold: float, padding: float, min_size: int
    ) -> list[FaceDetection]:
        """The detections of one ``detect(img, landmarks=True)`` that clear the thresholds, highest-confidence first."""
        det_boxes, det_probs, det_points = detected
        if det_boxes is None:
            return []
        if det_points is None:
            det_points = [None] * len(det_boxes)
        faces: list[FaceDetection] = []
        for box, prob, points in zip(det_boxes, det_probs, det_points):
            resolved = self._box_from_detection(box, prob, w, h, threshold, padding, min_size)
            if resolved is None:
                continue
            x1, y1, x2, y2, conf = resolved
            raw = tuple(float(v) for v in box)
            faces.append(FaceDetection((x1, y1, x2, y2), conf, raw, _usable_landmarks(points)))  # type: ignore[arg-type]
        # Highest-confidence face first so face index 0 is the "primary" face.
        faces.sort(key=lambda f: f.conf, reverse=True)
        return faces

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from vtscore.media.image.decode import decode_bounded_rgb  # noqa: PLC0415

        threshold, padding, min_size, align = self._resolve_params(params)

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

        faces = self._detect_faces(img, threshold, padding, min_size)

        stem = (media.get("filename") or "image").rsplit(".", 1)[0]
        out_ext = "png" if fmt.upper() == "PNG" else fmt.lower()
        outputs: list[dict[str, Any]] = []
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.box
            cropped = crop_face(img, face, align, padding)
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
            crop_bytes = buf.getvalue()
            outputs.append(
                {
                    "filename": f"{stem}_face_{idx + 1}.{out_ext}",
                    "media_bytes": crop_bytes,
                    "duration": 0,
                    "width": cropped.width,
                    "height": cropped.height,
                    # Where the face sat in the photo, as a fraction of the
                    # frame: the padded box, however the crop was turned. The
                    # bounded decode may be a downsample, so pixels here are not
                    # the original's (see SOURCE_BOX_FIELD).
                    SOURCE_BOX_FIELD: [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
                    "custom_metadata": {"Detection Confidence": round(face.conf, 4)},
                }
            )
        return outputs


CONVERTER = Image2FaceMediaConverter()
