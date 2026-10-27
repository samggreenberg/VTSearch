"""Localise faces in an image and emit one face crop per detected face.

This is the *convert-in* path for the :class:`~vtscore.media.face.media_type.FaceMediaType`
half type: images are the only source of faces (faces have no native import),
so every face in the app is produced here.  The photo goes to the face provider
this server is configured with (:mod:`vtscore.media.face.imagekit_bridge`: the
open SCRFD + ArcFace/MagFace ONNX weights, or a proprietary backend behind the
same interface), which detects **and embeds** every face in one pass from the
full photo.  Each face above the confidence ``threshold`` is cropped (with
optional ``padding``) into its own ``face``-type media, and leaves with its
vector already filed under the ``face`` embedder and its quality and
embedding-space tag in ``custom_metadata``, so the embed stage has nothing to
do and no face is embedded twice.

**Face is on only when this server has face models** (``imagekit_bridge.face_models``).
Without them the converter hides itself from every picker
(:attr:`Image2FaceMediaConverter.hidden_from_picker`) and a run that names it
anyway fails on its first photo, naming what is missing, rather than quietly
producing an empty dataset.

**Alignment** (``align``, #4777) turns the *stored* crop only; the vector comes
from the provider, which aligns on the full photo itself.  ``roll`` turns the
crop so the face is upright, about the box's centre and at the box's own size,
by the angle of the least-squares similarity fit of the face's five landmarks
onto the ArcFace template (:data:`imagekit.face.face_align.ARCFACE_TEMPLATE`,
the shape the provider's own embedder aligns to).  ``none`` is the plain box
crop.  Which faces are found and kept, and the ``source_box`` each records, do
not depend on it.  ``roll`` is the default: an upright crop is what a person
looks at, and it is what the ``face`` embedder re-detects on when a crop
arrives without a vector.
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

#: How a stored crop is turned: not at all, or upright about its own box (module docstring).
ALIGN_MODES = ("none", "roll")


class FaceParams(NamedTuple):
    """The converter's settings, resolved from a params dict."""

    threshold: float
    padding: float
    min_size: int
    align: str


class FaceDetection(NamedTuple):
    """One kept detection."""

    box: tuple[int, int, int, int]  # the padded crop box, clamped to the image, in pixels
    conf: float
    raw: tuple[float, float, float, float]  # the provider's own box, unpadded
    landmarks: Optional[np.ndarray]  # (5, 2) pixels, or None when the provider gave none


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

    ``none``, or a detection without landmarks, is the plain box crop.  ``roll``
    is the same box with the photo turned about the box's centre by the angle
    of the fit of the ArcFace template to the landmarks, so at no turn it IS the
    plain crop.  Anything the turned frame takes from outside the photo is
    black.  *padding* is already in ``det.box``; it is accepted so every caller
    passes the converter's settings the same way.
    """
    from PIL import Image  # noqa: PLC0415

    x1, y1, x2, y2 = det.box
    pts = det.landmarks
    if align not in ALIGN_MODES:
        raise ValueError(f"unknown align mode {align!r}; expected one of {ALIGN_MODES}")
    if align == "none" or pts is None:
        return img.crop((x1, y1, x2, y2))
    from imagekit.face.face_align import ARCFACE_TEMPLATE  # noqa: PLC0415

    # Template point (u, v) -> photo pixel (a*u - b*v + tx, b*u + a*v + ty).
    a, b, _tx, _ty = fit_similarity(ARCFACE_TEMPLATE, pts)
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
    return img.transform(
        (x2 - x1, y2 - y1), Image.Transform.AFFINE, coeffs, Image.Resampling.BICUBIC, fillcolor=(0, 0, 0)
    )


class Image2FaceMediaConverter(MediaConverter):
    """Find faces with the server's face provider and emit one crop per face as ``face`` media.

    Images with **no** detected faces yield zero outputs and drop out of the
    dataset - the intended semantic for a face-only collection.  Three
    tunables decide which faces are kept and how big each crop is: confidence
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
            description="Expand each crop by this fraction on every side, so a face is shown with some context.",
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
                "roll turns the box so the eyes are level."
            ),
            options=list(ALIGN_MODES),
            default="roll",
            required=False,
        ),
    ]

    @property
    def hidden_from_picker(self) -> bool:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Hidden while this server has no face models: Face is off (module docstring)."""
        from vtscore.media.face import imagekit_bridge as bridge  # noqa: PLC0415

        return not bridge.face_models().available

    @property
    def source_type(self) -> str:
        return "image"

    @property
    def target_type(self) -> str:
        return "face"

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _box_from_detection(
        self, box: Any, prob: Any, w: int, h: int, threshold: float, padding: float, min_size: int
    ) -> tuple[int, int, int, int, float] | None:
        """Turn one detection's ``(box, confidence)`` into a padded, clamped pixel box.

        *box* is ``[x1, y1, x2, y2]`` in pixel coordinates.  Returns
        ``(x1, y1, x2, y2, conf)``, or ``None`` when the detection is unusable,
        below *threshold*, or smaller than *min_size* on either axis after
        padding.
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

    def _resolve_params(self, params: dict[str, Any] | None) -> FaceParams:
        threshold = float(self.get_param(params, "threshold") or 0.5)
        padding = float(self.get_param(params, "padding") or 0.0)
        min_size = int(self.get_param(params, "min_size") or 32)
        align = str(self.get_param(params, "align") or "roll")
        if align not in ALIGN_MODES:
            raise ValueError(f"image2face: align must be one of {ALIGN_MODES}, not {align!r}")
        return FaceParams(threshold, max(0.0, padding), max(1, min_size), align)

    def _find_faces(
        self, img: Any, p: FaceParams, media_id: str, strict: bool = False
    ) -> list[tuple[FaceDetection, dict[str, Any]]]:
        """Detect **and embed** every face through the server's face provider (module docstring).

        Returns the kept detections, highest-confidence first, each with the
        extra output fields its crop leaves with: the vector filed under the
        ``face`` embedder, and the quality and compat tag for its
        ``custom_metadata``.  A server without face models, or whose models do
        not load, raises (:class:`~vtscore.media.face.imagekit_bridge.FaceProviderUnconfigured`
        names what is missing).  A provider that fails on one photo yields no
        faces for it, with a warning; with *strict* it raises instead, for a
        caller that must fail loudly.
        """
        from vtscore.media.face import imagekit_bridge as bridge  # noqa: PLC0415

        pipeline = bridge.get_face_pipeline()
        try:
            records = pipeline.process(np.asarray(img), confidence_threshold=p.threshold, media_id=media_id)
        except Exception:
            if strict:
                raise
            logger.warning(
                "image2face: the %s face provider failed on %s", bridge.provider_name(), media_id, exc_info=True
            )
            return []
        w, h = img.size
        found: list[tuple[FaceDetection, dict[str, Any]]] = []
        for rec in records:
            resolved = self._box_from_detection(rec.bbox, rec.confidence, w, h, p.threshold, p.padding, p.min_size)
            if resolved is None:
                continue
            x1, y1, x2, y2, conf = resolved
            raw = tuple(float(v) for v in rec.bbox)
            face = FaceDetection((x1, y1, x2, y2), conf, raw, _usable_landmarks(rec.landmarks))  # type: ignore[arg-type]
            extra = {
                "embeddings": {bridge.EMBEDDER_NAME: rec.unit_embedding()},
                "embedder": bridge.EMBEDDER_NAME,
                "custom_metadata": {
                    bridge.QUALITY_METADATA_KEY: round(float(rec.quality), 3),
                    bridge.COMPAT_METADATA_KEY: rec.compat,
                },
            }
            found.append((face, extra))
        # Highest-confidence face first so face index 0 is the "primary" face.
        found.sort(key=lambda pair: pair[0].conf, reverse=True)
        return found

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from vtscore.media.image.decode import decode_bounded_rgb  # noqa: PLC0415

        p = self._resolve_params(params)

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

        stem = (media.get("filename") or "image").rsplit(".", 1)[0]
        found = self._find_faces(img, p, media.get("filename") or stem)

        out_ext = "png" if fmt.upper() == "PNG" else fmt.lower()
        outputs: list[dict[str, Any]] = []
        for idx, (face, extra) in enumerate(found):
            x1, y1, x2, y2 = face.box
            cropped = crop_face(img, face, p.align, p.padding)
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
                    "custom_metadata": {"Detection Confidence": round(face.conf, 4), **extra["custom_metadata"]},
                    # The provider embedded the face; its vector rides with the crop.
                    "embeddings": extra["embeddings"],
                    "embedder": extra["embedder"],
                }
            )
        return outputs


CONVERTER = Image2FaceMediaConverter()
