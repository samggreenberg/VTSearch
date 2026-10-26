"""``norm_crop``: the ArcFace five-point alignment every embedder here expects.

A face embedder is trained on faces warped so that the eyes, nose and mouth
corners land on fixed template points in a 112 x 112 crop.  The same warp at
inference is what makes its vectors comparable, so alignment is part of the
embedding contract, not a preprocessing nicety.

:func:`norm_crop` fits the similarity (rotation, uniform scale, translation)
that takes a detection's five landmarks onto :data:`ARCFACE_TEMPLATE` by least
squares, and warps the whole image through it.  A detection without landmarks
falls back to its box, cropped and resized, so every face still gets a crop of
the right size - a worse one, but never a missing one.

The ``Align`` transform exposes the same thing to a model config:
``{"name": "Align", "args": {"height": 112, "width": 112, "fallback": "crop"}}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from imagekit.util import ImageTensor, Sample, Transform, clamp_box, register_transform

#: The standard ArcFace reference points for a 112 x 112 crop, in
#: :data:`~imagekit.face.records.LANDMARK_NAMES` order (left eye first).
ARCFACE_TEMPLATE: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

#: The crop side :data:`ARCFACE_TEMPLATE` is expressed in.
TEMPLATE_SIZE = 112

#: What ``norm_crop`` does when a detection has no usable landmarks.
FALLBACKS = ("crop", "error")


def similarity_transform(src: Any, dst: Any) -> np.ndarray:
    """The ``(2, 3)`` affine of the least-squares similarity taking *src* points onto *dst*.

    A similarity is ``x' = a*x - b*y + tx``, ``y' = b*x + a*y + ty``: uniform
    scale ``hypot(a, b)``, a rotation, a translation, and no reflection or
    shear, which is what keeps a face a face.  Solved as one linear least
    squares in ``(a, b, tx, ty)``.
    """
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if s.shape != d.shape or len(s) < 2:
        raise ValueError(f"need matching point sets of at least two points; got {s.shape} and {d.shape}")
    n = len(s)
    m = np.zeros((2 * n, 4))
    m[0::2] = np.column_stack([s[:, 0], -s[:, 1], np.ones(n), np.zeros(n)])
    m[1::2] = np.column_stack([s[:, 1], s[:, 0], np.zeros(n), np.ones(n)])
    (a, b, tx, ty), *_ = np.linalg.lstsq(m, d.reshape(-1), rcond=None)
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def alignment_matrix(landmarks: Any, height: int = TEMPLATE_SIZE, width: int = TEMPLATE_SIZE) -> np.ndarray:
    """The warp taking *landmarks* onto :data:`ARCFACE_TEMPLATE` scaled to ``height x width``."""
    template = ARCFACE_TEMPLATE * np.array([width / TEMPLATE_SIZE, height / TEMPLATE_SIZE], dtype=np.float32)
    return similarity_transform(landmarks, template)


def crop_box(image: ImageTensor, bbox: Any, height: int, width: int) -> ImageTensor:
    """Crop *bbox* (clamped to the image) and resize it to ``height x width``.

    A box that clamps to nothing (off-image, or zero area) yields a black crop
    rather than an error: it is the fallback's fallback.
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(bbox, img_w, img_h)
    ix1, iy1 = int(np.floor(x1)), int(np.floor(y1))
    ix2, iy2 = int(np.ceil(x2)), int(np.ceil(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((height, width, 3), dtype=image.dtype)
    return cv2.resize(image[iy1:iy2, ix1:ix2], (width, height), interpolation=cv2.INTER_LINEAR)


def norm_crop(
    image: ImageTensor,
    landmarks: Any,
    image_size: int = TEMPLATE_SIZE,
    *,
    bbox: Any = None,
    fallback: str = "crop",
    height: int | None = None,
    width: int | None = None,
) -> ImageTensor:
    """Warp *image* so *landmarks* land on the ArcFace template; ``image_size`` square RGB.

    Args:
        image: The **full** image, as an image tensor.
        landmarks: ``(5, 2)`` pixels in landmark order, or ``None``.
        image_size: Side of the square crop (the template scales with it).
        bbox: The detection box, used only when the landmarks are missing.
        fallback: ``"crop"`` (box crop + resize when landmarks are missing;
            the whole image when the box is missing too) or ``"error"``
            (raise :class:`ValueError` instead).
        height, width: A non-square crop, overriding ``image_size``.

    Pixels the warp takes from outside the image are black.
    """
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}, not {fallback!r}")
    out_h = int(height if height is not None else image_size)
    out_w = int(width if width is not None else image_size)
    pts = None if landmarks is None else np.asarray(landmarks, dtype=np.float32)
    if pts is None or pts.shape != (5, 2) or not bool(np.isfinite(pts).all()):
        if fallback == "error":
            raise ValueError("norm_crop needs (5, 2) finite landmarks and fallback='error' forbids the box crop")
        if bbox is None:
            return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop_box(image, bbox, out_h, out_w)
    matrix = alignment_matrix(pts, out_h, out_w)
    return cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def _align_factory(args: Mapping[str, Any]) -> Transform:
    """``Align``: :func:`norm_crop` as a transform; reads landmarks/bbox from ``sample.extra``."""
    height = int(args.get("height", TEMPLATE_SIZE))
    width = int(args.get("width", TEMPLATE_SIZE))
    fallback = str(args.get("fallback", "crop"))
    if fallback not in FALLBACKS:
        raise ValueError(f"Align fallback must be one of {FALLBACKS}, not {fallback!r}")

    def align(sample: Sample) -> Sample:
        sample.image = norm_crop(
            sample.image,
            sample.extra.get("landmarks"),
            bbox=sample.extra.get("bbox"),
            fallback=fallback,
            height=height,
            width=width,
        )
        return sample

    return align


register_transform("Align", _align_factory)
