"""Image I/O, the image-tensor convention, and the small transform registry.

The **image tensor** every :mod:`imagekit` API takes and returns is one
convention, stated once here: a ``numpy`` array of shape ``(H, W, 3)``,
``uint8``, **RGB** channel order.  :func:`read_image_tensor` and
:func:`decode_image_buffer` produce it; :func:`ensure_image_tensor` coerces
the near-misses (grayscale, RGBA, a non-contiguous view) and refuses the rest.

The transform registry is how a model's JSON config describes its own
preprocessing (``Normalize`` / ``Resize`` / ``ToTensor`` here, ``Align`` in
:mod:`imagekit.face.face_align`).  A transform maps one :class:`Sample` to the
next, so a step that changes geometry (``Resize``) can also record the scale
that maps model coordinates back onto the input image.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: ``(H, W, 3)`` ``uint8`` RGB.  A type alias for documentation; it is a plain array.
ImageTensor = np.ndarray

#: File extensions the CLIs treat as images when scanning a directory.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


# ---------------------------------------------------------------------------
# Image tensors
# ---------------------------------------------------------------------------


def ensure_image_tensor(image: Any) -> ImageTensor:
    """Return *image* as an ``(H, W, 3)`` ``uint8`` RGB array.

    Accepts an array that already is one, a grayscale ``(H, W)`` array (channels
    are replicated), or an ``(H, W, 4)`` RGBA array (alpha is dropped).  Anything
    else - a wrong rank, zero size, a float array - raises :class:`ValueError`,
    because every model downstream would otherwise fail somewhere less clear.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"image tensor must be (H, W, 3); got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"image tensor must have 3 channels; got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"image tensor must be uint8; got {arr.dtype}")
    return np.ascontiguousarray(arr)


def decode_image_buffer(buffer: bytes | bytearray | memoryview) -> ImageTensor:
    """Decode an encoded image (JPEG, PNG, ...) held in memory.

    Raises :class:`ValueError` when the bytes are not an image OpenCV can decode.
    """
    data = np.frombuffer(bytes(buffer), dtype=np.uint8)
    if data.size == 0:
        raise ValueError("cannot decode an empty image buffer")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot decode image buffer: not a supported image format")
    return ensure_image_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def read_image_tensor(image_filepath: str | Path) -> ImageTensor:
    """Read the image at *image_filepath* into an image tensor.

    Raises :class:`FileNotFoundError` for a missing file and :class:`ValueError`
    for a file that is not a decodable image.
    """
    path = Path(image_filepath)
    if not path.is_file():
        raise FileNotFoundError(f"no such image file: {path}")
    return decode_image_buffer(path.read_bytes())


def letterbox(image: ImageTensor, height: int, width: int, pad_value: int = 0) -> tuple[ImageTensor, float]:
    """Resize *image* to fit inside ``height x width`` keeping its aspect, pad the rest.

    The image lands in the top-left corner and the padding (``pad_value``) fills
    the bottom and right, so a model coordinate maps back to the input by one
    division: ``input = model / scale``.  Returns the padded image and that scale.
    """
    src_h, src_w = image.shape[:2]
    scale = min(height / src_h, width / src_w)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), pad_value, dtype=image.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def resize_within(image: ImageTensor, max_size: int) -> tuple[ImageTensor, float]:
    """Shrink *image* so its longer side is at most *max_size*; never enlarge.

    Returns the image and the scale applied (``1.0`` when nothing changed).
    """
    src_h, src_w = image.shape[:2]
    longest = max(src_h, src_w)
    if longest <= max_size:
        return image, 1.0
    scale = max_size / longest
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, best score first.

    *boxes* is ``(N, 4)`` as ``x1, y1, x2, y2``; *scores* is ``(N,)``.
    """
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def clamp_box(box: Any, width: int, height: int) -> tuple[float, float, float, float]:
    """Clamp an ``x1, y1, x2, y2`` box (any 4-sequence or array) to a ``width x height`` image."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """What flows through a transform chain: the image, plus what the chain did to it.

    ``scale`` is the factor applied by any resize so far (model pixels per input
    pixel); ``tensor`` is set once ``ToTensor`` has produced the model input.
    ``extra`` carries per-call context a transform may need (``Align`` reads the
    detection's landmarks and box from it).
    """

    image: np.ndarray
    scale: float = 1.0
    tensor: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


Transform = Callable[[Sample], Sample]
TransformFactory = Callable[[Mapping[str, Any]], Transform]

_TRANSFORMS: dict[str, TransformFactory] = {}


def register_transform(name: str, factory: TransformFactory) -> None:
    """Make ``{"name": name, "args": {...}}`` in a model config build *factory(args)*."""
    _TRANSFORMS[name] = factory


def build_transforms(specs: Iterable[Mapping[str, Any]]) -> list[Transform]:
    """Instantiate the transform chain a model config lists.

    Each spec is ``{"name": <registered name>, "args": {...}}``; ``args`` may be
    omitted.  An unknown name raises :class:`KeyError` naming it.
    """
    chain: list[Transform] = []
    for spec in specs:
        name = str(spec.get("name", ""))
        if name not in _TRANSFORMS:
            raise KeyError(f"unknown transform {name!r}; registered: {sorted(_TRANSFORMS)}")
        chain.append(_TRANSFORMS[name](spec.get("args") or {}))
    return chain


def apply_transforms(chain: Iterable[Transform], sample: Sample) -> Sample:
    """Run *sample* through *chain* in order."""
    for transform in chain:
        sample = transform(sample)
    return sample


def _per_channel(values: Any, n: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, n)
    if arr.size != n:
        raise ValueError(f"expected {n} per-channel values, got {arr.size}")
    return arr


def _normalize_factory(args: Mapping[str, Any]) -> Transform:
    """``Normalize``: ``(x - mean) * scale`` per channel, in two spellings.

    ``mean_vals`` / ``scale_vals`` work on the 0-255 pixel scale (SCRFD's
    ``(x - 127.5) / 128``).  ``mean`` / ``std`` are torchvision-style, on the
    0-1 scale (``(x / 255 - mean) / std``).  Either way the output is float32.
    """
    if "mean" in args or "std" in args:
        mean = _per_channel(args.get("mean", 0.0)) * 255.0
        std = _per_channel(args.get("std", 1.0)) * 255.0
        scale = 1.0 / std
    else:
        mean = _per_channel(args.get("mean_vals", 0.0))
        scale = _per_channel(args.get("scale_vals", 1.0))

    def normalize(sample: Sample) -> Sample:
        img = sample.image.astype(np.float32)
        sample.image = (img - mean) * scale
        return sample

    return normalize


def _resize_factory(args: Mapping[str, Any]) -> Transform:
    """``Resize``: letterbox to ``height x width`` (``preserve_aspect``) or a plain resize.

    With ``preserve_aspect`` and ``pad`` the image is letterboxed (top-left,
    padded bottom/right); with ``preserve_aspect`` alone it is shrunk to fit
    with no padding; without it, it is stretched to the target.  ``max_size``
    alone (no ``height``/``width``) only shrinks a longer side above it.
    ``sample.scale`` is multiplied by the scale applied.
    """
    height = args.get("height")
    width = args.get("width")
    preserve = bool(args.get("preserve_aspect", True))
    pad = bool(args.get("pad", 0))
    max_size = args.get("max_size")

    def resize(sample: Sample) -> Sample:
        img = sample.image
        if height is None or width is None:
            if max_size is not None:
                img, scale = resize_within(img, int(max_size))
                sample.image, sample.scale = img, sample.scale * scale
            return sample
        h, w = int(height), int(width)
        if preserve and pad:
            img, scale = letterbox(img, h, w)
        elif preserve:
            scale = min(h / img.shape[0], w / img.shape[1])
            size = (max(1, int(round(img.shape[1] * scale))), max(1, int(round(img.shape[0] * scale))))
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        sample.image, sample.scale = img, sample.scale * scale
        return sample

    return resize


def _to_tensor_factory(args: Mapping[str, Any]) -> Transform:  # noqa: ARG001 - no args; registry signature
    """``ToTensor``: ``(H, W, 3)`` -> ``(1, 3, H, W)`` float32, contiguous."""

    def to_tensor(sample: Sample) -> Sample:
        img = sample.image.astype(np.float32, copy=False)
        sample.tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...])
        return sample

    return to_tensor


def _channel_swap_factory(args: Mapping[str, Any]) -> Transform:  # noqa: ARG001 - no args; registry signature
    """``ChannelSwap``: RGB -> BGR (or back), for weights trained on OpenCV-ordered input."""

    def channel_swap(sample: Sample) -> Sample:
        sample.image = np.ascontiguousarray(sample.image[:, :, ::-1])
        return sample

    return channel_swap


register_transform("Normalize", _normalize_factory)
register_transform("Resize", _resize_factory)
register_transform("ToTensor", _to_tensor_factory)
register_transform("ChannelSwap", _channel_swap_factory)


# ---------------------------------------------------------------------------
# Model files and their JSON configs
# ---------------------------------------------------------------------------


def resolve_model_file(bundle: str | Path, suffix: str = ".onnx") -> Path:
    """The model file a ``model_bundle`` argument names.

    A bundle is either the model file itself or a directory holding exactly one
    file with *suffix* (plus its JSON config and whatever else ships with it).
    Model paths are always arguments, never defaults, so a missing one is an
    error here rather than a silent download somewhere else.
    """
    path = Path(bundle)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.suffix.lower() == suffix)
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f"model bundle {path} must hold exactly one {suffix} file; found {[c.name for c in candidates]}"
        )
    raise FileNotFoundError(f"no such model file or bundle directory: {path}")


def load_model_config(
    model_path: str | Path,
    config_path: str | Path | None = None,
    default_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the JSON config that describes *model_path*.

    Looked up in order: an explicit *config_path* (which must exist); the
    sidecar ``<model stem>.json`` beside the model; *default_path* (a config the
    package ships).  A config found nowhere raises :class:`FileNotFoundError`
    naming every place it looked, so the fix is one file write away.
    """
    import json  # noqa: PLC0415

    model = Path(model_path)
    looked: list[Path] = []
    if config_path is not None:
        looked.append(Path(config_path))
    else:
        looked.append(model.with_suffix(".json"))
        if default_path is not None:
            looked.append(Path(default_path))
    for candidate in looked:
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError(f"{candidate}: model config must be a JSON object")
            return loaded
    raise FileNotFoundError(f"no model config for {model}; looked at {[str(p) for p in looked]}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def collect_image_paths(
    directory: str | Path | None = None,
    list_file: str | Path | None = None,
    image: str | Path | None = None,
) -> list[Path]:
    """The image files a CLI's ``--dir`` / ``--list`` / ``--img`` name, in a stable order.

    ``--dir`` walks recursively and keeps :data:`IMAGE_EXTENSIONS`; ``--list`` is
    one path per line (blank lines and ``#`` comments skipped); ``--img`` is one
    file.  Exactly one of the three must be given.
    """
    given = [v for v in (directory, list_file, image) if v is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of directory, list_file or image")
    if directory is not None:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {root}")
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if list_file is not None:
        lines = Path(list_file).read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip() and not line.lstrip().startswith("#")]
    assert image is not None
    return [Path(image)]
