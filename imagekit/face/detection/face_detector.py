"""``FaceDetector``: SCRFD (primary) and RetinaFace-ResNet50 face detection on ONNX Runtime.

One class, two decoders, chosen by the ``model.name`` field of the JSON config
that sits beside the ``.onnx`` file (:func:`~imagekit.util.load_model_config`):

* **SCRFD** (any other name; the packaged default is ``scrfd_10g_gnkps``).
  Letterbox to the config's ``Resize`` (640 x 640 by default), normalise
  ``(x - 127.5) / 128``, NCHW float32.  The heads at strides 8/16/32 give, per
  anchor, a score, a box as four distances from the anchor centre, and five
  keypoints as offsets; :func:`decode_scrfd` turns those into boxes and
  landmarks in model-input pixels.
* **RetinaFace** (``model.name`` starting with ``retinaface``).  Prior boxes
  from ``min_sizes`` x ``steps``, ``loc`` / ``conf`` / ``landms`` heads decoded
  with the usual variances (:func:`decode_retinaface`); the config's
  transforms carry ImageNet normalisation and a 2048 px cap.

After either decoder: score filter, ``nms_pre`` per level, NMS at
``iou_threshold``, ``topk`` cap, and the division by the resize scale that maps
everything back onto the input image (:func:`postprocess`).

Initialise-before-use: :meth:`FaceDetector.detect` before
:meth:`FaceDetector.initialize` raises ``"Must initialize FaceDetector first!"``,
and ``initialize()`` raises on any failure rather than leaving a half-built
detector behind.

MTCNN is the legacy alternative and is deliberately not implemented here; see
the README.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from imagekit.face.records import FaceDetection
from imagekit.runtime import RuntimeOptions
from imagekit.util import (
    ImageTensor,
    Sample,
    Transform,
    apply_transforms,
    build_transforms,
    clamp_box,
    decode_image_buffer,
    ensure_image_tensor,
    load_model_config,
    nms,
    resolve_model_file,
    resize_within,
)

#: The SCRFD config this package ships, used when no ``<model>.json`` sits beside the weights.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("scrfd_10g_gnkps.json")

#: The reference RetinaFace-ResNet50 config this package ships (copy it beside such weights).
RETINAFACE_CONFIG_PATH = Path(__file__).with_name("retinaface_r50.json")

NOT_INITIALIZED = "Must initialize FaceDetector first!"

_RETINAFACE_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_RETINAFACE_VARIANCES = (0.1, 0.2)


@dataclass(frozen=True)
class DetectorConfig:
    """The ``model`` block of a detector's JSON config, with defaults filled in."""

    name: str = "scrfd_10g_gnkps"
    feat_stride: tuple[int, ...] = (8, 16, 32)
    num_anchors: int = 2
    score_threshold: float = 0.5
    iou_threshold: float = 0.45
    nms_pre: int = 1000
    max_nms: int = 30000
    topk: int = 400
    min_image_size: int = 12
    max_image_size: int = 2048
    transforms: tuple[Mapping[str, Any], ...] = ()
    #: RetinaFace only: anchor sizes per stride, and the box/landmark variances.
    min_sizes: tuple[tuple[int, ...], ...] = _RETINAFACE_MIN_SIZES
    variances: tuple[float, float] = _RETINAFACE_VARIANCES

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DetectorConfig":
        """Build from a loaded config; the ``model`` wrapper is optional."""
        model = raw.get("model", raw)
        if not isinstance(model, Mapping):
            raise ValueError("detector config: 'model' must be a JSON object")
        defaults = cls()
        return cls(
            name=str(model.get("name", defaults.name)),
            feat_stride=tuple(int(s) for s in model.get("feat_stride", defaults.feat_stride)),
            num_anchors=int(model.get("num_anchors", defaults.num_anchors)),
            score_threshold=float(model.get("score_threshold", defaults.score_threshold)),
            iou_threshold=float(model.get("iou_threshold", defaults.iou_threshold)),
            nms_pre=int(model.get("nms_pre", defaults.nms_pre)),
            max_nms=int(model.get("max_nms", defaults.max_nms)),
            topk=int(model.get("topk", defaults.topk)),
            min_image_size=int(model.get("min_image_size", defaults.min_image_size)),
            max_image_size=int(model.get("max_image_size", defaults.max_image_size)),
            transforms=tuple(model.get("transforms", ())),
            min_sizes=tuple(tuple(int(v) for v in level) for level in model.get("min_sizes", defaults.min_sizes)),
            variances=tuple(float(v) for v in model.get("variances", defaults.variances)),  # type: ignore[arg-type]
        )

    @property
    def backend(self) -> str:
        """``"retinaface"`` or ``"scrfd"``, from the model name."""
        return "retinaface" if self.name.lower().startswith("retinaface") else "scrfd"


# ---------------------------------------------------------------------------
# SCRFD decoding
# ---------------------------------------------------------------------------


def scrfd_anchor_centers(height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
    """``(height * width * num_anchors, 2)`` anchor centres, in input pixels, row-major, anchors innermost."""
    ys, xs = np.mgrid[:height, :width]
    centers = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * float(stride)
    return np.repeat(centers, num_anchors, axis=0)


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Boxes from anchor centres and ``(left, top, right, bottom)`` distances."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """``(N, 5, 2)`` keypoints from anchor centres and ``(N, 10)`` x/y offsets."""
    offsets = distance.reshape(-1, distance.shape[1] // 2, 2)
    return offsets + points[:, None, :]


def _flat(output: np.ndarray, last: int) -> np.ndarray:
    """``(N, last)`` from an output that may carry a leading batch dim."""
    return np.asarray(output, dtype=np.float32).reshape(-1, last)


def _topn(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the *n* highest *scores* (all of them when ``n <= 0``)."""
    if n <= 0 or len(scores) <= n:
        return np.arange(len(scores))
    return np.argpartition(-scores, n - 1)[:n]


def decode_scrfd(
    outputs: Sequence[np.ndarray],
    input_h: int,
    input_w: int,
    cfg: DetectorConfig,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Decode SCRFD head outputs into ``(boxes, scores, kps)`` in model-input pixels.

    *outputs* are the session outputs in export order: one score map per
    stride, then one box map per stride, then (with keypoints) one keypoint
    map per stride.  Each level is score-filtered at *threshold* and capped at
    ``cfg.nms_pre`` candidates before the levels are concatenated.
    """
    fmc = len(cfg.feat_stride)
    if len(outputs) == 3 * fmc:
        use_kps = True
    elif len(outputs) == 2 * fmc:
        use_kps = False
    else:
        raise ValueError(f"SCRFD model has {len(outputs)} outputs; expected {2 * fmc} or {3 * fmc} for {fmc} strides")
    boxes_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    kps_all: list[np.ndarray] = []
    for idx, stride in enumerate(cfg.feat_stride):
        scores = _flat(outputs[idx], 1)[:, 0]
        bbox_preds = _flat(outputs[idx + fmc], 4) * stride
        centers = scrfd_anchor_centers(input_h // stride, input_w // stride, stride, cfg.num_anchors)
        if len(centers) != len(scores):
            raise ValueError(
                f"SCRFD stride {stride}: {len(scores)} scores for {len(centers)} anchors "
                f"(input {input_h}x{input_w}, num_anchors {cfg.num_anchors})"
            )
        pos = np.where(scores >= threshold)[0]
        pos = pos[_topn(scores[pos], cfg.nms_pre)]
        boxes_all.append(distance2bbox(centers[pos], bbox_preds[pos]))
        scores_all.append(scores[pos])
        if use_kps:
            kps_preds = _flat(outputs[idx + 2 * fmc], 10) * stride
            kps_all.append(distance2kps(centers[pos], kps_preds[pos]))
    boxes = np.concatenate(boxes_all) if boxes_all else np.zeros((0, 4), np.float32)
    scores = np.concatenate(scores_all) if scores_all else np.zeros((0,), np.float32)
    kps = np.concatenate(kps_all) if use_kps and kps_all else None
    return boxes, scores, kps


# ---------------------------------------------------------------------------
# RetinaFace decoding
# ---------------------------------------------------------------------------


def retinaface_priors(input_h: int, input_w: int, cfg: DetectorConfig) -> np.ndarray:
    """``(N, 4)`` prior boxes as normalised ``(cx, cy, w, h)``, in head order."""
    priors: list[np.ndarray] = []
    for step, sizes in zip(cfg.feat_stride, cfg.min_sizes):
        f_h = int(np.ceil(input_h / step))
        f_w = int(np.ceil(input_w / step))
        ys, xs = np.mgrid[:f_h, :f_w]
        cx = ((xs + 0.5) * step / input_w).reshape(-1)
        cy = ((ys + 0.5) * step / input_h).reshape(-1)
        for size in sizes:
            s_kx = np.full_like(cx, size / input_w)
            s_ky = np.full_like(cy, size / input_h)
            priors.append(np.stack([cx, cy, s_kx, s_ky], axis=-1))
        # Interleave the anchors of one cell: (cell, size) is the head's order.
        cell = np.stack(priors[-len(sizes) :], axis=1).reshape(-1, 4)
        del priors[-len(sizes) :]
        priors.append(cell)
    return np.concatenate(priors).astype(np.float32)


def _retinaface_heads(outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Identify ``loc`` / ``conf`` / ``landms`` by their last dimension."""
    loc = conf = landms = None
    for out in outputs:
        arr = np.asarray(out, dtype=np.float32)
        last = arr.shape[-1]
        if last == 4:
            loc = arr.reshape(-1, 4)
        elif last == 2:
            conf = arr.reshape(-1, 2)
        elif last == 10:
            landms = arr.reshape(-1, 10)
    if loc is None or conf is None:
        raise ValueError("RetinaFace model must output loc (…, 4) and conf (…, 2) heads")
    return loc, conf, landms


def decode_retinaface(
    outputs: Sequence[np.ndarray],
    input_h: int,
    input_w: int,
    cfg: DetectorConfig,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Decode RetinaFace heads into ``(boxes, scores, kps)`` in model-input pixels."""
    loc, conf, landms = _retinaface_heads(outputs)
    priors = retinaface_priors(input_h, input_w, cfg)
    if len(priors) != len(loc):
        raise ValueError(f"RetinaFace: {len(loc)} predictions for {len(priors)} priors (input {input_h}x{input_w})")
    v0, v1 = cfg.variances
    scores = conf[:, 1]
    pos = np.where(scores >= threshold)[0]
    pos = pos[_topn(scores[pos], cfg.nms_pre)]
    p, locs = priors[pos], loc[pos]
    centers = p[:, :2] + locs[:, :2] * v0 * p[:, 2:]
    sizes = p[:, 2:] * np.exp(locs[:, 2:] * v1)
    scale = np.array([input_w, input_h], dtype=np.float32)
    boxes = np.concatenate([(centers - sizes / 2) * scale, (centers + sizes / 2) * scale], axis=1)
    kps = None
    if landms is not None:
        offsets = landms[pos].reshape(-1, 5, 2)
        kps = (p[:, None, :2] + offsets * v0 * p[:, None, 2:]) * scale
    return boxes, scores[pos], kps


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------


def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray | None,
    scale: float,
    cfg: DetectorConfig,
    image_hw: tuple[int, int],
) -> list[FaceDetection]:
    """NMS, the ``topk`` cap, and the map back to input-image pixels; best first."""
    if len(scores) == 0:
        return []
    keep = _topn(scores, cfg.max_nms)
    boxes, scores = boxes[keep], scores[keep]
    kps = None if kps is None else kps[keep]
    order = nms(boxes, scores, cfg.iou_threshold)
    if cfg.topk > 0:
        order = order[: cfg.topk]
    img_h, img_w = image_hw
    detections: list[FaceDetection] = []
    for i in order:
        box = clamp_box(boxes[i] / scale, img_w, img_h)
        landmarks = None if kps is None else (kps[i] / scale).astype(np.float32)
        detections.append(FaceDetection(box, landmarks, float(scores[i])))
    return detections


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class FaceDetector:
    """An ONNX face detector; see the module docstring for the two backends."""

    def __init__(self) -> None:
        self._session: Any = None
        self._input_name = ""
        self._config: DetectorConfig | None = None
        self._chain: list[Transform] = []
        self._model_path: Path | None = None

    @property
    def is_initialized(self) -> bool:
        return self._session is not None

    @property
    def config(self) -> DetectorConfig:
        """The loaded config; raises before :meth:`initialize`."""
        if self._config is None:
            raise RuntimeError(NOT_INITIALIZED)
        return self._config

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    def initialize(
        self,
        runtime_opts: RuntimeOptions,
        model_bundle_filepath: str | Path,
        model_config_path: str | Path | None = None,
    ) -> None:
        """Load the model and its config; raises on any failure.

        *model_bundle_filepath* is the ``.onnx`` file or a directory holding
        one.  The config is ``<model>.json`` beside it, or *model_config_path*,
        or the packaged SCRFD default.
        """
        model = resolve_model_file(model_bundle_filepath)
        raw = load_model_config(model, model_config_path, DEFAULT_CONFIG_PATH)
        config = DetectorConfig.from_dict(raw)
        chain = build_transforms(config.transforms)
        session = runtime_opts.make_session(str(model))
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{model}: expected one model input, found {[i.name for i in inputs]}")
        self._session = session
        self._input_name = inputs[0].name
        self._config = config
        self._chain = chain
        self._model_path = model

    def detect(self, image_tensor: ImageTensor, confidence_threshold: float | None = None) -> list[FaceDetection]:
        """Faces in *image_tensor*, best first, in its own pixel coordinates.

        *confidence_threshold* overrides the config's ``score_threshold``.  An
        image smaller than ``min_image_size`` on either side yields no faces.
        """
        if self._session is None or self._config is None:
            raise RuntimeError(NOT_INITIALIZED)
        image = ensure_image_tensor(image_tensor)
        cfg = self._config
        threshold = cfg.score_threshold if confidence_threshold is None else float(confidence_threshold)
        img_h, img_w = image.shape[:2]
        if min(img_h, img_w) < cfg.min_image_size:
            return []
        sample = self._preprocess(image)
        assert sample.tensor is not None
        outputs = self._session.run(None, {self._input_name: sample.tensor})
        in_h, in_w = sample.tensor.shape[2], sample.tensor.shape[3]
        if cfg.backend == "retinaface":
            boxes, scores, kps = decode_retinaface(outputs, in_h, in_w, cfg, threshold)
        else:
            boxes, scores, kps = decode_scrfd(outputs, in_h, in_w, cfg, threshold)
        return postprocess(boxes, scores, kps, sample.scale, cfg, (img_h, img_w))

    def detect_buffer(self, buffer: bytes, confidence_threshold: float | None = None) -> list[FaceDetection]:
        """:meth:`detect` on an encoded image held in memory."""
        if self._session is None:
            raise RuntimeError(NOT_INITIALIZED)
        return self.detect(decode_image_buffer(buffer), confidence_threshold)

    def _preprocess(self, image: ImageTensor) -> Sample:
        """Cap the longer side at ``max_image_size``, then run the config's transforms."""
        assert self._config is not None
        shrunk, scale = resize_within(image, self._config.max_image_size)
        sample = apply_transforms(self._chain, Sample(image=shrunk, scale=scale))
        if sample.tensor is None:
            raise ValueError("detector transforms must end with ToTensor")
        return sample
