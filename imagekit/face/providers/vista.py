"""``VistaFacePipeline``: the proprietary ``vistaimage`` backend behind the provider seam.

``vistaimage`` (with ``vistacommon``) exposes nearly the same classes as this
package: ``vistaimage.face.FaceDetector`` with ``initialize(runtime_opts,
model_bundle)`` / ``detect(image_tensor, confidence_threshold)``,
``vistaimage.face.FaceEmbeddingExtractor`` with ``initialize(runtime_opts,
model_filepath, model_config_path=None)`` / ``extract(detection, image_tensor)``
/ ``get_compatibility()``, and ``vistacommon.RuntimeOptions(use_gpu,
gpu_device_id, num_threads)``.  Two things differ and are mapped here: the
embedding it returns is an object with a ``.values`` sequence, and its
compatibility is an integer, which becomes ``"vistaface:<n>"``.

Both packages are imported **lazily inside** :meth:`VistaFacePipeline.initialize`,
so this module imports on every machine and the missing package is named only
when the provider is actually asked for.  Every return value is converted to
:class:`~imagekit.face.records.FaceRecord` at this boundary; no ``vistaimage``
type leaks out.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np

from imagekit.face.pipeline import DETECTOR_KEY, EMBEDDER_KEY
from imagekit.face.records import FaceDetection, FaceRecord, as_bbox, as_landmarks, vista_compat
from imagekit.runtime import RuntimeOptions
from imagekit.util import decode_image_buffer, ensure_image_tensor

#: The proprietary packages the provider needs, in the order they are imported.
VISTA_PACKAGES: tuple[str, ...] = ("vistaimage", "vistacommon")

NOT_INITIALIZED = "Must initialize VistaFacePipeline first!"


class VistaProviderUnavailable(ImportError):
    """The ``vista`` provider was asked for on a machine without its package."""


def import_vista() -> tuple[ModuleType, ModuleType]:
    """Import ``(vistaimage, vistacommon)``, raising :class:`VistaProviderUnavailable` naming the first one missing."""
    modules: list[ModuleType] = []
    for name in VISTA_PACKAGES:
        try:
            modules.append(importlib.import_module(name))
        except ImportError as exc:
            raise VistaProviderUnavailable(
                f"the 'vista' face provider needs the proprietary package {name!r}, which is not installed "
                f"on this machine; install it, or select another provider (IMAGEKIT_FACE_PROVIDER=oss)"
            ) from exc
    return modules[0], modules[1]


def _attr(obj: Any, *names: str) -> Any:
    """The first of *names* present on *obj* as an attribute or mapping key, else ``None``."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def to_detection(det: Any) -> FaceDetection:
    """Convert one ``vistaimage`` detection (duck-typed) into a :class:`FaceDetection`."""
    box = _attr(det, "bbox", "box")
    if box is None:
        raise ValueError(f"vista detection {det!r} has no bbox")
    confidence = _attr(det, "confidence", "score", "conf")
    landmarks = _attr(det, "landmarks", "kps", "keypoints")
    return FaceDetection(as_bbox(box), as_landmarks(landmarks), float(confidence if confidence is not None else 1.0))


def to_vector(embedding: Any) -> np.ndarray:
    """The ``float32`` vector behind a ``vistaimage`` embedding object (its ``.values`` sequence)."""
    values = _attr(embedding, "values")
    if values is None:
        values = embedding
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32).reshape(-1)


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = config.get(key) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"pipeline config: {key!r} must be a JSON object")
    return section


class VistaFacePipeline:
    """Thin wrapper: ``vistaimage``'s detector + extractor, converted to :class:`FaceRecord` s.

    Reads the same config keys as the ``oss`` provider
    (``detector.model_bundle``, ``embedder.model``, ``embedder.config``), so one
    config file shape serves both.
    """

    def __init__(self) -> None:
        self._detector: Any = None
        self._extractor: Any = None
        self._compat = ""

    def initialize(self, opts: RuntimeOptions, config: Mapping[str, Any]) -> None:
        vistaimage, vistacommon = import_vista()
        detector_cfg = _section(config, DETECTOR_KEY)
        embedder_cfg = _section(config, EMBEDDER_KEY)
        bundle = detector_cfg.get("model_bundle") or detector_cfg.get("model")
        model = embedder_cfg.get("model")
        if not bundle or not model:
            raise ValueError(
                "vista provider: config['detector']['model_bundle'] and config['embedder']['model'] are required"
            )
        runtime = vistacommon.RuntimeOptions(bool(opts.use_gpu), int(opts.gpu_device_id), int(opts.num_threads))
        detector = vistaimage.face.FaceDetector()
        detector.initialize(runtime, str(bundle))
        extractor = vistaimage.face.FaceEmbeddingExtractor()
        model_config = embedder_cfg.get("config")
        if model_config:
            extractor.initialize(runtime, str(model), str(model_config))
        else:
            extractor.initialize(runtime, str(model))
        self._compat = vista_compat(extractor.get_compatibility())
        self._detector = detector
        self._extractor = extractor

    def compatibility(self) -> str:
        if self._extractor is None:
            raise RuntimeError(NOT_INITIALIZED)
        return self._compat

    def process(self, image: Any, *, confidence_threshold: float | None = None, media_id: str = "") -> list[FaceRecord]:
        if self._detector is None or self._extractor is None:
            raise RuntimeError(NOT_INITIALIZED)
        tensor = ensure_image_tensor(image)
        raw_detections = self._detector.detect(tensor, confidence_threshold) or []
        records: list[FaceRecord] = []
        for index, raw in enumerate(raw_detections):
            vector = to_vector(self._extractor.extract(raw, tensor))
            records.append(FaceRecord.from_detection(to_detection(raw), vector, self._compat, media_id, index))
        records.sort(key=lambda r: -r.confidence)
        return records

    def process_buffer(
        self, buffer: bytes, *, confidence_threshold: float | None = None, media_id: str = ""
    ) -> list[FaceRecord]:
        if self._detector is None:
            raise RuntimeError(NOT_INITIALIZED)
        return self.process(decode_image_buffer(buffer), confidence_threshold=confidence_threshold, media_id=media_id)
