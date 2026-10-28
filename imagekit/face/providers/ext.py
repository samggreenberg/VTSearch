"""``ExtFacePipeline``: an external face backend behind the provider seam.

The backend is one module, named by :func:`backend_module_name`
(``IMAGEKIT_EXT_BACKEND``, else the pipeline config's ``ext_backend`` key, else
``ext_backend``), that exposes nearly the same classes as this package:
``backend.face.FaceDetector`` with ``initialize(runtime_opts, model_bundle)`` /
``detect(image_tensor, confidence_threshold)``,
``backend.face.FaceEmbeddingExtractor`` with ``initialize(runtime_opts,
model_filepath, model_config_path=None)`` / ``extract(detection, image_tensor)``
/ ``get_compatibility()``, and ``backend.RuntimeOptions(use_gpu,
gpu_device_id, num_threads)``.  One thing differs and is mapped here: the
embedding it returns is an object with a ``.values`` sequence.  Its
compatibility is already the embedding-space tag, and is passed through
verbatim.

The backend is imported **lazily inside** :meth:`ExtFacePipeline.initialize`,
so this module imports on every machine and the missing module is named only
when the provider is actually asked for.  Every return value is converted to
:class:`~imagekit.face.records.FaceRecord` at this boundary; no backend type
leaks out.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np

from imagekit.face.pipeline import DETECTOR_KEY, EMBEDDER_KEY
from imagekit.face.records import FaceDetection, FaceRecord, as_bbox, as_landmarks
from imagekit.runtime import RuntimeOptions
from imagekit.util import decode_image_buffer, ensure_image_tensor

#: Environment variable naming the backend module; it wins over the config's.
BACKEND_ENV_VAR = "IMAGEKIT_EXT_BACKEND"

#: Pipeline-config key naming the backend module when the environment names none.
BACKEND_KEY = "ext_backend"

#: The backend module imported when neither the environment nor the config names one.
DEFAULT_BACKEND = "ext_backend"

NOT_INITIALIZED = "Must initialize ExtFacePipeline first!"


class ExtProviderUnavailable(ImportError):
    """The ``ext`` provider was asked for on a machine without its backend module."""


def backend_module_name(config: Mapping[str, Any] | None = None) -> str:
    """The module the ``ext`` provider imports: the environment's, else *config*'s, else ``ext_backend``."""
    named = os.environ.get(BACKEND_ENV_VAR, "").strip()
    if not named and config is not None:
        named = str(config.get(BACKEND_KEY) or "").strip()
    return named or DEFAULT_BACKEND


def import_backend(name: str | None = None) -> ModuleType:
    """Import the backend module *name* (default: :func:`backend_module_name`).

    Raises :class:`ExtProviderUnavailable` naming the module when it cannot be
    imported.
    """
    module_name = name or backend_module_name()
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ExtProviderUnavailable(
            f"the 'ext' face provider needs its backend module {module_name!r}, which cannot be imported "
            f"on this machine ({exc}); install it, point {BACKEND_ENV_VAR} at the module that provides it, "
            f"or select another provider (IMAGEKIT_FACE_PROVIDER=oss)"
        ) from exc


def _attr(obj: Any, *names: str) -> Any:
    """The first of *names* present on *obj* as an attribute or mapping key, else ``None``."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def to_detection(det: Any) -> FaceDetection:
    """Convert one backend detection (duck-typed) into a :class:`FaceDetection`."""
    box = _attr(det, "bbox", "box")
    if box is None:
        raise ValueError(f"ext detection {det!r} has no bbox")
    confidence = _attr(det, "confidence", "score", "conf")
    landmarks = _attr(det, "landmarks", "kps", "keypoints")
    return FaceDetection(as_bbox(box), as_landmarks(landmarks), float(confidence if confidence is not None else 1.0))


def to_vector(embedding: Any) -> np.ndarray:
    """The ``float32`` vector behind a backend embedding object (its ``.values`` sequence)."""
    values = _attr(embedding, "values")
    if values is None:
        values = embedding
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32).reshape(-1)


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = config.get(key) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"pipeline config: {key!r} must be a JSON object")
    return section


class ExtFacePipeline:
    """Thin wrapper: the backend's detector + extractor, converted to :class:`FaceRecord` s.

    Reads the same config keys as the ``oss`` provider
    (``detector.model_bundle``, ``embedder.model``, ``embedder.config``), so one
    config file shape serves both, plus the optional ``ext_backend``.
    """

    def __init__(self) -> None:
        self._detector: Any = None
        self._extractor: Any = None
        self._compat = ""

    def initialize(self, opts: RuntimeOptions, config: Mapping[str, Any]) -> None:
        module_name = backend_module_name(config)
        backend = import_backend(module_name)
        detector_cfg = _section(config, DETECTOR_KEY)
        embedder_cfg = _section(config, EMBEDDER_KEY)
        bundle = detector_cfg.get("model_bundle") or detector_cfg.get("model")
        model = embedder_cfg.get("model")
        if not bundle or not model:
            raise ValueError(
                "ext provider: config['detector']['model_bundle'] and config['embedder']['model'] are required"
            )
        runtime = backend.RuntimeOptions(bool(opts.use_gpu), int(opts.gpu_device_id), int(opts.num_threads))
        detector = backend.face.FaceDetector()
        detector.initialize(runtime, str(bundle))
        extractor = backend.face.FaceEmbeddingExtractor()
        model_config = embedder_cfg.get("config")
        if model_config:
            extractor.initialize(runtime, str(model), str(model_config))
        else:
            extractor.initialize(runtime, str(model))
        compat = extractor.get_compatibility()
        if not isinstance(compat, str) or not compat:
            raise ValueError(
                f"ext provider: backend {module_name!r} reported compatibility {compat!r}; "
                "get_compatibility() must return the embedding-space tag as a non-empty string"
            )
        self._compat = compat
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
