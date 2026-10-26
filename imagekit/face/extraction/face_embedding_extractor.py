"""``FaceEmbeddingExtractor``: open-weights face embeddings (MagFace / ArcFace) on ONNX Runtime.

The extractor takes a detection (for its landmarks and box) **and the full
image**, aligns the face to a 112 x 112 RGB crop (``Align``, see
:mod:`imagekit.face.face_align`), normalises it the way the config says
(``mean 0.5 / std 0.5`` for MagFace and ArcFace), and runs the model.  What
comes back is the model's raw 512-d ``float32`` vector, **not** L2-normalised:
MagFace puts face quality in the norm, and a record keeps both the vector and
that norm (:class:`~imagekit.face.records.FaceRecord`).

**Compatibility is enforced by config, not by convention.**  Every extractor
model needs a JSON config beside it (``<model>.json``, or an explicit
``model_config_path``) naming its embedding space, e.g.
``"compatibility": "oss-magface-ir100:1"``; there is no packaged default,
because a default would let two different weights claim the same space.  The
package ships reference configs for the two open models next to this module
(:data:`MAGFACE_CONFIG_PATH`, :data:`ARCFACE_CONFIG_PATH`) to copy beside the
weights.

Initialise-before-use and raise-on-failure exactly as the detector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from imagekit.face import face_align  # noqa: F401 - registers the Align transform
from imagekit.face.records import EMBEDDING_DIM, FaceDetection, oss_compat
from imagekit.runtime import RuntimeOptions
from imagekit.util import (
    ImageTensor,
    Sample,
    Transform,
    apply_transforms,
    build_transforms,
    decode_image_buffer,
    ensure_image_tensor,
    load_model_config,
    resolve_model_file,
)

#: Reference config for the official MagFace iResNet-100 release.
MAGFACE_CONFIG_PATH = Path(__file__).with_name("magface_ir100.json")

#: Reference config for InsightFace's ArcFace ResNet-50 (``buffalo_l`` / ``w600k_r50``).
ARCFACE_CONFIG_PATH = Path(__file__).with_name("arcface_r50.json")

NOT_INITIALIZED = "Must initialize FaceEmbeddingExtractor first!"

_DEFAULT_TRANSFORMS: tuple[Mapping[str, Any], ...] = (
    {"name": "Align", "args": {"height": 112, "width": 112, "fallback": "crop"}},
    {"name": "Normalize", "args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
    {"name": "ToTensor"},
)


@dataclass(frozen=True)
class ExtractorConfig:
    """The ``model`` block of an extractor's JSON config."""

    name: str
    compatibility: str
    embedding_dim: int = EMBEDDING_DIM
    transforms: tuple[Mapping[str, Any], ...] = _DEFAULT_TRANSFORMS

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], source: str = "config") -> "ExtractorConfig":
        """Build from a loaded config; ``compatibility`` or ``name`` + ``version`` is required."""
        model = raw.get("model", raw)
        if not isinstance(model, Mapping):
            raise ValueError(f"{source}: 'model' must be a JSON object")
        name = str(model.get("name", "")).strip()
        compat = str(model.get("compatibility", "")).strip()
        if not compat:
            if not name or "version" not in model:
                raise ValueError(
                    f"{source}: an extractor config must name its embedding space: "
                    'give model.compatibility (e.g. "oss-magface-ir100:1"), or model.name plus model.version'
                )
            compat = oss_compat(name, model["version"])
        return cls(
            name=name or compat,
            compatibility=compat,
            embedding_dim=int(model.get("embedding_dim", EMBEDDING_DIM)),
            transforms=tuple(model.get("transforms", _DEFAULT_TRANSFORMS)),
        )


class FaceEmbeddingExtractor:
    """An ONNX face embedder; see the module docstring for the contract."""

    def __init__(self) -> None:
        self._session: Any = None
        self._input_name = ""
        self._fixed_batch: int | None = None
        self._config: ExtractorConfig | None = None
        self._chain: list[Transform] = []
        self._model_path: Path | None = None

    @property
    def is_initialized(self) -> bool:
        return self._session is not None

    @property
    def config(self) -> ExtractorConfig:
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
        model_filepath: str | Path,
        model_config_path: str | Path | None = None,
    ) -> None:
        """Load the model and the config naming its embedding space; raises on any failure."""
        model = resolve_model_file(model_filepath)
        raw = load_model_config(model, model_config_path)
        config = ExtractorConfig.from_dict(raw, source=str(model_config_path or model.with_suffix(".json")))
        chain = build_transforms(config.transforms)
        session = runtime_opts.make_session(str(model))
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{model}: expected one model input, found {[i.name for i in inputs]}")
        batch = inputs[0].shape[0] if inputs[0].shape else None
        self._fixed_batch = int(batch) if isinstance(batch, int) else None
        self._session = session
        self._input_name = inputs[0].name
        self._config = config
        self._chain = chain
        self._model_path = model

    def get_compatibility(self) -> str:
        """The embedding-space tag of the loaded weights, straight from their config."""
        return self.config.compatibility

    def extract(self, detection: FaceDetection, image_tensor: ImageTensor) -> np.ndarray:
        """The raw ``(512,)`` ``float32`` embedding of *detection*'s face in the **full** *image_tensor*."""
        return self.extract_many([detection], image_tensor)[0]

    def extract_buffer(self, detection: FaceDetection, buffer: bytes) -> np.ndarray:
        """:meth:`extract` on an encoded image held in memory."""
        if self._session is None:
            raise RuntimeError(NOT_INITIALIZED)
        return self.extract(detection, decode_image_buffer(buffer))

    def extract_many(self, detections: Sequence[FaceDetection], image_tensor: ImageTensor) -> list[np.ndarray]:
        """One raw embedding per detection, batched through the model where its input allows."""
        if self._session is None or self._config is None:
            raise RuntimeError(NOT_INITIALIZED)
        if not detections:
            return []
        image = ensure_image_tensor(image_tensor)
        crops = np.concatenate([self._crop_tensor(det, image) for det in detections], axis=0)
        batch = self._fixed_batch or len(crops)
        outputs: list[np.ndarray] = []
        for start in range(0, len(crops), batch):
            chunk = np.ascontiguousarray(crops[start : start + batch])
            (out,) = self._session.run(None, {self._input_name: chunk})
            outputs.append(np.asarray(out, dtype=np.float32).reshape(len(chunk), -1))
        vectors = np.concatenate(outputs, axis=0)
        if vectors.shape[1] != self._config.embedding_dim:
            raise ValueError(
                f"{self._model_path}: model produced {vectors.shape[1]}-d vectors; config says {self._config.embedding_dim}"
            )
        return [vectors[i].copy() for i in range(len(detections))]

    def _crop_tensor(self, detection: FaceDetection, image: ImageTensor) -> np.ndarray:
        """``(1, 3, H, W)`` model input for one face: the config's transforms on the full image."""
        extra = {"landmarks": detection.landmarks, "bbox": detection.bbox}
        sample = apply_transforms(self._chain, Sample(image=image, extra=extra))
        if sample.tensor is None:
            raise ValueError("extractor transforms must end with ToTensor")
        return sample.tensor
