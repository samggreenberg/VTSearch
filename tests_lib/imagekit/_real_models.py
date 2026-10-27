"""The real weights the opt-in tests run on, from ``IMAGEKIT_TEST_MODEL_DIR``.

A directory holding a pipeline config, ``face.json`` (a deployment's models directory, as the GRID's default
is since #5341), names its own detector, embedder and embedder config, so the tests run on whatever pack it
describes. A directory without one holds InsightFace ``buffalo_l``'s files under their own names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from imagekit.face.extraction import ARCFACE_CONFIG_PATH

MODEL_DIR = os.environ.get("IMAGEKIT_TEST_MODEL_DIR")
FACE_IMAGE = os.environ.get("IMAGEKIT_TEST_FACE_IMAGE")


def _pipeline_config() -> dict | None:
    assert MODEL_DIR is not None
    path = Path(MODEL_DIR) / "face.json"
    return json.loads(path.read_text()) if path.is_file() else None


def detector_path() -> Path:
    """The detector's ONNX file."""
    assert MODEL_DIR is not None
    config = _pipeline_config()
    return Path(MODEL_DIR) / (config["detector"]["model_bundle"] if config else "det_10g.onnx")


def embedder_paths() -> tuple[Path, Path]:
    """The embedder's ONNX file and the extractor config naming its embedding space."""
    assert MODEL_DIR is not None
    config = _pipeline_config()
    if config is None:
        return Path(MODEL_DIR) / "w600k_r50.onnx", ARCFACE_CONFIG_PATH
    return Path(MODEL_DIR) / config["embedder"]["model"], Path(MODEL_DIR) / config["embedder"]["config"]


def expected_compat() -> str:
    """The compat tag the embedder's config promises."""
    return json.loads(embedder_paths()[1].read_text())["model"]["compatibility"]
