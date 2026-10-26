"""Hand-built ONNX graphs with the shapes of the real face models, for weight-free tests.

A synthetic SCRFD / RetinaFace graph emits **constant** head outputs that
encode a chosen set of faces, so the decoders, NMS, the scale map-back and the
whole pipeline run through ``onnxruntime`` exactly as with real weights, while
the expected boxes are known to the pixel.  The synthetic embedder is a tiny
real network (global average pool -> linear) whose output depends on the
crop, so alignment and batching are exercised too.

Every graph consumes its input through a ``0 * mean(input)`` term, so the
session sees a real dependence and the shapes are checked at run time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT_NAME = "input.1"

#: Opset / IR versions every installed onnxruntime accepts.
_OPSET = 13
_IR_VERSION = 8


def _const(name: str, array: np.ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name + "_value"))


def _zero_from_input(nodes: list[onnx.NodeProto]) -> str:
    """Append nodes producing a scalar ``0 * mean(input)``; return its name."""
    nodes.append(helper.make_node("ReduceMean", [INPUT_NAME], ["__mean"], keepdims=0))
    nodes.append(_const("__zero_c", np.zeros((), dtype=np.float32)))
    nodes.append(helper.make_node("Mul", ["__mean", "__zero_c"], ["__zero"]))
    return "__zero"


def _save(nodes: list[onnx.NodeProto], inputs: list[Any], outputs: list[Any], path: Path, name: str) -> Path:
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = _IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _constant_outputs(
    nodes: list[onnx.NodeProto], arrays: dict[str, np.ndarray], zero: str, batch_dim: bool
) -> list[Any]:
    outputs = []
    for name, array in arrays.items():
        value = array[None, ...] if batch_dim else array
        nodes.append(_const(name + "_c", value.astype(np.float32)))
        nodes.append(helper.make_node("Add", [name + "_c", zero], [name]))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)))
    return outputs


# ---------------------------------------------------------------------------
# SCRFD
# ---------------------------------------------------------------------------


def scrfd_face(
    level: int,
    x: int,
    y: int,
    score: float,
    box: Sequence[float],
    kps: Sequence[Sequence[float]] | None = None,
    anchor: int = 0,
) -> dict[str, Any]:
    """One face for :func:`build_scrfd_model`: at grid cell ``(x, y)`` of stride level *level*.

    *box* is ``(left, top, right, bottom)`` distances from the anchor centre in
    input pixels; *kps* are five ``(dx, dy)`` offsets from it.
    """
    return {"level": level, "x": x, "y": y, "score": score, "box": tuple(box), "kps": kps, "anchor": anchor}


def scrfd_expected(face: dict[str, Any], feat_stride: Sequence[int] = (8, 16, 32)) -> dict[str, Any]:
    """The box / landmarks :func:`scrfd_face` encodes, in input pixels."""
    stride = feat_stride[face["level"]]
    cx, cy = face["x"] * stride, face["y"] * stride
    left, top, right, bottom = face["box"]
    out: dict[str, Any] = {"bbox": (cx - left, cy - top, cx + right, cy + bottom), "score": face["score"]}
    if face.get("kps") is not None:
        out["landmarks"] = np.asarray([(cx + dx, cy + dy) for dx, dy in face["kps"]], dtype=np.float32)
    return out


def build_scrfd_model(
    path: Path,
    faces: Sequence[dict[str, Any]] = (),
    *,
    input_hw: tuple[int, int] = (640, 640),
    feat_stride: Sequence[int] = (8, 16, 32),
    num_anchors: int = 2,
    use_kps: bool = True,
    batch_dim: bool = False,
) -> Path:
    """Write an SCRFD-shaped ONNX model whose heads encode *faces*; returns *path*."""
    height, width = input_hw
    arrays: dict[str, np.ndarray] = {}
    for li, stride in enumerate(feat_stride):
        n = (height // stride) * (width // stride) * num_anchors
        arrays[f"score{li}"] = np.zeros((n, 1), np.float32)
        arrays[f"bbox{li}"] = np.zeros((n, 4), np.float32)
        arrays[f"kps{li}"] = np.zeros((n, 10), np.float32)
    for face in faces:
        li = face["level"]
        stride = feat_stride[li]
        idx = (face["y"] * (width // stride) + face["x"]) * num_anchors + face.get("anchor", 0)
        arrays[f"score{li}"][idx, 0] = face["score"]
        arrays[f"bbox{li}"][idx] = np.asarray(face["box"], np.float32) / stride
        if face.get("kps") is not None:
            arrays[f"kps{li}"][idx] = (np.asarray(face["kps"], np.float32) / stride).reshape(-1)
    order = [f"score{i}" for i in range(len(feat_stride))] + [f"bbox{i}" for i in range(len(feat_stride))]
    if use_kps:
        order += [f"kps{i}" for i in range(len(feat_stride))]
    nodes: list[onnx.NodeProto] = []
    zero = _zero_from_input(nodes)
    outputs = _constant_outputs(nodes, {k: arrays[k] for k in order}, zero, batch_dim)
    inputs = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 3, height, width])]
    return _save(nodes, inputs, outputs, path, "scrfd_synthetic")


def write_scrfd_config(path: Path, **overrides: Any) -> Path:
    """Write the packaged SCRFD defaults (plus *overrides*) as a detector config beside a model."""
    from imagekit.face.detection.face_detector import DEFAULT_CONFIG_PATH  # noqa: PLC0415

    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"].update(overrides)
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# RetinaFace
# ---------------------------------------------------------------------------


def retinaface_prior_count(input_hw: tuple[int, int], steps: Sequence[int] = (8, 16, 32), anchors: int = 2) -> int:
    height, width = input_hw
    return sum(int(np.ceil(height / s)) * int(np.ceil(width / s)) * anchors for s in steps)


def build_retinaface_model(
    path: Path,
    input_hw: tuple[int, int],
    hits: Sequence[tuple[int, float]] = (),
    *,
    with_landmarks: bool = True,
) -> Path:
    """Write a RetinaFace-shaped model (``loc`` / ``conf`` / ``landms`` over the priors of *input_hw*).

    Each hit is ``(prior_index, score)``; its ``loc`` and ``landms`` are zero,
    so the decoded box is the prior itself and every landmark its centre.
    """
    height, width = input_hw
    n = retinaface_prior_count(input_hw)
    loc = np.zeros((n, 4), np.float32)
    conf = np.zeros((n, 2), np.float32)
    conf[:, 0] = 1.0
    for index, score in hits:
        conf[index] = (1.0 - score, score)
    arrays = {"loc": loc, "conf": conf}
    if with_landmarks:
        arrays["landms"] = np.zeros((n, 10), np.float32)
    nodes: list[onnx.NodeProto] = []
    zero = _zero_from_input(nodes)
    outputs = _constant_outputs(nodes, arrays, zero, batch_dim=True)
    inputs = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 3, height, width])]
    return _save(nodes, inputs, outputs, path, "retinaface_synthetic")


def write_retinaface_config(path: Path, **overrides: Any) -> Path:
    from imagekit.face.detection.face_detector import RETINAFACE_CONFIG_PATH  # noqa: PLC0415

    raw = json.loads(RETINAFACE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"].update(overrides)
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


def embedder_weight(dim: int = 512, seed: int = 0) -> np.ndarray:
    """The fixed ``(3, dim)`` linear weight of the synthetic embedder."""
    return np.random.default_rng(seed).standard_normal((3, dim)).astype(np.float32)


def build_embedder_model(path: Path, dim: int = 512, *, fixed_batch: int | None = None, side: int = 112) -> Path:
    """Write ``mean over pixels -> linear(3 -> dim)``; input ``(N, 3, side, side)`` (dynamic N unless *fixed_batch*)."""
    batch: Any = fixed_batch if fixed_batch is not None else "N"
    nodes: list[onnx.NodeProto] = [
        helper.make_node("GlobalAveragePool", [INPUT_NAME], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        _const("weight", embedder_weight(dim)),
        helper.make_node("MatMul", ["flat", "weight"], ["emb"]),
    ]
    inputs = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [batch, 3, side, side])]
    outputs = [helper.make_tensor_value_info("emb", TensorProto.FLOAT, [batch, dim])]
    return _save(nodes, inputs, outputs, path, "embedder_synthetic")


def expected_embedding(crop_rgb: np.ndarray, dim: int = 512) -> np.ndarray:
    """What :func:`build_embedder_model` returns for an aligned crop normalised ``mean/std 0.5``."""
    normalised = (crop_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return normalised.reshape(-1, 3).mean(axis=0) @ embedder_weight(dim)


def write_embedder_config(path: Path, compatibility: str = "oss-synthetic:1", **overrides: Any) -> Path:
    raw: dict[str, Any] = {
        "model": {
            "name": "synthetic",
            "compatibility": compatibility,
            "embedding_dim": 512,
            "transforms": [
                {"name": "Align", "args": {"height": 112, "width": 112, "fallback": "crop"}},
                {"name": "Normalize", "args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
                {"name": "ToTensor"},
            ],
        }
    }
    raw["model"].update(overrides)
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# A whole synthetic model directory
# ---------------------------------------------------------------------------


def build_model_dir(root: Path, faces: Sequence[dict[str, Any]] = (), **detector_overrides: Any) -> dict[str, Path]:
    """Detector + embedder models and configs under *root*; returns their paths."""
    root.mkdir(parents=True, exist_ok=True)
    det = build_scrfd_model(root / "det.onnx", faces)
    det_cfg = write_scrfd_config(root / "det.json", **detector_overrides)
    emb = build_embedder_model(root / "emb.onnx")
    emb_cfg = write_embedder_config(root / "emb.json")
    return {"detector": det, "detector_config": det_cfg, "embedder": emb, "embedder_config": emb_cfg}


def pipeline_config(paths: dict[str, Path], **extra: Any) -> dict[str, Any]:
    """A pipeline config dict over :func:`build_model_dir`'s paths."""
    config: dict[str, Any] = {
        "detector": {"model_bundle": str(paths["detector"])},
        "embedder": {"model": str(paths["embedder"]), "config": str(paths["embedder_config"])},
    }
    config.update(extra)
    return config
