"""The two command lines: detection only, and the full detect + align + embed pipeline.

``scripts/detect_faces.py`` and ``scripts/process_faces.py`` are thin shims
over :func:`detect_faces_main` and :func:`process_faces_main`, so the logic is
importable and testable in-process.  Both share the input selection
(``--dir`` / ``--list`` / ``--img``), the runtime flags, and the failure
semantics: a file that cannot be read or processed is reported on stderr and
skipped, and the run carries on; only a model that fails to load, or no usable
input at all, is fatal.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from imagekit.face.face_util import save_detections_csv, save_embeddings_npz, save_face_records_csv
from imagekit.face.records import FaceDetection, FaceRecord
from imagekit.runtime import RuntimeOptions
from imagekit.util import collect_image_paths, read_image_tensor


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", metavar="DIR", help="process every image under this directory (recursive)")
    group.add_argument("--list", metavar="FILE", help="process the image paths listed one per line in FILE")
    group.add_argument("--img", metavar="PATH", help="process one image")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-gpu", action="store_true", default=None, help="run on the CUDA execution provider")
    parser.add_argument("--gpu-device-id", type=int, default=None, metavar="N", help="CUDA device (default 0)")
    parser.add_argument("--num-threads", type=int, default=None, metavar="N", help="CPU intra-op threads")
    parser.add_argument(
        "--confidence-threshold", type=float, default=None, metavar="T", help="override the model's score threshold"
    )


def _runtime_options(args: argparse.Namespace, base: RuntimeOptions | None = None) -> RuntimeOptions:
    """*base* (the config's runtime block, or the defaults) overridden by whichever flags were given."""
    base = base or RuntimeOptions()
    return RuntimeOptions(
        use_gpu=bool(args.use_gpu) if args.use_gpu is not None else base.use_gpu,
        gpu_device_id=args.gpu_device_id if args.gpu_device_id is not None else base.gpu_device_id,
        num_threads=args.num_threads if args.num_threads is not None else base.num_threads,
    )


def _inputs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """``(media_id, path)`` pairs; under ``--dir`` the id is the path relative to it."""
    paths = collect_image_paths(directory=args.dir, list_file=args.list, image=args.img)
    if args.dir:
        root = Path(args.dir)
        return [(str(p.relative_to(root)), p) for p in paths]
    return [(str(p), p) for p in paths]


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _each_image(inputs: Sequence[tuple[str, Path]]) -> Iterator[tuple[str, Any]]:
    """Yield ``(media_id, image_tensor)`` for every readable input; warn and skip the rest."""
    for media_id, path in inputs:
        try:
            yield media_id, read_image_tensor(path)
        except Exception as exc:  # per-file failures warn and continue
            _warn(f"{path}: {exc}")


# ---------------------------------------------------------------------------
# detect_faces
# ---------------------------------------------------------------------------


def _detect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect_faces.py",
        description="Detect faces (SCRFD / RetinaFace on ONNX Runtime) and write a detections CSV.",
    )
    parser.add_argument("detect_model", help="the detector .onnx file, or a directory holding it and its JSON config")
    parser.add_argument("output", help="the detections CSV to write")
    _add_input_args(parser)
    _add_runtime_args(parser)
    parser.add_argument(
        "--model-config", default=None, metavar="PATH", help="detector JSON config (default: beside the model)"
    )
    return parser


def detect_faces_main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``scripts/detect_faces.py``; returns the exit status."""
    from imagekit.face.detection.face_detector import FaceDetector  # noqa: PLC0415

    args = _detect_parser().parse_args(argv)
    detector = FaceDetector()
    detector.initialize(_runtime_options(args), args.detect_model, args.model_config)
    inputs = _inputs(args)
    rows: list[tuple[str, int, FaceDetection]] = []
    for media_id, image in _each_image(inputs):
        try:
            detections = detector.detect(image, args.confidence_threshold)
        except Exception as exc:
            _warn(f"{media_id}: detection failed: {exc}")
            continue
        rows.extend((media_id, index, det) for index, det in enumerate(detections))
    total = save_detections_csv(args.output, rows)
    print(f"Total Detects: {total}")
    return 0


# ---------------------------------------------------------------------------
# process_faces
# ---------------------------------------------------------------------------


def _process_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_faces.py",
        description="Detect, align and embed faces through a provider; write <prefix>.csv and <prefix>.npz.",
    )
    parser.add_argument("config", help="pipeline config JSON (provider, runtime, detector and embedder models)")
    parser.add_argument("output_prefix", help="writes <output_prefix>.csv and <output_prefix>.npz")
    _add_input_args(parser)
    parser.add_argument(
        "--provider",
        default=None,
        metavar="NAME",
        help="face provider (default: IMAGEKIT_FACE_PROVIDER, else the config's, else oss)",
    )
    _add_runtime_args(parser)
    return parser


def process_faces_main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``scripts/process_faces.py``; returns the exit status."""
    from imagekit.face.pipeline import load_pipeline, read_pipeline_config, runtime_options_from_config  # noqa: PLC0415

    args = _process_parser().parse_args(argv)
    config = read_pipeline_config(args.config)
    pipeline = load_pipeline(config, _runtime_options(args, runtime_options_from_config(config)), args.provider)
    inputs = _inputs(args)
    records: list[FaceRecord] = []
    for media_id, image in _each_image(inputs):
        try:
            found = pipeline.process(image, confidence_threshold=args.confidence_threshold, media_id=media_id)
        except Exception as exc:
            _warn(f"{media_id}: processing failed: {exc}")
            continue
        records.extend(found)
    prefix = Path(args.output_prefix)
    save_face_records_csv(prefix.with_suffix(".csv"), records)
    total = save_embeddings_npz(prefix.with_suffix(".npz"), records)
    print(f"Total Faces: {total}")
    return 0
