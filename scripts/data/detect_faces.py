#!/usr/bin/env python
"""Detection-only CLI: find faces and write a detections CSV.

Usage::

    python scripts/data/detect_faces.py <detect_model> <output> (--dir DIR | --list FILE | --img PATH)
        [--use-gpu] [--gpu-device-id N] [--num-threads N] [--confidence-threshold T] [--model-config PATH]

``<detect_model>`` is the detector ``.onnx`` (or a directory holding it and its
JSON config); ``<output>`` is the CSV to write.  Prints ``Total Detects: N``.
A file that cannot be read or detected on is reported on stderr and skipped.
The logic lives in :mod:`imagekit.cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imagekit.cli import detect_faces_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(detect_faces_main())
