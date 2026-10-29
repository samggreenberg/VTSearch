#!/usr/bin/env python
"""Full-pipeline CLI: detect, align and embed faces through a provider.

Usage::

    python scripts/data/process_faces.py <config> <output_prefix> (--dir DIR | --list FILE | --img PATH)
        [--provider NAME] [--use-gpu] [--gpu-device-id N] [--num-threads N] [--confidence-threshold T]

``<config>`` is a pipeline config JSON (see ``imagekit/README.md``); the run
writes ``<output_prefix>.csv`` (one FaceRecord row per face, embedding
omitted) and ``<output_prefix>.npz`` (the embeddings, keyed by media id and
face index, with the compat tag).  Prints ``Total Faces: N``.  A file that
cannot be read or processed is reported on stderr and skipped.  The logic
lives in :mod:`imagekit.cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imagekit.cli import process_faces_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(process_faces_main())
