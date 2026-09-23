"""The calibration harness, with the patch-row dtype pinned per arm (#3159).

    VTS_PATCH_ROW_DTYPE=float32 python run_cells_3159.py --index 7
    VTS_PATCH_ROW_DTYPE=float16 python run_cells_3159.py --prepare

A thin wrapper over ``calibration/run_cells.py`` and ``prepare_data.py``: it
sets :data:`vtscore.embedding.matrix.PATCH_ROW_DTYPE` before the harness
imports anything, then **asserts the premise** on every cell pickle the harness
opens.  A float32 arm that silently loaded float16 grids - a stale symlink, the
wrong datadir - would score float16 against float16 and report a perfect null,
which is the #2877 shape: a knob that did nothing reads as a treatment with no
effect.  So the wrapper refuses to run unless the grids on disk are the dtype
the arm is named for, and it refuses to *write* any cell pickle at all (the
float16 arm's cells are symlinks into the shared pile).

Each cell also leaves ``dtype_<index>.json`` beside its CSV, recording the dtype
the scoring stack actually held, read off the harness's own flattened matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALIB = HERE.parent / "calibration"
sys.path.insert(0, str(CALIB))

import common  # noqa: E402

common.setup_env()

import numpy as np  # noqa: E402

WANT = os.environ.get("VTS_PATCH_ROW_DTYPE", "")
if WANT not in ("float16", "float32"):
    raise SystemExit(f"VTS_PATCH_ROW_DTYPE must be float16 or float32, got {WANT!r}")

from vtscore.embedding import matrix  # noqa: E402

matrix.PATCH_ROW_DTYPE = np.dtype(WANT).type

import _cells_io  # noqa: E402

_real_load = _cells_io.load_medias
_seen: dict[str, str] = {}


def _checked_load(path, *a, **kw):
    medias = _real_load(path, *a, **kw)
    dtypes = {str(m["patch_grid"].dtype) for m in medias.values() if m.get("patch_grid") is not None}
    if dtypes and dtypes != {WANT}:
        raise SystemExit(f"{path}: grids on disk are {sorted(dtypes)} but this arm is {WANT}; refusing to run")
    _seen[Path(path).name] = ",".join(sorted(dtypes)) or "no patch grid"
    common.log(f"[3159] {Path(path).name}: grid dtype {_seen[Path(path).name]} (arm {WANT})")
    return medias


def _no_write(*_a, **_kw):
    raise SystemExit("run_cells_3159 never writes a cell pickle; the fp16 arm's cells are the shared pile's")


_cells_io.load_medias = _checked_load
_cells_io.dump_medias = _no_write


def _record_stack_dtype(index: int) -> None:
    """What the MaxPatch stack actually held, from the harness's own class."""
    from vtscore.eval.patch_styles import MaxPatchStyle  # noqa: PLC0415

    fake = {
        "id": 0,
        "media_type": "image",
        "embedder": "dinov3_patch",
        "embeddings": {"dinov3_patch": np.ones(4, dtype=np.float32)},
        "patch_grid": np.ones((1, 1, 4), dtype=np.float32),
    }
    held = str(MaxPatchStyle()._flattened({0: fake})[1].dtype)
    if held != WANT:
        raise SystemExit(f"MaxPatch stack holds {held}, arm is {WANT}: PATCH_ROW_DTYPE did not reach the harness")
    out = common.RESULTS / "cells" / f"dtype_{index}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm_dtype": WANT, "stack_dtype": held, "pickles": _seen}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="run prepare_data.py instead of a cell")
    ap.add_argument("--index", type=int, default=None)
    args, rest = ap.parse_known_args(argv)
    os.chdir(CALIB)
    if args.prepare:
        import prepare_data  # noqa: PLC0415

        return prepare_data.main(rest)

    import run_cells  # noqa: PLC0415

    index = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    rc = run_cells.main(["--index", str(index), *rest])
    _record_stack_dtype(index)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
