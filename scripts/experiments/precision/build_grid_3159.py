"""Build float32-grid ``dinov3_patch`` cells, then check they ARE the pile's cells (#3159).

Region voting reads a patch grid stored float16
(:data:`vtscore.embedding.matrix.PATCH_ROW_DTYPE`).  This builds the same cells
with that one name flipped to float32, in this study's own pile root, so the
float16 cost can be measured against the published cells rather than against a
second rebuild::

    python build_grid_3159.py build --datasets visual_genome_m,coco_val,caltech101_m
    python build_grid_3159.py verify --datasets visual_genome_m,coco_val,caltech101_m

**Why ``verify`` gates everything after it.**  The comparison is only a
measurement of the *cast* if the float32 forward is the forward the pile ran.
If the rebuild differed in anything else - media set, order, preprocessing, the
card's kernels - then "fp32 vs pile" would mix that difference into the cast's.
So ``verify`` asserts, per cell:

1. identical media ids, categories and boxes;
2. identical image-level (CLS) vectors.  The rebuild reproduces them only to
   ~3e-8, so ``adopt-cls`` copies the pile's in first (and refuses beyond
   1e-6); the arms then differ in the grid cast and nothing else;
3. **the float32 grid, cast to float16, equals the pile's grid bit for bit.**
   That is the whole premise in one check: the pile cell is exactly this
   study's float32 cell passed through the storage cast, and nothing else.

Nothing here writes to the shared pile.  Cells land under
``$VTS_GRID_STUDY/piles/fp32/datadir/embeddings``; the shared pile's models
dir is read so nothing re-downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pile"))
sys.path.insert(0, str(HERE.parent / "calibration"))

USER = os.environ.get("USER", "sgreenberg")
STUDY = Path(os.environ.get("VTS_GRID_STUDY", f"/expscratch/{USER}/patchgrid-3159"))
SHARED_PILE = Path(os.environ.get("VTS_PILE", f"/expscratch/{USER}/vts-cache"))
SHARED_EMBEDDINGS = SHARED_PILE / "datadir" / "embeddings"
SHARED_MODELS = SHARED_PILE / "models"
EMBEDDER = "dinov3_patch"
DATASETS = ["visual_genome_m", "coco_val", "caltech101_m"]


def log(msg: str) -> None:
    print(f"[grid-3159] {msg}", flush=True)


def arm_datadir(arm: str) -> Path:
    return STUDY / "piles" / arm / "datadir"


def _point_pile_config_at(datadir: Path):
    """Redirect ``pile_config`` to this study's root before anything builds.

    Same move ``build_arm.py`` makes for #3143: the builder writes cells and
    provenance to ``pc.EMBEDDINGS``, so this is what keeps the shared pile
    untouched.
    """
    import pile_config as pc  # noqa: PLC0415

    pc.PILE = datadir.parent
    pc.DATADIR = datadir
    pc.EMBEDDINGS = datadir / "embeddings"
    pc.MODELS = SHARED_MODELS
    pc.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    return pc


def _link_sources(pc, datasets: list[str]) -> None:
    """Link each demo dataset's staged source in, read-only.

    A *missing* source dir reads to the demo loader as "not downloaded yet" and
    it silently substitutes a truncated re-download (the pile once shipped a
    1662-of-4193 cell that way), so link first and let ``require_demo_source``
    confirm.
    """
    for ds in datasets:
        name = pc.DATASETS[ds].get("source_dir")
        if not name:
            continue
        link = pc.DATADIR / name
        src = (SHARED_PILE / "datadir" / name).resolve()
        if not link.exists():
            if not src.exists():
                raise SystemExit(f"missing demo source {src}")
            link.symlink_to(src)
            log(f"  linked {link} -> {src}")


def cmd_build(datasets: list[str], force: bool) -> int:
    datadir = arm_datadir("fp32")
    os.environ["VTS_PILE"] = str(datadir.parent)
    os.environ["VTSEARCH_DATA_DIR"] = str(datadir)
    os.environ["VTSEARCH_MODELS_DIR"] = str(SHARED_MODELS)
    os.environ["HF_HOME"] = str(SHARED_MODELS)
    # The pile was built fp32; only the STORAGE cast is under test.
    os.environ["VTSEARCH_EMBED_PRECISION"] = "fp32"

    pc = _point_pile_config_at(datadir)
    pc.setup_env()
    _link_sources(pc, datasets)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import build_pile  # noqa: PLC0415

    build_pile.assert_vtscore_is_this_checkout()

    from vtscore.embedding import matrix  # noqa: PLC0415

    # THE treatment: one name, read at call time by the ingest cast.
    shipped = matrix.PATCH_ROW_DTYPE
    matrix.PATCH_ROW_DTYPE = np.float32
    log(f"PATCH_ROW_DTYPE {np.dtype(shipped).name} -> {np.dtype(matrix.PATCH_ROW_DTYPE).name}")

    device = {
        "hostname": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device["gpu_name"] = torch.cuda.get_device_name(0)
        device["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        device["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    log(f"device: {device}")

    summaries = []
    for ds in datasets:
        t0 = time.time()
        summary = build_pile.build_cell(ds, EMBEDDER, force=force)
        summary["wall_seconds"] = round(time.time() - t0, 1)
        summaries.append(summary)
        log(f"  {ds}: {summary}")

    # Assert the premise, not the parameter: a knob that silently did nothing
    # would produce float16 cells labelled float32, and every later number would
    # compare the pile against itself.
    from _cells_io import load_medias  # noqa: PLC0415

    for ds in datasets:
        medias = load_medias(pc.cell_path(ds, EMBEDDER))
        dtypes = {str(m["patch_grid"].dtype) for m in medias.values() if m.get("patch_grid") is not None}
        if dtypes != {"float32"}:
            raise SystemExit(f"{ds}: built grids are {dtypes}, not float32 -- the treatment did not apply")
        log(f"  {ds}: {len(medias)} medias, grids {dtypes}")

    out = STUDY / "piles" / "fp32" / "provenance.json"
    out.write_text(
        json.dumps({"arm": "fp32", "patch_row_dtype": "float32", "device": device, "cells": summaries}, indent=2) + "\n"
    )
    log(f"wrote {out}")
    return 0


#: Largest image-level (CLS) disagreement ``adopt-cls`` will paper over.  The
#: batch-32 rebuild reproduces the pile's GRID bit for bit but its CLS vectors
#: still differ by up to ~3e-8 on some images; anything near the float16
#: grid's own error (~1e-4) would be a different forward, not rounding.
CLS_ADOPT_TOLERANCE = 1e-6


def cmd_adopt_cls(datasets: list[str]) -> int:
    """Copy the pile's CLS vectors into the fp32 cells, so the arms differ ONLY by the cast.

    The CLS vector is stored float32 in both arms and never cast at ingest; it
    enters the float16 path only as row 0 of the scoring stack.  A rebuild
    that differed from the pile there by 3e-8 would put a second, unrelated
    difference into every paired comparison - small, but enough to reroute a
    trajectory, which is exactly what a paired null must not contain.  Refuses
    if any vector differs by more than :data:`CLS_ADOPT_TOLERANCE`.
    """
    import numpy as np  # noqa: PLC0415

    from _cells_io import dump_medias, load_medias  # noqa: PLC0415

    out: dict[str, dict] = {}
    for ds in datasets:
        path = arm_datadir("fp32") / "embeddings" / f"{ds}__{EMBEDDER}.pkl"
        f32 = load_medias(path)
        f16 = load_medias(SHARED_EMBEDDINGS / f"{ds}__{EMBEDDER}.pkl")
        changed, worst = 0, 0.0
        for i, m in f32.items():
            a = np.asarray(m["embeddings"][EMBEDDER], dtype=np.float32)
            b = np.asarray(f16[i]["embeddings"][EMBEDDER], dtype=np.float32)
            d = float(np.abs(a - b).max())
            worst = max(worst, d)
            if d > CLS_ADOPT_TOLERANCE:
                raise SystemExit(f"{ds}:{i}: CLS differs by {d:.1e} > {CLS_ADOPT_TOLERANCE}; not a rounding difference")
            if d > 0:
                m["embeddings"][EMBEDDER] = b.copy()
                changed += 1
        dump_medias(f32, path)
        out[ds] = {"cls_adopted": changed, "n": len(f32), "worst_abs_diff": worst}
        log(f"{ds}: adopted the pile's CLS on {changed}/{len(f32)} medias (worst diff {worst:.1e})")
    (STUDY / "adopt_cls.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


def _boxes(media: dict) -> list:
    return [(tuple(r.get("box") or ()), r.get("label")) for r in (media.get("regions") or [])]


def cmd_verify(datasets: list[str]) -> int:
    import numpy as np  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415

    report: dict[str, dict] = {}
    failed = False
    for ds in datasets:
        f32 = load_medias(arm_datadir("fp32") / "embeddings" / f"{ds}__{EMBEDDER}.pkl")
        f16 = load_medias(SHARED_EMBEDDINGS / f"{ds}__{EMBEDDER}.pkl")
        rec: dict = {"n_fp32": len(f32), "n_pile": len(f16)}
        rec["same_ids"] = sorted(f32) == sorted(f16)
        ids = sorted(set(f32) & set(f16))
        rec["same_categories"] = all(
            (f32[i].get("categories"), f32[i].get("category")) == (f16[i].get("categories"), f16[i].get("category"))
            for i in ids
        )
        rec["same_boxes"] = all(_boxes(f32[i]) == _boxes(f16[i]) for i in ids)

        cls_equal = 0
        cls_max = 0.0
        grid_bit_equal_media = 0
        grid_elems = 0
        grid_elems_diff = 0
        grid_dtypes: set[str] = set()
        for i in ids:
            a = np.asarray(f32[i]["embeddings"][EMBEDDER], dtype=np.float32)
            b = np.asarray(f16[i]["embeddings"][EMBEDDER], dtype=np.float32)
            cls_equal += int(np.array_equal(a, b))
            cls_max = max(cls_max, float(np.abs(a - b).max()))
            ga, gb = f32[i]["patch_grid"], f16[i]["patch_grid"]
            grid_dtypes.add(f"{ga.dtype}/{gb.dtype}")
            cast = ga.astype(np.float16)
            diff = int(np.count_nonzero(cast.view(np.uint16) != gb.view(np.uint16)))
            grid_elems += cast.size
            grid_elems_diff += diff
            grid_bit_equal_media += int(diff == 0)
        rec.update(
            {
                "grid_dtypes(fp32/pile)": sorted(grid_dtypes),
                "cls_bit_identical": f"{cls_equal}/{len(ids)}",
                "cls_max_abs_diff": cls_max,
                "cast_grid_bit_identical_medias": f"{grid_bit_equal_media}/{len(ids)}",
                "cast_grid_elements_differing": f"{grid_elems_diff}/{grid_elems}",
            }
        )
        ok = (
            rec["same_ids"]
            and rec["same_categories"]
            and rec["same_boxes"]
            and grid_elems_diff == 0
            and cls_equal == len(ids)
            and grid_dtypes == {"float32/float16"}
        )
        rec["verdict"] = "PASS: pile cell == this fp32 cell through the storage cast" if ok else "FAIL"
        failed |= not ok
        report[ds] = rec
        log(f"{ds}: {json.dumps(rec)}")

    out = STUDY / "verify.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {out}")
    if failed:
        log("VERIFY FAILED -- the fp32 cells are not the pile's forward; do not analyse")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "adopt-cls", "verify"])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    datasets = [d for d in args.datasets.split(",") if d]
    if args.command == "build":
        return cmd_build(datasets, args.force)
    if args.command == "adopt-cls":
        return cmd_adopt_cls(datasets)
    return cmd_verify(datasets)


if __name__ == "__main__":
    raise SystemExit(main())
