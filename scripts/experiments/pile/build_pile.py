"""Build (and verify) the shared pre-embedded pile of ``(dataset, embedder)`` cells.

One cell = one ``<dataset>__<embedder>.pkl`` of media dicts carrying vectors
(and ``patch_grid`` for patch embedders) but no pixels. Studies point
``VTSEARCH_DATA_DIR`` at the pile and load cells in place, so an embedder runs
once per pair ever rather than once per study.

Idempotent: a cell that already exists is skipped unless ``--force``. That makes
this safe to re-run after a partial SLURM job, and makes it the rebuild path if
scratch is ever purged.

Usage::

    python build_pile.py --list                      # what exists / what's missing
    python build_pile.py                             # build every missing cell
    python build_pile.py --datasets coco_val         # just COCO's cells
    python build_pile.py --embedders siglip2,siglip2_l
    python build_pile.py --verify                    # load every cell, check geometry
    python build_pile.py --manifest                  # (re)write MANIFEST.{json,md}

``--verify`` is the guard the region-voting studies needed: it asserts that
every cell whose ``(dataset, embedder)`` pair claims region capability actually
carries ``patch_grid`` on its medias, and that no cell silently holds zero.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[pile] {msg}", flush=True)


def _cells_io():
    """Import the calibration harness's pickle IO (drops bytes, keeps patch_grid)."""
    calib = Path(__file__).resolve().parent.parent / "calibration"
    if str(calib) not in sys.path:
        sys.path.insert(0, str(calib))
    import _cells_io  # noqa: PLC0415

    return _cells_io


# --------------------------------------------------------------------------
# COCO: not a demo dataset, so assemble medias from the staged zip + annotations
# --------------------------------------------------------------------------


def _coco_annotations() -> tuple[dict[int, list[dict]], dict[int, str]]:
    """``({image_id: [{box, label}, ...]}, {image_id: file_name})``.

    Mirrors ``calibration/build_coco_pickle.py``: boxes already normalised to
    [0, 1], ``iscrowd`` regions kept (they are still true instances of the
    category, and positives are defined by category presence).
    """
    regions: dict[int, list[dict]] = defaultdict(list)
    filenames: dict[int, str] = {}
    with gzip.open(pc.COCO_ANNOTATIONS, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            image_id = int(row["image_id"])
            filenames[image_id] = row["file_name"]
            regions[image_id].append(
                {
                    "box": [float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])],
                    "label": row["name"],
                }
            )
    return dict(regions), filenames


def _load_coco(medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with COCO-val images (bytes read straight from the zip).

    Only images the annotation file covers are kept: an image with no category
    can be neither a positive nor a meaningful negative.
    """
    if not pc.COCO_ANNOTATIONS.exists():
        raise SystemExit(f"missing COCO annotations: {pc.COCO_ANNOTATIONS}")
    zip_path = pc.COCO_ROOT / "images" / "val2017.zip"
    if not zip_path.exists():
        raise SystemExit(f"missing COCO images zip: {zip_path}")

    regions_by_image, filenames = _coco_annotations()
    log(f"  coco: {len(regions_by_image)} annotated images")

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".jpg")}
        missing = 0
        for image_id in sorted(regions_by_image):
            regions = regions_by_image[image_id]
            fname = Path(filenames[image_id]).name
            member = members.get(fname)
            if member is None:
                missing += 1
                continue
            counts: dict[str, int] = defaultdict(int)
            for r in regions:
                counts[r["label"]] += 1
            # Most-annotated first, so ``category`` is the dominant object.
            ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
            medias[image_id] = {
                "id": image_id,
                "media_type": "image",
                "embedder": embedder_name,
                "duration": 0,
                "file_size": 0,
                "md5": "",
                "embeddings": {},
                "media_bytes": zf.read(member),
                "media_string": None,
                "filename": fname,
                "category": ordered[0],
                "categories": ordered,
                "regions": regions,
                "origin": {"importer": "staged_coco_val", "params": {"embedder": embedder_name}},
                "origin_name": filenames[image_id],
            }
    if missing:
        log(f"  coco: WARNING {missing} annotated images absent from the zip")


def _load_demo(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    from vtscore.datasets.loader_demo import load_demo_dataset  # noqa: PLC0415

    load_demo_dataset(dataset, medias, embedder_name=embedder_name)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_cell(dataset: str, embedder: str, force: bool = False) -> dict:
    """Build one cell, returning a summary record."""
    out = pc.cell_path(dataset, embedder)
    if out.exists() and not force:
        log(f"skip {dataset} x {embedder} (exists: {out.name})")
        return {"dataset": dataset, "embedder": embedder, "status": "exists"}

    if pc.EMBEDDERS.get(embedder, {}).get("gated") and not os.environ.get("HF_TOKEN"):
        log(f"SKIP {dataset} x {embedder}: HF_TOKEN unset (weights are licence-gated)")
        return {"dataset": dataset, "embedder": embedder, "status": "skipped_gated"}

    kind = pc.DATASETS[dataset]["kind"]
    log(f"=== build {dataset} x {embedder} ({kind}) ===")
    t0 = time.time()

    medias: dict[int, dict] = {}
    if kind == "coco":
        _load_coco(medias, embedder)
    else:
        pc.require_demo_source(dataset)
        _load_demo(dataset, medias, embedder)
    log(f"  loaded {len(medias)} medias in {time.time() - t0:.0f}s")

    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    t1 = time.time()
    embed_missing(medias, embedder)
    embed_s = time.time() - t1

    n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
    nbytes = _cells_io().dump_medias(medias, out)
    total_s = time.time() - t0
    log(
        f"  wrote {out.name}: {nbytes / 1e6:.0f} MB, {len(medias)} medias, "
        f"patch grids {n_patch}/{len(medias)}, embed {embed_s:.0f}s, total {total_s:.0f}s"
    )
    return {
        "dataset": dataset,
        "embedder": embedder,
        "status": "built",
        "n_medias": len(medias),
        "n_patch_grids": n_patch,
        "megabytes": round(nbytes / 1e6, 1),
        "embed_seconds": round(embed_s, 1),
    }


# --------------------------------------------------------------------------
# Verify + manifest
# --------------------------------------------------------------------------


def verify() -> int:
    """Load every present cell and check it is usable. Returns an exit code."""
    io = _cells_io()
    problems: list[str] = []
    rows = []
    counts_by_dataset: dict[str, dict[str, int]] = defaultdict(dict)
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        if not path.exists():
            rows.append((ds, emb, "MISSING", "", "", ""))
            continue
        medias = io.load_medias(path)
        n = len(medias)
        counts_by_dataset[ds][emb] = n
        n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
        first = next(iter(medias.values()), None)
        dim = ""
        if first is not None:
            from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

            vec = media_embedding(first)
            dim = str(len(vec)) if vec is not None else "NO-VECTOR"
        want_region = pc.region_capable(ds, emb)
        state = "ok"
        if n == 0:
            state = "EMPTY"
            problems.append(f"{ds} x {emb}: 0 medias")
        elif dim in ("", "NO-VECTOR"):
            state = "NO-VECTOR"
            problems.append(f"{ds} x {emb}: medias carry no embedding")
        elif want_region and n_patch < n:
            state = "PATCH-GAP"
            problems.append(f"{ds} x {emb}: region-capable but patch_grid on only {n_patch}/{n}")
        elif not pc.is_patch_embedder(emb) and n_patch:
            state = "UNEXPECTED-PATCH"
            problems.append(f"{ds} x {emb}: single-vector embedder carries patch grids")
        rows.append((ds, emb, state, str(n), f"{n_patch}/{n}", dim))

    # A dataset's cells must all cover the same medias, or cross-embedder
    # comparisons silently compare different populations. This is not
    # hypothetical: a datadir missing its demo-source symlink sent the loader
    # off to re-download the dataset, and it embedded a truncated 1662-media
    # subset of a 4193-media dataset into a cell that otherwise looked healthy.
    for ds, per_emb in counts_by_dataset.items():
        if len(set(per_emb.values())) > 1:
            majority = max(set(per_emb.values()), key=list(per_emb.values()).count)
            odd = {e: n for e, n in per_emb.items() if n != majority}
            problems.append(
                f"{ds}: cells disagree on media count (most are {majority}); "
                f"rebuild {', '.join(f'{e} ({n})' for e, n in sorted(odd.items()))}"
            )

    log(f"{'dataset':18s} {'embedder':14s} {'state':16s} {'medias':>7s} {'patch':>12s} {'dim':>6s}")
    for ds, emb, state, n, patch, dim in rows:
        log(f"{ds:18s} {emb:14s} {state:16s} {n:>7s} {patch:>12s} {dim:>6s}")

    if problems:
        log("")
        for p in problems:
            log(f"PROBLEM: {p}")
        return 1
    log("all present cells verified")
    return 0


def write_manifest() -> None:
    """Write MANIFEST.json + MANIFEST.md describing the pile and how to rebuild it."""
    io = _cells_io()
    entries = []
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        if not path.exists():
            entries.append({"dataset": ds, "embedder": emb, "present": False})
            continue
        medias = io.load_medias(path)
        n = len(medias)
        entries.append(
            {
                "dataset": ds,
                "embedder": emb,
                "present": True,
                "file": path.name,
                "megabytes": round(path.stat().st_size / 1e6, 1),
                "n_medias": n,
                "n_patch_grids": sum(1 for m in medias.values() if m.get("patch_grid") is not None),
                "region_capable": pc.region_capable(ds, emb),
            }
        )

    doc = {
        "pile": str(pc.PILE),
        "sources": {
            "demo_cache": str(pc.DEMO_CACHE),
            "coco_root": str(pc.COCO_ROOT),
        },
        "datasets": pc.DATASETS,
        "embedders": pc.EMBEDDERS,
        "cells": entries,
    }
    (pc.PILE / "MANIFEST.json").write_text(json.dumps(doc, indent=2) + "\n")

    present = [e for e in entries if e["present"]]
    total_mb = sum(e["megabytes"] for e in present)
    lines = [
        "# Pre-embedded pile",
        "",
        f"`{pc.PILE}` — {len(present)}/{len(entries)} cells, {total_mb / 1000:.1f} GB of embeddings.",
        "",
        "Point a study at it with:",
        "",
        "```bash",
        f'export VTSEARCH_DATA_DIR="{pc.DATADIR}"',
        f'export VTSEARCH_MODELS_DIR="{pc.MODELS}"',
        "```",
        "",
        "## Cells",
        "",
        "| dataset | embedder | medias | patch grids | region-voting | size |",
        "|---|---|---:|---:|:--:|---:|",
    ]
    for e in entries:
        if not e["present"]:
            lines.append(f"| `{e['dataset']}` | `{e['embedder']}` | — | — | — | *missing* |")
            continue
        region = "**yes**" if e["region_capable"] else "no"
        lines.append(
            f"| `{e['dataset']}` | `{e['embedder']}` | {e['n_medias']} | "
            f"{e['n_patch_grids']} | {region} | {e['megabytes']:.0f} MB |"
        )
    lines += [
        "",
        "**Region voting needs both halves**: ground-truth boxes (dataset) *and* a patch",
        "grid (embedder). A boxed dataset on a single-vector embedder silently runs as",
        "binary voting — the failure behind #2877, #2897 and #2905. `build_pile.py --verify`",
        "asserts the geometry rather than trusting the arm table.",
        "",
        "## Rebuilding",
        "",
        "Scratch is treated as purgeable. Every cell rebuilds from staged, non-scratch",
        "sources, so the pile is disposable:",
        "",
        "```bash",
        "python build_pile.py            # rebuild whatever is missing (idempotent)",
        "python build_pile.py --verify   # check geometry after a rebuild",
        "```",
        "",
        f"Sources: demo datasets from `{pc.DEMO_CACHE}`, COCO from `{pc.COCO_ROOT}`.",
        "",
    ]
    (pc.PILE / "MANIFEST.md").write_text("\n".join(lines))
    log(f"wrote MANIFEST.json + MANIFEST.md ({len(present)}/{len(entries)} cells)")


def list_cells() -> None:
    log(f"pile: {pc.PILE}")
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        mark = "present" if path.exists() else "MISSING"
        size = f"{path.stat().st_size / 1e6:8.0f} MB" if path.exists() else " " * 11
        region = " region-voting" if pc.region_capable(ds, emb) else ""
        log(f"  {ds:18s} x {emb:14s} {mark:8s} {size}{region}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", help="comma-separated subset (default: all)")
    ap.add_argument("--embedders", help="comma-separated subset (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild cells that already exist")
    ap.add_argument("--list", action="store_true", help="show cell status and exit")
    ap.add_argument("--verify", action="store_true", help="load every cell and check geometry")
    ap.add_argument("--manifest", action="store_true", help="(re)write the manifest and exit")
    args = ap.parse_args()

    pc.EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_cells()
        return 0
    if args.verify:
        return verify()
    if args.manifest:
        write_manifest()
        return 0

    datasets = args.datasets.split(",") if args.datasets else list(pc.DATASETS)
    embedders = args.embedders.split(",") if args.embedders else list(pc.EMBEDDERS)
    for bad in [d for d in datasets if d not in pc.DATASETS]:
        raise SystemExit(f"unknown dataset {bad!r}; known: {sorted(pc.DATASETS)}")
    for bad in [e for e in embedders if e not in pc.EMBEDDERS]:
        raise SystemExit(f"unknown embedder {bad!r}; known: {sorted(pc.EMBEDDERS)}")

    summaries = []
    for ds in datasets:
        for emb in embedders:
            summaries.append(build_cell(ds, emb, force=args.force))

    built = [s for s in summaries if s["status"] == "built"]
    log(f"done: {len(built)} built, {len(summaries) - len(built)} skipped")
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
