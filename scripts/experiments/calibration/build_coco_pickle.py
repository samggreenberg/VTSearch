"""Build a COCO-val dataset pickle from the #2790 region cache (issue #2841).

COCO is not a VTSearch demo dataset, so the calibration harness has never been
able to run on it.  It does not need to be one: the #2790 salient-object sweep
already embedded all 4952 COCO-2017-val images and left the vectors on the GRID
under ``<cache>/regions/coco/<embedder>/``, and COCO's boxes are staged in
flattened form alongside the images.  This stage joins the two into the ordinary
media-dict pickle every other arm of the study loads, so no model runs and
nothing is re-embedded.

    python build_coco_pickle.py --embedders siglip,siglip2

**Whole-image embedders only.**  The cached ``.npz`` carries each image's
``whole_vec`` (and, for the patch embedders, its 24 HAC *region* vectors) but
**not** the raw 14x14 patch grid.  ``max_patch`` and the HAC styles score by
max-pooling that grid, so they cannot be reconstructed from this cache at any
fidelity - a COCO region-voting arm would need a genuine re-embed.  This script
therefore refuses patch embedders outright rather than emitting a pickle that
would silently fall back to whole-image scoring and be reported as a region arm.

Categories come from the staged annotations (``objects_flat_val2017.jsonl.gz``,
one row per annotation with its box already normalised to ``[0, 1]``), so the
resulting medias carry the same ``categories`` + ``regions`` shape as the Visual
Genome pickles and feed the identical category selector.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402

#: Where the #2790 sweep left its per-image vectors.
DEFAULT_CACHE = Path(os.environ.get("MIXIN_COCO_CACHE", "/exp/sgreenberg/threshold-stability/cache"))
#: Staged COCO-2017-val annotations: one JSON row per annotation.
DEFAULT_ANNOTATIONS = Path(
    os.environ.get(
        "MIXIN_COCO_ANNOTATIONS",
        "/exp/scale26/datasets/external/COCO/derived/objects_flat_val2017.jsonl.gz",
    )
)


def _cache_dir(cache: Path, embedder: str) -> Path:
    """The per-embedder vector directory, whose one subdir names the region rule."""
    root = cache / "regions" / "coco" / embedder
    if not root.is_dir():
        raise SystemExit(f"no cached COCO vectors for {embedder!r} under {root}")
    subs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(subs) != 1:
        raise SystemExit(f"expected exactly one region-rule dir under {root}, found {[p.name for p in subs]}")
    return subs[0]


def _annotations(path: Path) -> tuple[dict[int, list[dict]], dict[int, str]]:
    """``({image_id: [{box, label}, ...]}, {image_id: file_name})`` from the flat rows.

    ``iscrowd`` regions are kept: they are still true instances of the category,
    and the study's positives are defined by category presence, not by whether a
    box is individually resolvable.
    """
    regions: dict[int, list[dict]] = defaultdict(list)
    filenames: dict[int, str] = {}
    with gzip.open(path, "rt") as fh:
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


def build(embedder: str, cache: Path, annotations: Path, out_dir: Path) -> Path:
    """Write ``coco_val__<embedder>.pkl`` and return its path."""
    if cfg.is_patch_embedder(embedder):
        raise SystemExit(
            f"{embedder!r} is a patch embedder: the #2790 cache stores HAC region vectors but not the "
            "raw patch grid the max_patch/HAC styles score on, so a region arm cannot be built from it."
        )

    vec_dir = _cache_dir(cache, embedder)
    regions_by_image, filenames = _annotations(annotations)

    medias: dict[int, dict] = {}
    missing_vectors: list[int] = []
    for npz_path in sorted(vec_dir.glob("*.npz"), key=lambda p: int(p.stem)):
        image_id = int(npz_path.stem)
        regions = regions_by_image.get(image_id)
        if not regions:
            # An image the sweep embedded but the annotation file does not
            # cover (or covers with zero objects) has no category, so it can be
            # neither a positive nor a meaningful negative.  Skip it.
            continue
        with np.load(npz_path) as z:
            whole = np.asarray(z["whole_vec"], dtype=np.float32)
        if whole.ndim != 1:
            missing_vectors.append(image_id)
            continue
        # Category order: most-annotated first, so ``category`` (the singular
        # "primary" field the media dict carries) is the image's dominant object.
        counts: dict[str, int] = defaultdict(int)
        for r in regions:
            counts[r["label"]] += 1
        ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        medias[image_id] = {
            "id": image_id,
            "media_type": "image",
            "embedder": embedder,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {embedder: whole},
            "media_string": None,
            "filename": Path(filenames[image_id]).name,
            "category": ordered[0],
            "categories": ordered,
            "regions": regions,
            "origin": {"importer": "cached_coco_val", "params": {"embedder": embedder}},
            "origin_name": filenames[image_id],
        }

    if not medias:
        raise SystemExit(f"no medias assembled for {embedder!r} - check {vec_dir} and {annotations}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.pickle_name("coco_val", embedder)
    with out_path.open("wb") as fh:
        pickle.dump(medias, fh, protocol=pickle.HIGHEST_PROTOCOL)

    n_cats = len({c for m in medias.values() for c in m["categories"]})
    print(
        f"{embedder}: {len(medias)} medias, {n_cats} categories -> {out_path}"
        + (f"  (skipped {len(missing_vectors)} with unusable vectors)" if missing_vectors else "")
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedders", default="siglip,siglip2", help="comma-separated whole-image embedders")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="the #2790 sweep cache root")
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS, help="flattened COCO rows")
    ap.add_argument("--out-dir", type=Path, default=None, help="datadir/embeddings (default: from common)")
    args = ap.parse_args()

    out_dir = args.out_dir or common.DATADIR / "embeddings"
    for embedder in [e.strip() for e in args.embedders.split(",") if e.strip()]:
        build(embedder, args.cache, args.annotations, out_dir)


if __name__ == "__main__":
    main()
