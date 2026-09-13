"""Scan the FULL Visual Genome annotations to band object categories by box size.

The pile's `visual_genome_m` cell is the demo pipeline's view of VG: a curated
100-category vocabulary, capped at 1000 items each, over a 4% ``slice_frac``
window (1/50 -> 3/50). That is the right dataset for "load something into the
UI quickly" and the wrong one for "how does box scale behave", because the
scale question is about *which categories exist at which size*, and the demo
vocabulary was chosen for recognisability, not for spanning scales.

This scans the original source instead -- all 108k images across VG_100K and
VG_100K_2, with the full free-text object vocabulary from ``objects.json`` --
and reports the per-category voted-box scale so banded datasets can be built
from the whole pool.

**Per-band supply, not just a median.** The scan reports each category's *whole
distribution* over the size bands, not only its median. Banding a category by
its median puts it in exactly one band, which is what made the three
``vg_box_*`` sets carry disjoint vocabularies -- so a small-vs-large difference
confounded box size with class identity (noses vs fences). The question worth
asking is "how well can we find buses in the middleground", which needs the
*same* class present at every size, and that needs the histogram: how many
images hold a background bus, a middleground bus, a foreground bus. See
``shortlist_scale_classes.py``, which reads this scan and ranks the classes that
have real support in all three bands.

Areas are the **union** over a category's instances in an image -- the box one
Good vote actually drags (``vtscore.eval.labels.region_box_for_category``) -- so
an image holding one foreground bus and three background buses is a
foreground-bus image, which is the honest reading of what the user would draw.

``objects.json`` stores boxes in **pixels** and carries no image dimensions, so
areas are normalised against dims read from each JPEG header (PIL reads the
header without decoding the image, which is what makes 108k files tractable).
Dims are cached; the scan is re-runnable.

Usage::

    python scan_vg_boxes.py                 # scan, cache, report
    python scan_vg_boxes.py --min-images 50 # require more support per category
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pile_config as pc

from pilebuild.vgsource import image_dims, vg_dims_cache

pc.setup_env()

VG_ROOT = pc.DEMO_CACHE / "visual_genome"
IMAGE_DIRS = [VG_ROOT / "VG_100K", VG_ROOT / "VG_100K_2"]
OBJECTS_JSON = VG_ROOT / "objects.json"
DIMS_CACHE = vg_dims_cache()


def log(msg: str) -> None:
    print(f"[vgscan] {msg}", flush=True)


def _image_paths() -> dict[int, Path]:
    """``{image_id: path}`` over both VG image dirs."""
    out: dict[int, Path] = {}
    for d in IMAGE_DIRS:
        if not d.is_dir():
            raise SystemExit(f"missing VG image dir: {d}")
        for p in d.iterdir():
            if p.suffix.lower() == ".jpg":
                try:
                    out[int(p.stem)] = p
                except ValueError:
                    continue
    return out


def _read_dims(paths: dict[int, Path]) -> dict[int, tuple[int, int]]:
    """``{image_id: (w, h)}``, filling and rewriting the shared cache.

    The scan is the cache's **owner**: it is the one caller that writes it, and
    the one that passes ``write=True``. Everything else reads (``vg_source``).

    The rule itself lives in :func:`pilebuild.vgsource.image_dims` rather than
    here, because two implementations of "is this cache usable?" is how it broke:
    a length check that both copies agreed on still rejected a cache 170 entries
    short of the image directory, and every VG build re-read 108k JPEG headers
    for a month with only a log line to say so (#3822).
    """
    t0 = time.time()
    dims = image_dims(paths, DIMS_CACHE, write=True)
    log(f"dims for {len(dims)}/{len(paths)} images in {time.time() - t0:.0f}s")
    return dims


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """The single box a Good vote drags: the union over a category's instances."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def _band_of(area: float) -> str:
    """Which size band a voted-box *area* falls in.

    ``oversize`` is above ``MAX_VOTED_AREA``: a box covering >80% of the image
    is not a region, it is the image, so those instances are counted and
    reported but are not usable as band positives.
    """
    for name, (lo, hi) in pc.BOX_BANDS.items():
        if lo <= area < hi:
            return name
    return "oversize"


def scan(min_images: int) -> dict:
    """Per-category voted-box scale over the whole of VG."""
    paths = _image_paths()
    log(f"found {len(paths)} images across {len(IMAGE_DIRS)} dirs")
    dims = _read_dims(paths)

    log(f"loading {OBJECTS_JSON.name} ({OBJECTS_JSON.stat().st_size / 1e6:.0f} MB)...")
    t0 = time.time()
    with OBJECTS_JSON.open() as fh:
        records = json.load(fh)
    log(f"  parsed {len(records)} image records in {time.time() - t0:.0f}s")

    # category -> list of per-image (voted_area, instance_area) fractions
    voted: dict[str, list[float]] = defaultdict(list)
    instance: dict[str, list[float]] = defaultdict(list)
    n_images: dict[str, int] = defaultdict(int)
    # category -> {band: how many images hold this category at that size}
    bands: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys((*pc.BOX_BANDS, "oversize"), 0))
    # Same histogram, but counting only images where the union box is COMPACT:
    # within `BAND_MAX_INFLATION` of this image's own largest instance. A class
    # is scattered *in an image*, not in general -- one bus in the foreground is
    # a foreground bus however many other images hold four scattered buses. The
    # per-class `union_inflation` filter drops the whole class on that evidence,
    # which costs 30 of the 65 COCO classes (`bus` among them, at 1.71).
    bands_compact: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys((*pc.BOX_BANDS, "oversize"), 0))
    n_compact: dict[str, int] = defaultdict(int)
    n_scanned = 0

    for rec in records:
        iid = int(rec["image_id"])
        wh = dims.get(iid)
        if wh is None:
            continue
        W, H = wh
        if W <= 0 or H <= 0:
            continue
        n_scanned += 1
        area = float(W * H)
        by_name: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if not name:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            by_name[name].append((x, y, x + w, y + h))
        for name, boxes in by_name.items():
            ux0, uy0, ux1, uy1 = _union(boxes)
            union_area = max(0.0, (ux1 - ux0)) * max(0.0, (uy1 - uy0)) / area
            voted[name].append(union_area)
            instance[name].extend((b[2] - b[0]) * (b[3] - b[1]) / area for b in boxes)
            n_images[name] += 1
            bands[name][_band_of(union_area)] += 1
            largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / area
            if union_area <= largest * pc.BAND_MAX_INFLATION:
                n_compact[name] += 1
                bands_compact[name][_band_of(union_area)] += 1

    stats = {}
    for name, areas in voted.items():
        if n_images[name] < min_images:
            continue
        v = float(statistics.median(areas))
        i = float(statistics.median(instance[name]))
        stats[name] = {
            "voted_area": v,
            "instance_area": i,
            "union_inflation": (v / i) if i > 0 else float("inf"),
            "n_images": n_images[name],
            # Positive supply per band. The clean-negative supply is the
            # complement, `meta.n_images_scanned - n_images`: an image holding
            # no instance of the category at any size.
            "bands": dict(bands[name]),
            # Positives under the per-image rule: the non-compact remainder is
            # *excluded* from the cell, not turned into a negative.
            "bands_compact": dict(bands_compact[name]),
            "n_compact": n_compact[name],
            "compact_frac": n_compact[name] / n_images[name],
        }
    log(f"{len(stats)} categories with >= {min_images} images (of {len(voted)} distinct names)")
    return {
        "meta": {
            "n_images_scanned": n_scanned,
            "min_images": min_images,
            "bands": {name: list(edges) for name, edges in pc.BOX_BANDS.items()},
            "max_voted_area": pc.MAX_VOTED_AREA,
        },
        "categories": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-images", type=int, default=20, help="minimum images per category (default 20)")
    ap.add_argument("--out", default=str(pc.PILE / "vg_box_scale.json"), help="where to write the scan")
    args = ap.parse_args()

    out = scan(args.min_images)
    stats = out["categories"]
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    log(f"wrote {args.out}")

    # Report against the geometry-anchored bands the calibration harness uses.
    PATCH, LEAF = 1 / 196, 1 / 12
    median_bands = [
        ("sub_patch", 0.0, PATCH),
        ("patch_to_leaf", PATCH, LEAF),
        ("leaf_to_4x", LEAF, 4 * LEAF),
        ("above_4x", 4 * LEAF, 1.01),
    ]
    log("")
    log("=== categories BANDED BY MEDIAN (one band each -- the old construction) ===")
    for name, lo, hi in median_bands:
        pool = [c for c, s in stats.items() if lo <= s["voted_area"] < hi]
        clean = [c for c in pool if stats[c]["union_inflation"] <= 1.5]
        log(
            f"  {name:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): {len(pool):5d} categories "
            f"({len(clean)} with union_inflation <= 1.5)"
        )
        example = sorted(pool, key=lambda c: -stats[c]["n_images"])[:8]
        log(f"      most-supported: {example}")

    # The number that actually decides whether a same-class-across-bands study
    # is possible: how many categories are present at EVERY size.
    log("")
    log("=== categories present in ALL THREE bands (the new construction) ===")
    for floor in (25, 50, 100, 200):
        viable = [c for c, s in stats.items() if min(s["bands"][b] for b in pc.BOX_BANDS) >= floor]
        compact = [c for c, s in stats.items() if min(s["bands_compact"][b] for b in pc.BOX_BANDS) >= floor]
        log(
            f"  >= {floor:4d} images in every band: {len(viable):5d} categories "
            f"({len(compact)} counting only compact-union images)"
        )
    log("  (rank and filter them with shortlist_scale_classes.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
