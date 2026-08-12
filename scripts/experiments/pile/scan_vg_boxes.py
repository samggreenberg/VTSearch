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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pile_config as pc

pc.setup_env()

VG_ROOT = pc.DEMO_CACHE / "visual_genome"
IMAGE_DIRS = [VG_ROOT / "VG_100K", VG_ROOT / "VG_100K_2"]
OBJECTS_JSON = VG_ROOT / "objects.json"
DIMS_CACHE = pc.PILE / "vg_image_dims.json"


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


def _read_dims(paths: dict[int, Path], workers: int = 16) -> dict[int, tuple[int, int]]:
    """``{image_id: (w, h)}``, cached to disk. Header-only reads, threaded."""
    if DIMS_CACHE.exists():
        cached = {int(k): tuple(v) for k, v in json.loads(DIMS_CACHE.read_text()).items()}
        if len(cached) >= len(paths):
            log(f"reusing cached dims for {len(cached)} images")
            return cached  # type: ignore[return-value]

    from PIL import Image  # noqa: PLC0415

    def one(item):
        iid, path = item
        try:
            with Image.open(path) as im:  # header only; no decode
                return iid, im.size
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            return iid, None

    t0 = time.time()
    dims: dict[int, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (iid, size) in enumerate(ex.map(one, paths.items(), chunksize=256), 1):
            if size:
                dims[iid] = size
            if i % 20000 == 0:
                log(f"  dims {i}/{len(paths)} ({time.time() - t0:.0f}s)")
    log(f"read dims for {len(dims)}/{len(paths)} images in {time.time() - t0:.0f}s")
    DIMS_CACHE.write_text(json.dumps({str(k): list(v) for k, v in dims.items()}))
    return dims


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """The single box a Good vote drags: the union over a category's instances."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


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

    for rec in records:
        iid = int(rec["image_id"])
        wh = dims.get(iid)
        if wh is None:
            continue
        W, H = wh
        if W <= 0 or H <= 0:
            continue
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
            voted[name].append(max(0.0, (ux1 - ux0)) * max(0.0, (uy1 - uy0)) / area)
            instance[name].extend((b[2] - b[0]) * (b[3] - b[1]) / area for b in boxes)
            n_images[name] += 1

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
        }
    log(f"{len(stats)} categories with >= {min_images} images (of {len(voted)} distinct names)")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-images", type=int, default=20, help="minimum images per category (default 20)")
    ap.add_argument("--out", default=str(pc.PILE / "vg_box_scale.json"), help="where to write the scan")
    args = ap.parse_args()

    stats = scan(args.min_images)
    Path(args.out).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    log(f"wrote {args.out}")

    # Report against the geometry-anchored bands the calibration harness uses.
    PATCH, LEAF = 1 / 196, 1 / 12
    bands = [
        ("sub_patch", 0.0, PATCH),
        ("patch_to_leaf", PATCH, LEAF),
        ("leaf_to_4x", LEAF, 4 * LEAF),
        ("above_4x", 4 * LEAF, 1.01),
    ]
    log("")
    log("=== full-VG category counts per geometry band ===")
    for name, lo, hi in bands:
        pool = [c for c, s in stats.items() if lo <= s["voted_area"] < hi]
        clean = [c for c in pool if stats[c]["union_inflation"] <= 1.5]
        log(
            f"  {name:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): {len(pool):5d} categories "
            f"({len(clean)} with union_inflation <= 1.5)"
        )
        example = sorted(pool, key=lambda c: -stats[c]["n_images"])[:8]
        log(f"      most-supported: {example}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
