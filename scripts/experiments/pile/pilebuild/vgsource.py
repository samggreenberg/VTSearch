"""Reading the Visual Genome source tree: image paths, records, dimensions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pile_config as pc

from pilebuild.env import log


def vg_image_paths() -> dict[int, Path]:
    """``{image_id: path}`` over both VG image dirs."""
    vg_root = pc.DEMO_CACHE / "visual_genome"
    paths: dict[int, Path] = {}
    for d in (vg_root / "VG_100K", vg_root / "VG_100K_2"):
        for p in d.iterdir():
            if p.suffix.lower() == ".jpg":
                try:
                    paths[int(p.stem)] = p
                except ValueError:
                    continue
    return paths


def vg_objects_json() -> Path:
    """The VG annotation file every VG-derived build reads.

    Named once, and both the loaders and the rebuild canary go through it. A
    canary that spells a source path of its own is how ``coco_val`` reported
    REBUILD-BROKEN against a staging area that was entirely intact (#3299).
    """
    return pc.DEMO_CACHE / "visual_genome" / "objects.json"


def vg_boxes_by_name(rec: dict, wanted: set[str]) -> dict[str, list[list[float]]]:
    """This record's boxes for the categories in *wanted*, in VG pixel space.

    VG names an object with a list of synonyms and the first is its primary, so
    only that one is matched -- taking any of them would file one object under
    several categories. Degenerate boxes drop out.

    **In this release of VG there is nothing else to take.** All 2,516,939
    objects carry a ``names`` list of length exactly one (#3618), so the
    primary-name restriction costs nothing here and reading the rest of the list
    is not an available fix for a class built from one spelling. That fix is
    :data:`pile_config.SCALE_VG_NAMES`.
    """
    from collections import defaultdict  # noqa: PLC0415

    by_name: dict[str, list[list[float]]] = defaultdict(list)
    for obj in rec.get("objects") or []:
        names = obj.get("names") or []
        if not names:
            continue
        name = str(names[0]).strip().lower()
        if name not in wanted:
            continue
        x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
        w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
        if w > 0 and h > 0:
            by_name[name].append([x, y, x + w, y + h])
    return dict(by_name)


def vg_dims_cache() -> Path:
    """The one path the dims cache is spelled at, for the same reason as
    :func:`vg_objects_json`: two spellings of a shared artifact is how a caller
    ends up reporting on a file nobody else is using (#3299)."""
    return pc.PILE / "vg_image_dims.json"


def read_jpeg_dims(paths: dict[int, Path], workers: int = 16) -> tuple[dict[int, tuple[int, int]], list[int]]:
    """``(dims, ids whose header would not read)``, from the JPEGs themselves.

    Header-only reads (PIL does not decode), threaded, which is what makes 108k
    files tractable. **The misses are returned rather than dropped**: "this file
    is corrupt" is a fact worth caching, and losing it is what made the cache
    unusable for a month (#3822).
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    def one(item):
        iid, path = item
        try:
            with Image.open(path) as im:  # header only; no decode
                return iid, im.size
        except Exception:  # noqa: BLE001 - a corrupt file is a result, not an error
            return iid, None

    dims: dict[int, tuple[int, int]] = {}
    misses: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for iid, size in ex.map(one, paths.items(), chunksize=256):
            if size:
                dims[iid] = size
            else:
                misses.append(iid)
    return dims, misses


def image_dims(paths: dict[int, Path], cache: Path | None = None, *, write: bool = False) -> dict[int, tuple[int, int]]:
    """``{image_id: (w, h)}`` for *paths*, reading only the headers the cache lacks.

    **Per image, not all-or-nothing, and that is the whole repair.** The cache
    used to be accepted only when ``len(raw) >= len(paths)``, so a single file
    appearing in ``VG_100K_2`` disabled it for every caller -- and one had. The
    file on scratch held 108,075 entries against 108,245 JPEGs, exactly the 170
    corrupt images an older writer dropped instead of recording, so the guard
    had been failing since the day the cache was written and every VG build
    silently re-read 108k headers (#3822). Filling the gaps instead means the
    cache degrades by the number of files it has not seen rather than collapsing,
    and a caller that writes it back repairs it in passing.

    A ``null`` entry is an answer, not a gap: it says the header was read and
    would not parse, so that image is not re-read on every run. Ids the cache
    knows but *paths* does not are dropped -- the contract is "dims for these
    files", which is what keeps a cache outliving its source from quietly
    widening a build.

    *write* is opt-in and only the scan (the cache's owner) passes it: a build is
    not the right thing to have writing a shared artifact, and several run at
    once. The write it does do is atomic, so the losing side of a race leaves a
    whole file rather than half of one.
    """
    cache = cache or vg_dims_cache()
    known: dict[int, tuple[int, int] | None] = {}
    if cache.exists():
        known = {int(k): (tuple(v) if v else None) for k, v in json.loads(cache.read_text()).items()}  # type: ignore[misc]

    missing = {iid: p for iid, p in paths.items() if iid not in known}
    if missing:
        log(f"  dims: {len(paths) - len(missing)} cached, reading {len(missing)} JPEG header(s)")
        got, bad = read_jpeg_dims(missing)
        known.update(got)
        known.update(dict.fromkeys(bad))
    else:
        log(f"  dims: all {len(paths)} from {cache.name}")

    if write and missing:
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({str(k): (list(v) if v else None) for k, v in known.items()}))
        os.replace(tmp, cache)
        log(f"  dims: wrote {len(known)} entries to {cache.name}")

    return {iid: wh for iid, wh in known.items() if wh and iid in paths}


def vg_source() -> tuple[dict[int, Path], list, dict[int, tuple[int, int]]]:
    """``(image paths, objects.json records, image dims)`` for the whole VG source.

    Dims come from ``scan_vg_boxes.py``'s cache for every image it has seen and
    from the JPEG headers for the rest -- see :func:`image_dims`. Read-only: a
    build never rewrites the cache.
    """
    objects_json = vg_objects_json()
    if not objects_json.exists():
        raise SystemExit(f"missing {objects_json}")

    paths = vg_image_paths()
    dims = image_dims(paths)

    with objects_json.open() as fh:
        records = json.load(fh)
    return paths, records, dims
