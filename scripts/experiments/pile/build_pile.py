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
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[pile] {msg}", flush=True)


def assert_vtscore_is_this_checkout() -> None:
    """Refuse to run against a different checkout's ``vtscore``.

    The venv's editable install points at the main checkout. If anything
    resolves ``vtscore`` there instead of here, cells get embedded by whatever
    code that tree happens to be on — silently, and possibly by a different
    embedder implementation. Cheap to assert, expensive to discover later.
    """
    import vtscore  # noqa: PLC0415

    want = Path(__file__).resolve().parents[3]
    got = Path(vtscore.__file__).resolve().parent.parent
    if got != want:
        raise SystemExit(
            f"vtscore resolved to {got}, not this checkout ({want}).\n"
            f"  Something put another checkout ahead on sys.path — usually the venv's\n"
            f"  editable install. Re-run with VTS_REPO={want} set for THIS command\n"
            f"  (note `VAR=x cmd1 && cmd2` sets VAR for cmd1 only)."
        )


def _calibration_path() -> None:
    calib = Path(__file__).resolve().parent.parent / "calibration"
    if str(calib) not in sys.path:
        sys.path.insert(0, str(calib))


def _cells_io():
    """Import the calibration harness's pickle IO (drops bytes, keeps patch_grid)."""
    _calibration_path()
    import _cells_io  # noqa: PLC0415

    return _cells_io


def _experiment_config():
    """Import the calibration harness's category-selection config."""
    _calibration_path()
    import experiment_config  # noqa: PLC0415

    return experiment_config


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


# --------------------------------------------------------------------------
# Box-size-banded Visual Genome, assembled from the full source
# --------------------------------------------------------------------------


def _band_categories(band: str) -> list[str]:
    """Pick this band's categories from the full-VG scan, stratified within it.

    Stratified on purpose: taking the N best-supported categories in a band
    would cluster them at one end of it (support correlates with size), so the
    "band" would silently be a point. Splitting the band into N slots by
    voted-area rank and taking the best-supported category in each keeps the
    band spanning its own range.
    """
    scan_path = pc.PILE / "vg_box_scale.json"
    if not scan_path.exists():
        raise SystemExit(f"missing {scan_path}; run scan_vg_boxes.py first")
    stats = json.loads(scan_path.read_text())["categories"]
    lo, hi = pc.BOX_BANDS[band]

    pool = [
        (s["voted_area"], name)
        for name, s in stats.items()
        if lo <= s["voted_area"] < hi
        and s["n_images"] >= pc.BAND_MIN_IMAGES
        and s["union_inflation"] <= pc.BAND_MAX_INFLATION
        and pc.is_object_category(name)
    ]
    if not pool:
        raise SystemExit(f"no categories qualify for band {band!r}")
    pool.sort()

    if len(pool) < pc.BAND_N_CATEGORIES:
        # Say so rather than quietly returning a shorter list: a band that
        # cannot fill its quota is a real limit on what it can support.
        log(
            f"  band {band}: ONLY {len(pool)} categories qualify "
            f"(wanted {pc.BAND_N_CATEGORIES}) -- band is supply-limited"
        )
    n = min(pc.BAND_N_CATEGORIES, len(pool))
    chosen: list[str] = []
    for i in range(n):
        slot = pool[i * len(pool) // n : max((i + 1) * len(pool) // n, i * len(pool) // n + 1)]
        best = max(slot, key=lambda t: stats[t[1]]["n_images"])
        chosen.append(best[1])
    log(f"  band {band}: {len(chosen)} categories from {len(pool)} candidates")
    return sorted(set(chosen))


def _load_vg_band(band: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with full-VG images carrying this band's categories."""
    import random  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    vg_root = pc.DEMO_CACHE / "visual_genome"
    objects_json = vg_root / "objects.json"
    if not objects_json.exists():
        raise SystemExit(f"missing {objects_json}")

    wanted = set(_band_categories(band))
    paths: dict[int, Path] = {}
    for d in (vg_root / "VG_100K", vg_root / "VG_100K_2"):
        for p in d.iterdir():
            if p.suffix.lower() == ".jpg":
                try:
                    paths[int(p.stem)] = p
                except ValueError:
                    continue

    with objects_json.open() as fh:
        records = json.load(fh)

    # Every image carrying at least one of the band's categories.
    hits: list[tuple[int, dict]] = []
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in paths:
            continue
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
        if by_name:
            hits.append((iid, dict(by_name)))

    rng = random.Random(0xB0FFED)  # deterministic sample, stable across rebuilds
    rng.shuffle(hits)
    hits = hits[: pc.BAND_MAX_IMAGES]
    log(f"  band {band}: {len(hits)} images carry a band category")

    for iid, by_name in hits:
        path = paths[iid]
        try:
            with Image.open(path) as im:
                W, H = im.size
            data = path.read_bytes()
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            continue
        if W <= 0 or H <= 0:
            continue
        regions = [
            {"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "label": name}
            for name, boxes in by_name.items()
            for b in boxes
        ]
        counts = {name: len(b) for name, b in by_name.items()}
        ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        medias[iid] = {
            "id": iid,
            "media_type": "image",
            "embedder": embedder_name,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {},
            "media_bytes": data,
            "media_string": None,
            "filename": path.name,
            "category": ordered[0],
            "categories": ordered,
            "regions": regions,
            "origin": {"importer": "vg_box_band", "params": {"band": band, "embedder": embedder_name}},
            "origin_name": str(path),
        }


def _vg_image_paths() -> dict[int, Path]:
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


def _vg_source() -> tuple[dict[int, Path], list, dict[int, tuple[int, int]]]:
    """``(image paths, objects.json records, image dims)`` for the whole VG source.

    Dims come from ``scan_vg_boxes.py``'s cache when it exists (it always does
    in practice -- the scan is what chooses the classes), and are read from the
    JPEG headers otherwise, which costs ~30 s.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    objects_json = pc.DEMO_CACHE / "visual_genome" / "objects.json"
    if not objects_json.exists():
        raise SystemExit(f"missing {objects_json}")

    paths = _vg_image_paths()
    cache = pc.PILE / "vg_image_dims.json"
    dims: dict[int, tuple[int, int]] = {}
    if cache.exists():
        raw = json.loads(cache.read_text())
        # Unreadable images are cached as null, so a complete cache has one
        # entry per file (see scan_vg_boxes._read_dims).
        if len(raw) >= len(paths):
            dims = {int(k): tuple(v) for k, v in raw.items() if v}  # type: ignore[misc]
    if not dims:
        log("  no dims cache; reading JPEG headers")

        def one(item):
            iid, path = item
            try:
                with Image.open(path) as im:
                    return iid, im.size
            except Exception:  # noqa: BLE001 - a corrupt file just drops out
                return iid, None

        with ThreadPoolExecutor(max_workers=16) as ex:
            for iid, size in ex.map(one, paths.items(), chunksize=256):
                if size:
                    dims[iid] = size

    with objects_json.open() as fh:
        records = json.load(fh)
    return paths, records, dims


def _load_corrections() -> dict[tuple[int, str], dict]:
    """``{(image_id, class): verdict}`` from the corrections file, if any.

    Verdicts, not corrections: a row exists for every reviewed ``(image, class)``
    pair whether or not the human disagreed, so review *coverage* is knowable.
    Without that, "no bus here" and "nobody looked" are the same absence, and
    every rate computed afterwards is biased by an unknown amount.

    Written by ``ingest_slate.py``; absent until the first review lands, which
    is why this returns empty rather than failing.
    """
    path = Path(os.environ.get("VTS_CORRECTIONS", pc.PILE / "corrections.json"))
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out: dict[tuple[int, str], dict] = {}
    for r in rows:
        out[(int(r["image_id"]), r["class"])] = r
    return out


def _load_vg_scale(medias: dict[int, dict], embedder_name: str) -> None:
    """One pickle holding every ``(class, band)`` cell of the scale study (#3156).

    The construction, in one paragraph: one image pool and one class list
    (:data:`pile_config.SCALE_CLASSES`); for a class *c* and band *B* an image is
    a **positive** when its compact union box for *c* falls in *B*, a
    **negative** when it holds no instance of any class in *C*, and **excluded**
    otherwise -- it holds *c* at some other size, so scoring it as a negative
    would penalise a detector for finding a real bus, which is what #3156 is
    about. Exclusion is carried per media as ``evaluable_categories`` and
    honoured by ``vtscore.eval.labels.evaluable_pool``.

    Cells are *designated* rather than inferred: exactly ``SCALE_N_POS``
    positives and one shared pool of ``SCALE_N_NEG`` negatives each. Every cell
    therefore has identical prevalence and identical negatives, so a
    small-vs-large difference is a paired contrast on one class rather than two
    datasets of different difficulty.

    **The labels are COCO's, and the pool is the half of VG that can carry
    them.** VG's own annotation is not exhaustive and measurably fails this
    construction -- see the note in the body and ``coco_anchor.py``.
    """
    from PIL import Image  # noqa: PLC0415

    import coco_anchor as ca  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES)
    bands = list(pc.BOX_BANDS)
    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in bands]

    paths = _vg_image_paths()
    _, records, dims = _vg_source()

    # --- labels: VG, repaired where a better reference exists ---------------
    #
    # VG's own annotation is not exhaustive: measured against COCO its recall
    # over C is 0.61, and 1.35% of the images it treats as negatives actually
    # hold the object -- ~54 hidden positives against 100 labelled ones per cell,
    # and 4.1% for `backpack` puts more real backpacks among the negatives than
    # among the positives (`coco_anchor.py`, issue #3156).
    #
    # 48% of VG's images ARE COCO images, and COCO annotates C exhaustively, so
    # on that half the repair is free and total. The other half has no reference
    # and is what the human slates are for (`make_audit_slate.py`); its verdicts
    # arrive here through the same corrections file. The pool stays the whole of
    # VG either way -- restricting it to the COCO half would make this a COCO
    # subset with extra steps, losing VG's non-COCO diversity for nothing.
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = _load_corrections()
    log(f"  {len(coco_of)} VG images carry a coco_id; {len(corrections)} human verdicts on file")

    # image -> {class: [boxes]}, and whether the image's labels are exhaustive.
    labels: dict[int, dict[str, list[list[float]]]] = {}
    exhaustive: set[int] = set()
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in paths or dims.get(iid) is None:
            continue
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
        labels[iid] = dict(by_name)

    # The pixel space each image's boxes live in. VG ships DOWNSCALED copies of
    # the COCO originals -- 500 px wide against COCO's 640, on 95% of the
    # overlap -- so a COCO box normalised by the VG file's dimensions lands in
    # the wrong place and with the wrong extent. Normalise every box by the
    # dimensions of the image its coordinates were measured on.
    box_dims: dict[int, tuple[int, int]] = dict(dims)
    n_anchored = 0
    n_reframed = 0
    for iid in labels:
        cid = coco_of.get(iid)
        ref = truth.get(cid) if cid is not None else None
        if ref is None:
            continue
        wh = ca.COCO_DIMS.get(cid)
        if wh is None:
            continue
        # A normalised box only transfers between two copies of an image if they
        # frame the same thing. 49 of the 51,497 overlaps disagree on aspect
        # ratio -- some are transposed (VG 500x375 against COCO 375x500), i.e. a
        # rotated or re-cropped copy -- and there COCO's box does not describe
        # VG's pixels at all. Those keep VG's own labels and stay unanchored
        # rather than importing a box that points at the wrong part of a
        # different framing.
        vw, vh = dims[iid]
        if abs((vw / vh) - (wh[0] / wh[1])) / (wh[0] / wh[1]) > pc.MAX_ASPECT_DRIFT:
            n_reframed += 1
            continue
        box_dims[iid] = wh
        # COCO's annotation REPLACES VG's for this image rather than merging
        # with it: the two disagree in both directions, and only one of them is
        # exhaustive. Keeping VG's extra boxes would reintroduce exactly the
        # unverifiable labels this is repairing.
        labels[iid] = {name: bs for name, bs in ref.items() if name in wanted}
        exhaustive.add(iid)
        n_anchored += 1

    # A pair a reviewer ruled "present" without drawing a box cannot be banded:
    # a band is a claim about size, and no size was measured. It leaves every
    # cell of that class instead -- neither a positive nor a negative -- which
    # is precisely what the third value is for.
    unbanded: set[tuple[int, str]] = set()
    for (iid, name), verdict in corrections.items():
        if iid not in labels:
            continue
        if verdict.get("present"):
            boxes = verdict.get("boxes") or []
            if boxes:
                labels[iid][name] = boxes
            else:
                labels[iid].pop(name, None)
                unbanded.add((iid, name))
        else:
            labels[iid].pop(name, None)
        exhaustive.add(iid)  # someone looked at this pair

    log(
        f"  labels: {len(labels)} VG images, {n_anchored} repaired from COCO, "
        f"{len(exhaustive)} with a verified pair, {n_reframed} skipped as re-framed copies"
    )

    # --- candidates ---------------------------------------------------------
    supply: dict[str, dict[str, list[int]]] = {c: {b: [] for b in bands} for c in pc.SCALE_CLASSES}
    boxes_for: dict[tuple[int, str], list[list[float]]] = {}
    clean: list[int] = []

    for iid, by_name in labels.items():
        W, H = box_dims[iid]
        area = float(W * H)
        if not by_name:
            # Only a true negative for every class in C may join the shared pool.
            if not any((iid, c) in unbanded for c in pc.SCALE_CLASSES):
                clean.append(iid)
            continue
        for name, bs in by_name.items():
            ux0 = min(b[0] for b in bs)
            uy0 = min(b[1] for b in bs)
            ux1 = max(b[2] for b in bs)
            uy1 = max(b[3] for b in bs)
            union = max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0) / area
            largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in bs) / area
            # Scattered instances in *this* image: the union box describes the
            # scatter rather than the object, so the image is excluded from
            # every band of this class rather than banded by a box no user
            # would drag.
            if union > largest * pc.BAND_MAX_INFLATION:
                continue
            for band, (lo, hi) in pc.BOX_BANDS.items():
                if lo <= union < hi:
                    supply[name][band].append(iid)
                    boxes_for[(iid, pc.scale_cell(name, band))] = bs
                    break

    # Selection must be stable under a changing candidate list, not merely
    # deterministic. `rng.sample` is deterministic given the same list, but any
    # edit to the pool -- a label fix, an image excluded as a re-framed copy --
    # reshuffles the entire draw, and a rebuild then silently retires images a
    # human already reviewed (49 of 360 in one such rebuild). Ranking each
    # candidate by a hash of (cell, image_id) instead means adding or removing
    # one image changes only that image's membership.
    def _rank(cell: str, iid: int) -> str:
        return hashlib.sha1(f"{cell}:{iid}".encode()).hexdigest()  # noqa: S324 - not security

    # A roster pins the membership a review was actually carried out against.
    # Without it, switching selection rules retires images a human has already
    # judged -- the hash draw and the earlier random draw share only ~228 of
    # 3,900 negatives, so the entire negative review would have been orphaned.
    # Entries that are no longer eligible (a correction moved or removed them)
    # drop out and the shortfall is backfilled by rank, so the roster adapts
    # without ever reshuffling what it can keep.
    roster = {}
    if pc.ROSTER.exists():
        roster = json.loads(pc.ROSTER.read_text())
        log(f"  roster: {pc.ROSTER.name} pins {len(roster.get('cells', {}))} cells")

    chosen: dict[str, list[int]] = {}
    for c in pc.SCALE_CLASSES:
        for band in bands:
            pool = sorted(supply[c][band])
            cell = pc.scale_cell(c, band)
            if len(pool) < pc.SCALE_N_POS:
                # Say so rather than quietly building a smaller cell: unequal
                # prevalence between bands is the defect this construction
                # exists to remove.
                log(f"  UNDER-SUPPLIED {cell}: {len(pool)} positives (wanted {pc.SCALE_N_POS})")
            eligible = set(pool)
            kept = [i for i in roster.get("cells", {}).get(cell, []) if i in eligible]
            if len(kept) < pc.SCALE_N_POS:
                spare = sorted(eligible - set(kept), key=lambda i: _rank(cell, i))
                kept += spare[: pc.SCALE_N_POS - len(kept)]
            chosen[cell] = kept[: pc.SCALE_N_POS]

    clean.sort()
    # Draw spares beyond the designated pool. A human verdict can retire a
    # contaminated negative later, and re-designating from spares costs a
    # relabel rather than a re-embed of every cell.
    want_neg = pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE
    clean_set = set(clean)
    drawn = [i for i in roster.get("negatives", []) + roster.get("spares", []) if i in clean_set]
    if len(drawn) < want_neg:
        extra = sorted(clean_set - set(drawn), key=lambda i: _rank("__negatives__", i))
        drawn += extra[: want_neg - len(drawn)]
    drawn = drawn[:want_neg]
    negatives, spares = drawn[: pc.SCALE_N_NEG], drawn[pc.SCALE_N_NEG :]
    neg_set = set(negatives)
    pc.ROSTER.write_text(json.dumps({"cells": chosen, "negatives": negatives, "spares": spares}, indent=1) + "\n")
    log(
        f"  {sum(len(v) for v in chosen.values())} positives over {len(cells)} cells, "
        f"{len(negatives)} shared negatives + {len(spares)} spares (from {len(clean)} clean images)"
    )

    # media id -> the cells it is a positive for. Negatives get every cell.
    positive_in: dict[int, list[str]] = defaultdict(list)
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in[iid].append(cell)

    for iid in sorted(set(positive_in) | set(negatives) | set(spares)):
        path = paths[iid]
        try:
            with Image.open(path) as im:
                vw, vh = im.size
            data = path.read_bytes()
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            continue
        if vw <= 0 or vh <= 0:
            continue
        # Normalised region boxes are resolution-independent, so they must be
        # divided by the size of the image the coordinates came from -- which is
        # the COCO original for a repaired image, not the VG copy carrying the
        # pixels.
        W, H = box_dims[iid]
        cats = sorted(positive_in.get(iid, []))
        regions = [
            {"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "label": cell}
            for cell in cats
            for b in boxes_for.get((iid, cell), [])
        ]
        medias[iid] = {
            "id": iid,
            "media_type": "image",
            "embedder": embedder_name,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {},
            "media_bytes": data,
            "media_string": None,
            "filename": path.name,
            "category": cats[0] if cats else "",
            "categories": cats,
            # A designated cell membership, not a closed world: a positive is
            # scorable only in the cells it was drawn for, and the shared
            # negatives are scorable everywhere.
            "evaluable_categories": cats if cats else (list(cells) if iid in neg_set else []),
            # Whether this image's labels rest on an exhaustive reference (COCO,
            # or a human who looked). False means VG's silence is the only
            # evidence of absence -- which is what the review slates target.
            "labels_exhaustive": iid in exhaustive,
            "regions": regions,
            "origin": {"importer": "vg_scale", "params": {"embedder": embedder_name, "labels": "coco"}},
            "origin_name": str(path),
        }


def _load_demo(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    from vtscore.datasets.loader_demo import load_demo_dataset  # noqa: PLC0415

    load_demo_dataset(dataset, medias, embedder_name=embedder_name)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


@contextmanager
def _embed_batch_size(embedder: str):
    """Apply this embedder's batch size for the duration of the embed pass.

    The app reads ``VTSEARCH_EMBED_BATCH_SIZE`` per bulk call, so one build
    process can run each embedder at its own size rather than every model at
    the shipped default of 32. An explicit env var wins: someone who set one
    is tuning for the card in front of them, and the table cannot know that.
    """
    want = pc.embed_batch_size(embedder)
    if want is None or os.environ.get("VTSEARCH_EMBED_BATCH_SIZE", "").strip():
        yield
        return
    os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(want)
    log(f"  embed batch size {want}")
    try:
        yield
    finally:
        os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE", None)


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
    elif kind == "vg_band":
        _load_vg_band(pc.DATASETS[dataset]["band"], medias, embedder)
    elif kind == "vg_scale":
        _load_vg_scale(medias, embedder)
    else:
        pc.require_demo_source(dataset)
        _load_demo(dataset, medias, embedder)
    log(f"  loaded {len(medias)} medias in {time.time() - t0:.0f}s")

    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    t1 = time.time()
    with _embed_batch_size(embedder):
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

    # A banded cell's NAME asserts the size of its boxes, so the stored box has
    # to agree with it. That is what catches a coordinate-space mistake: VG
    # ships 500 px copies of COCO's 640 px originals, and normalising a COCO box
    # by the VG file's dimensions leaves every box shifted and mis-scaled while
    # every other check still passes -- the medias load, the vectors are there,
    # the patch grids are there, and the boxes are quietly pointing at the wrong
    # pixels. Recomputing the band from the box is cheap and would have caught
    # it at build time instead of via a human noticing a box drawn on snow.
    for ds, emb in pc.cells():
        if pc.DATASETS.get(ds, {}).get("kind") != "vg_scale":
            continue
        path = pc.cell_path(ds, emb)
        if not path.exists():
            continue
        medias = io.load_medias(path)
        bad = 0
        checked = 0
        for m in medias.values():
            for cell in m.get("categories") or []:
                boxes = [r["box"] for r in (m.get("regions") or []) if r.get("label") == cell]
                if not boxes:
                    continue
                area = (max(b[2] for b in boxes) - min(b[0] for b in boxes)) * (
                    max(b[3] for b in boxes) - min(b[1] for b in boxes)
                )
                lo, hi = pc.BOX_BANDS[cell.rsplit("@", 1)[1]]
                checked += 1
                if not (lo <= area < hi):
                    bad += 1
        if checked and bad:
            problems.append(
                f"{ds} x {emb}: {bad}/{checked} region boxes fall outside the band their cell "
                f"name claims -- boxes and bands were measured in different pixel spaces"
            )
        break  # one embedder is enough; the boxes are identical across cells

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


def report_bands() -> int:
    """Report voted-box scale-band populations for each boxed dataset.

    The bands are anchored to the patch embedder's geometry: ``sub_patch`` is
    "smaller than one DINOv3 patch", i.e. below what the patch grid can resolve
    at all. That anchoring is the band's whole meaning, so a thin ``sub_patch``
    is a fact about the data, not a threshold to tune — widening the edge would
    inflate the count with objects that *are* resolvable.

    Reads the smallest available cell for each dataset: scale stats need only
    ``regions``, which every cell carries, so there is no reason to page in the
    multi-GB patch cell.
    """
    io = _cells_io()
    cfg = _experiment_config()
    from vtscore.eval.labels import category_scale_stats  # noqa: PLC0415

    boxed = [ds for ds, info in pc.DATASETS.items() if info.get("boxed")]
    if not boxed:
        log("no boxed datasets in the pile; nothing to stratify")
        return 0

    for ds in boxed:
        present = [(pc.cell_path(ds, e).stat().st_size, e) for e in pc.EMBEDDERS if pc.cell_path(ds, e).exists()]
        if not present:
            log(f"{ds}: no cells present")
            continue
        _, emb = min(present)
        medias = io.load_medias(pc.cell_path(ds, emb))

        counts: dict[str, int] = defaultdict(int)
        for m in medias.values():
            for c in m.get("categories") or [m.get("category")]:
                if c:
                    counts[c] += 1

        selected, report = cfg.select_categories_by_scale(medias, dict(counts))
        log("")
        log(f"=== {ds}: {len(medias)} medias, {len(counts)} categories (via {emb}) ===")
        dropped = report.get("dropped_above_max_voted_area") or []
        log(f"  dropped above max_voted_area={report.get('max_voted_area')}: {len(dropped)}")
        for name, info in (report.get("bands") or {}).items():
            lo, hi = info["range"]
            flag = "  ** UNDER-POPULATED **" if info["under_populated"] else ""
            log(
                f"  {name:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): "
                f"{len(info['selected'])}/{info['target']} of {info['n_candidates']} candidates{flag}"
            )
            log(f"      {info['selected']}")

        # When a band is starved, say whether the min-count filter is even the
        # binding constraint. Measured on the first run it was not: the
        # sub_patch pool held 5 categories (VG) and 1 (COCO) at every
        # min_count from 5 to 30, so lowering it recovers nothing.
        starved = [n for n, i in (report.get("bands") or {}).items() if i["under_populated"]]
        if starved:
            stats = {c: s for c in counts if (s := category_scale_stats(medias, c)) is not None}
            for name in starved:
                lo, hi = report["bands"][name]["range"]
                pools = {
                    mc: sum(1 for c, s in stats.items() if counts[c] >= mc and lo <= s["voted_area"] < hi)
                    for mc in (5, 10, 20, 30)
                }
                spread = "same at every min_count" if len(set(pools.values())) == 1 else str(pools)
                log(f"  {name}: candidate pool by min category count -> {spread}")
        log(f"  -> selected {len(selected)} categories")
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
    ap.add_argument("--bands", action="store_true", help="report voted-box scale bands for boxed datasets")
    ap.add_argument("--manifest", action="store_true", help="(re)write the manifest and exit")
    args = ap.parse_args()

    pc.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    assert_vtscore_is_this_checkout()

    if args.list:
        list_cells()
        return 0
    if args.verify:
        return verify()
    if args.bands:
        return report_bands()
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
