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
    python build_pile.py --provenance                # which device built each cell
    python build_pile.py --backfill-provenance       # fingerprint the pre-#3160 cells

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


def _vg_source() -> tuple[dict[int, Path], list, dict[int, tuple[int, int]]]:
    """``(image paths, objects.json records, image dims)`` for the whole VG source.

    Dims come from ``scan_vg_boxes.py``'s cache when it exists (it always does
    in practice -- the scan is what chooses the classes), and are read from the
    JPEG headers otherwise, which costs ~30 s.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    vg_root = pc.DEMO_CACHE / "visual_genome"
    objects_json = vg_root / "objects.json"
    if not objects_json.exists():
        raise SystemExit(f"missing {objects_json}")

    paths: dict[int, Path] = {}
    for d in (vg_root / "VG_100K", vg_root / "VG_100K_2"):
        for p in d.iterdir():
            if p.suffix.lower() == ".jpg":
                try:
                    paths[int(p.stem)] = p
                except ValueError:
                    continue

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
    """
    import random  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES)
    bands = list(pc.BOX_BANDS)
    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in bands]

    paths, records, dims = _vg_source()
    log(f"  scanning {len(records)} VG records for {len(wanted)} classes")

    # class -> band -> [image_id], plus the boxes to stamp and the clean pool.
    supply: dict[str, dict[str, list[int]]] = {c: {b: [] for b in bands} for c in pc.SCALE_CLASSES}
    boxes_for: dict[tuple[int, str], list[list[float]]] = {}
    clean: list[int] = []

    for rec in records:
        iid = int(rec["image_id"])
        wh = dims.get(iid)
        if iid not in paths or wh is None:
            continue
        W, H = wh
        if W <= 0 or H <= 0:
            continue
        area = float(W * H)
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
        if not by_name:
            # Holds none of C at any size: a sound negative for every cell.
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

    rng = random.Random(0x5CA1E)  # deterministic sample, stable across rebuilds
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
            chosen[cell] = rng.sample(pool, min(pc.SCALE_N_POS, len(pool)))

    clean.sort()
    negatives = rng.sample(clean, min(pc.SCALE_N_NEG, len(clean)))
    log(
        f"  {sum(len(v) for v in chosen.values())} positives over {len(cells)} cells, "
        f"{len(negatives)} shared negatives (from {len(clean)} clean images)"
    )

    # media id -> the cells it is a positive for. Negatives get every cell.
    positive_in: dict[int, list[str]] = defaultdict(list)
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in[iid].append(cell)

    for iid in sorted(set(positive_in) | set(negatives)):
        path = paths[iid]
        try:
            with Image.open(path) as im:
                W, H = im.size
            data = path.read_bytes()
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            continue
        if W <= 0 or H <= 0:
            continue
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
            "evaluable_categories": cats if cats else list(cells),
            "regions": regions,
            "origin": {"importer": "vg_scale", "params": {"embedder": embedder_name}},
            "origin_name": str(path),
        }


def _load_vg_scale_any(medias: dict[int, dict], embedder_name: str) -> None:
    """``vg_scale`` with the box-size band collapsed away (#3115).

    #3156 built ``vg_scale`` to ask a question about *scale*, so its cells are
    keyed ``class@band`` and each holds exactly 100 positives.  A study that does
    not care how big the box is wants the same verified images with the band
    dropped: 12 classes, **300 positives each**, one shared negative pool, and
    identical prevalence everywhere.  That is what the calibration studies
    actually need, and what ``visual_genome_m`` conspicuously is not - its
    selected categories run from 25 positives (``banana``) to 1645
    (``building``), and the thin ones produce cells with *no trainable step at
    all* (``ball``, 51 positives, wrote a header and zero rows).

    Derived from the built ``vg_scale`` pickle rather than re-scanned from the
    VG source, so it is **the same images, the same boxes and the same
    hand-checked label corrections** with one string operation applied - and it
    costs no image decode and no embedding pass.

    The whole build is a relabel, ``class@band -> class``, applied to
    ``category`` / ``categories`` / ``evaluable_categories`` and to each region's
    ``label``.  What matters is what that does *not* change, because the
    exclusion semantics are the point of #3156 and the easy version of this
    build destroys them:

    * The **300 medias evaluable on nothing** stay evaluable on nothing.  They
      hold one of the 12 classes at a size outside every band, so scoring them
      as negatives would penalise a detector for finding a real bus - exactly
      what #3156 exists to prevent.  A naive "positive if in any band, negative
      otherwise" rule turns all 300 into negatives.
    * A media positive for one class stays evaluable **only** for that class.
      VG's labels are not exhaustive (``labels_exhaustive: False``), so an image
      of a dog is not evidence of the absence of a clock.
    * The 3900-image clean pool - images holding no instance of any of the 12 -
      stays evaluable for all of them.

    Result per class: 300 positives against 3900 shared negatives, i.e. a 4200
    -image evaluable pool at **7.1 % prevalence, identical for all twelve**.
    """
    src = pc.cell_path("vg_scale", embedder_name)
    if not src.exists():
        raise SystemExit(f"vg_scale_any is derived from {src.name}, which does not exist yet - build vg_scale first")

    def _base(label: str) -> str:
        return label.split("@", 1)[0]

    def _collapse(labels) -> list[str]:
        """Base classes, order-preserving and deduped."""
        return list(dict.fromkeys(_base(c) for c in (labels or [])))

    loaded = _cells_io().load_medias(src)
    log(f"  derived from {src.name}: {len(loaded)} medias")
    for mid, media in loaded.items():
        d = dict(media)
        d["categories"] = _collapse(d.get("categories"))
        # `evaluable_categories` is collapsed rather than rebuilt: an empty list
        # must survive as an empty list (see the docstring), and rebuilding it
        # from the class roster is precisely the bug that would not.
        d["evaluable_categories"] = _collapse(d.get("evaluable_categories"))
        if d.get("category"):
            d["category"] = _base(d["category"])
        d["regions"] = [{**r, "label": _base(r["label"])} for r in (d.get("regions") or [])]
        d["origin"] = {"importer": "vg_scale_any", "params": {"embedder": embedder_name, "derived_from": src.name}}
        medias[mid] = d

    n_pos = sum(1 for d in medias.values() if d["categories"])
    n_clean = sum(1 for d in medias.values() if not d["categories"] and d["evaluable_categories"])
    n_excluded = sum(1 for d in medias.values() if not d["categories"] and not d["evaluable_categories"])
    log(f"  {n_pos} positives, {n_clean} shared negatives, {n_excluded} excluded-everywhere (must be nonzero)")
    if not n_excluded:
        raise SystemExit("vg_scale_any: no excluded-everywhere medias survived - the exclusion semantics were lost")


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


# --------------------------------------------------------------------------
# Provenance: which machine produced this cell (#3160)
# --------------------------------------------------------------------------


def _device_record() -> dict:
    """Everything about the machine that a later reader needs to compare cells.

    ``gres/gpu:v100`` is a *type*, and #3143 measured that a type is not a
    device: two nodes both answering to it produced ``siglip2_l`` vectors 1.5e-04
    apart, while three other devices agreed to ~1e-12. Nothing in ``scontrol`` or
    ``--gres`` distinguishes the parts, so the only way a rebuild can be told
    apart from the cell it replaces is if the build **writes down** what it ran
    on. That is what this is; it does not make the arithmetic reproducible, it
    makes the difference visible.
    """
    import torch  # noqa: PLC0415

    from vtscore.config import EMBED_PRECISION, embed_precision  # noqa: PLC0415

    rec: dict = {
        "hostname": os.uname().nodename,
        # The host, not the card, turned out to be the axis (#3160): PyTorch's
        # CPU kernel dispatch decides how the 384px resize rounds, and an AVX2
        # host disagrees with an AVX-512 one on 12.3% of pixels by one 8-bit
        # level -- which is the whole of the 1.5e-04 that #3143 attributed to the
        # GPU. Pinning the dispatch removes it; recording it explains a cell that
        # was built before the pin.
        "cpu": _cpu_model(),
        # What the process RESOLVED, not what was asked for. `ATEN_CPU_CAPABILITY`
        # is advisory: an unsupported or misspelled value is ignored in silence,
        # so echoing the request would record a pin that never took. Both are
        # written; a reader can see a request that did not land.
        "cpu_capability": _cpu_capability(),
        "aten_cpu_capability_requested": os.environ.get("ATEN_CPU_CAPABILITY"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_gres": os.environ.get("SLURM_JOB_GRES") or os.environ.get("SBATCH_GRES"),
        "precision_requested": EMBED_PRECISION,
        "precision_resolved": embed_precision(),
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        # The node is not the only unrecorded axis. `requirements/image-embedders.txt`
        # pins `transformers>=4.49`, and v5 renamed the image processors: the plain
        # name is now the torchvision implementation and the PIL one moved to a
        # `Pil` suffix. So two hosts resolving different versions preprocess the
        # same image differently -- measured at 7.8e-3 max abs in pixels between
        # the two paths, well above the 1.5e-04 device effect this record was
        # written for. Recording the version and the class that actually loaded
        # costs nothing and makes that axis visible too.
        "transformers": _transformers_version(),
        "commit": _git_commit(),
    }
    if not torch.cuda.is_available():
        rec["gpu_name"] = None
        rec["note"] = "no CUDA device; embedded on CPU"
        return rec
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    rec.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": f"sm_{major}{minor}",
            # SM count was #3143's leading hypothesis for the drift (different
            # tiling, different accumulation order). It is not the cause -- the
            # two V100 parts have 80 SMs each and produce bit-identical GEMMs --
            # but it stays in the record because it is the cheapest way to tell
            # two cards apart that share a name.
            "multi_processor_count": props.multi_processor_count,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    )
    return rec


def _cpu_capability() -> str | None:
    """The CPU kernel ISA torch is actually dispatching to (``AVX512``/``AVX2``/...)."""
    import torch  # noqa: PLC0415

    getter = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
    return str(getter()) if getter else None


def _cpu_model() -> str | None:
    """The host CPU model string, or None where /proc/cpuinfo is not readable."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _transformers_version() -> str | None:
    try:
        import transformers  # noqa: PLC0415
    except ImportError:
        return None
    return getattr(transformers, "__version__", None)


def _processor_record(embedder: str) -> dict:
    """The preprocessing classes this embedder actually resolved to.

    Best effort: an embedder with no HF processor (or one that failed to load)
    records nulls rather than sinking a build that has already produced a cell.
    """
    try:
        from vtscore.media import get_embedder  # noqa: PLC0415

        emb = get_embedder(embedder)
        proc = getattr(emb, "_processor", None)
        image_proc = getattr(proc, "image_processor", None)
        return {
            "processor_class": type(proc).__name__ if proc is not None else None,
            "image_processor_class": type(image_proc).__name__ if image_proc is not None else None,
        }
    except Exception as exc:  # noqa: BLE001 -- provenance must never fail a build
        return {"processor_class": None, "image_processor_class": None, "error": repr(exc)[:120]}


def _git_commit() -> str | None:
    """The commit of the checkout that is about to embed, or None outside git."""
    import subprocess  # noqa: PLC0415, S404 -- fixed argv, no shell

    repo = Path(os.environ.get("VTS_REPO") or Path(__file__).resolve().parents[3])
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def cell_fingerprint(dataset: str, embedder: str, medias: dict | None = None) -> dict:
    """A hash of the cell's vectors, in a fixed media-id order.

    The point of the hash is that it survives the cell it describes: a rebuild
    can be compared against it without keeping the old 900 MB pickle, which is
    exactly the check a purge-and-rebuild needs and cannot otherwise make.

    ``medias`` lets a fresh build hand over what it already has; the patch cells
    are 3.5 GB on disk, and re-reading one purely to hash it would add minutes
    and a second copy in RAM to every build.
    """
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    if medias is None:
        medias = _cells_io().load_medias(pc.cell_path(dataset, embedder))
    ids = sorted(medias)
    vecs = [media_embedding(medias[i]) for i in ids]
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in vecs if v is not None])
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return {
        "n_vectors": int(arr.shape[0]),
        "dim": int(arr.shape[1]) if arr.ndim > 1 else None,
        "vectors_sha256": digest,
        "id_range": [int(ids[0]), int(ids[-1])] if ids else None,
    }


def write_provenance(dataset: str, embedder: str, summary: dict, medias: dict | None = None) -> Path:
    """Write the per-cell provenance sidecar."""
    record = {
        "dataset": dataset,
        "embedder": embedder,
        "cell": pc.cell_path(dataset, embedder).name,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": _device_record(),
        "preprocessing": _processor_record(embedder),
        "cell_summary": {k: v for k, v in summary.items() if k != "status"},
        "fingerprint": cell_fingerprint(dataset, embedder, medias),
    }
    path = pc.provenance_path(dataset, embedder)
    path.write_text(json.dumps(record, indent=2) + "\n")
    dev = record["device"]
    log(f"  provenance: {dev.get('gpu_name')} on {dev.get('hostname')} -> {path.name}")
    return path


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
    elif kind == "vg_scale_any":
        _load_vg_scale_any(medias, embedder)
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
    summary = {
        "dataset": dataset,
        "embedder": embedder,
        "status": "built",
        "n_medias": len(medias),
        "n_patch_grids": n_patch,
        "megabytes": round(nbytes / 1e6, 1),
        "embed_seconds": round(embed_s, 1),
        "wall_seconds": round(total_s, 1),
    }
    write_provenance(dataset, embedder, summary, medias)
    return summary


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


def _sacct_build_nodes() -> dict[str, str]:
    """dataset -> node, recovered from SLURM's accounting of the ``pile-*`` jobs.

    Cells built before the sidecar existed are not anonymous after all: the build
    ran as ``pile-<dataset>`` and ``sacct`` still knows which node took it. This
    is recorded as ``hostname_recovered``, never as ``hostname`` -- it is an
    inference from a job name, and one ambiguous dataset (two completed jobs) is
    left out rather than guessed. It matters because the node determines the CPU,
    and the CPU determines how the 384px resize rounds (#3160).
    """
    import subprocess  # noqa: PLC0415, S404 -- fixed argv, no shell

    try:
        out = subprocess.run(  # noqa: S603
            ["sacct", "-X", "-S", "2026-01-01", "-n", "-P", "--format=JobName,NodeList,State"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    seen: dict[str, set[str]] = defaultdict(set)
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or not parts[0].startswith("pile-") or parts[2] != "COMPLETED":
            continue
        seen[parts[0][len("pile-") :]].add(parts[1])
    return {ds: next(iter(nodes)) for ds, nodes in seen.items() if len(nodes) == 1}


def provenance_report(backfill: bool = False) -> int:
    """Show which device built each cell -- and, with ``--backfill-provenance``,
    stamp what is still knowable for the cells built before this existed.

    A backfilled sidecar deliberately records ``gpu_name: null``: the node a 2026
    job ran on is not recoverable from the pickle, and writing a guess would be
    worse than writing nothing. What it *can* record is the fingerprint, and that
    is the half that matters for a rebuild -- it turns "did the rebuild reproduce
    the cell?" from an unanswerable question into a hash comparison.
    """
    rows, missing, devices = [], [], defaultdict(list)
    recovered = _sacct_build_nodes() if backfill else {}
    for ds, emb in pc.cells():
        cell = pc.cell_path(ds, emb)
        if not cell.exists():
            continue
        path = pc.provenance_path(ds, emb)
        if not path.exists():
            if backfill:
                stat = cell.stat()
                record = {
                    "dataset": ds,
                    "embedder": emb,
                    "cell": cell.name,
                    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
                    "backfilled": True,
                    "device": {
                        "gpu_name": None,
                        "hostname_recovered": recovered.get(ds),
                        "recovered_from": "sacct pile-<dataset> job" if recovered.get(ds) else None,
                        "note": "unknown: cell predates per-cell provenance (#3160)",
                    },
                    "cell_summary": {"megabytes": round(stat.st_size / 1e6, 1)},
                    "fingerprint": cell_fingerprint(ds, emb),
                }
                path.write_text(json.dumps(record, indent=2) + "\n")
                log(f"backfilled {path.name} ({record['fingerprint']['vectors_sha256'][:12]})")
            else:
                missing.append(f"{ds} x {emb}")
                continue
        rec = json.loads(path.read_text())
        dev = rec.get("device", {})
        if backfill and not dev.get("hostname") and not dev.get("hostname_recovered") and recovered.get(ds):
            dev["hostname_recovered"] = recovered[ds]
            dev["recovered_from"] = "sacct pile-<dataset> job"
            rec["device"] = dev
            path.write_text(json.dumps(rec, indent=2) + "\n")
            log(f"recovered build node for {ds} x {emb}: {recovered[ds]}")
        rows.append(
            (
                ds,
                emb,
                dev.get("gpu_name") or "unknown",
                dev.get("hostname") or (f"{dev['hostname_recovered']}?" if dev.get("hostname_recovered") else "-"),
                str(dev.get("cpu_capability") or "-"),
                (dev.get("commit") or "-")[:9],
                rec.get("fingerprint", {}).get("vectors_sha256", "")[:12],
            )
        )
        devices[(dev.get("gpu_name") or "unknown", dev.get("cpu_capability"))].append(f"{ds}x{emb}")

    log(f"{'dataset':<18} {'embedder':<14} {'device':<26} {'node':<10} {'dispatch':<9} {'commit':<10} vectors")
    for row in sorted(rows):
        log("{:<18} {:<14} {:<26} {:<10} {:<9} {:<10} {}".format(*row))
    if missing:
        log(f"\n{len(missing)} cell(s) with NO provenance (run --backfill-provenance): {', '.join(missing)}")
    if len(devices) > 1:
        log(f"\nthis pile MIXES {len(devices)} build environments. The measured cost of mixing")
        log("hosts is 1.5e-04 median 1-cos on siglip2_l when CPU dispatch is unpinned (#3160):")
        for (name, cap), cells in sorted(devices.items(), key=lambda kv: str(kv[0])):
            log(f"  {str(name):<26} dispatch={cap or 'unrecorded':<10} {len(cells)} cell(s)")
    return 0


def _manifest_provenance(dataset: str, embedder: str) -> dict:
    """The provenance fields the manifest carries per cell, or nulls if unknown."""
    path = pc.provenance_path(dataset, embedder)
    if not path.exists():
        return {"gpu_name": None, "built_by": None, "commit": None, "vectors_sha256": None}
    rec = json.loads(path.read_text())
    dev = rec.get("device", {})
    return {
        "gpu_name": dev.get("gpu_name"),
        "built_by": dev.get("hostname"),
        "commit": dev.get("commit"),
        "vectors_sha256": rec.get("fingerprint", {}).get("vectors_sha256"),
    }


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
                # Which machine built it (#3160). None for cells that predate the
                # sidecar; a null here is a fact about the pile, not a gap to hide.
                **_manifest_provenance(ds, emb),
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
    ap.add_argument("--provenance", action="store_true", help="show which device built each cell")
    ap.add_argument(
        "--backfill-provenance",
        action="store_true",
        help="stamp a sidecar (fingerprint only, device unknown) on cells built before #3160",
    )
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
    if args.provenance or args.backfill_provenance:
        return provenance_report(backfill=args.backfill_provenance)

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
