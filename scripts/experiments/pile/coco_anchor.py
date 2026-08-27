"""Measure (and repair) VG's annotation noise against COCO, with no human review.

Roughly half of Visual Genome's images *are* COCO images -- VG records the
source id as ``coco_id`` in ``image_data.json`` -- and COCO is **exhaustively**
annotated over its 80 classes. Every class in :data:`pile_config.SCALE_CLASSES`
is one of those 80, deliberately (issue #3156). So for the COCO-sourced half of
the pool, ground truth already exists and both of the things the correction pass
needs come for free:

* **A measurement.** How often does VG omit an object COCO annotates? That is
  the noise rate the report has to state rather than assume, and it is the
  number that says how much manual review the rest is worth.
* **A repair.** On those images the COCO boxes *are* the correction, in exactly
  the ``(image_id, class, box)`` shape the merge expects.

What it cannot do is clear the other half: VG's non-COCO images (Flickr) have no
exhaustive reference, and neither does any class outside COCO's 80. Those are
what the human slates are for -- and this scan is what decides how big they need
to be.

**Both directions are real.** ``vg_missing`` (COCO annotates it, VG does not) is
the bug #3156 opened with: a bus front and centre on an image annotated
``car, clouds``. ``coco_missing`` is not simply its mirror -- VG's free-text
vocabulary is finer and its boxes are not exhaustive per class, so a VG-only
``bird`` may be a real bird COCO skipped, a duplicate name for something else,
or a genuine VG error. It is reported, not silently trusted.

Usage::

    python coco_anchor.py --fetch          # download what is missing, then scan
    python coco_anchor.py                  # scan (sources already staged)
    python coco_anchor.py --out anchor.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()

#: VG's per-image metadata, which is what carries ``coco_id``. Stanford's own
#: copy 403s; this UW mirror is the one that still serves it.
IMAGE_DATA_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

VG_ROOT = pc.DEMO_CACHE / "visual_genome"

#: ``{coco_image_id: (width, height)}``, filled by :func:`coco_truth`.
COCO_DIMS: dict[int, tuple[int, int]] = {}


def log(msg: str) -> None:
    print(f"[anchor] {msg}", flush=True)


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        log(f"have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as fh:  # noqa: S310 - fixed https/http URLs above
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    log(f"  -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def ensure_sources(anchor: Path, fetch: bool) -> tuple[Path, list[Path]]:
    """``(image_data.json, [instances_*.json])``, downloading them if asked."""
    image_data = anchor / "image_data.json"
    if not image_data.exists():
        if not fetch:
            raise SystemExit(f"missing {image_data}; re-run with --fetch")
        zpath = anchor / "image_data.json.zip"
        _fetch(IMAGE_DATA_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith("image_data.json"))
            image_data.write_bytes(z.read(member))
        log(f"extracted {image_data.name}")

    # val2017 is already staged under COCO_ROOT; train2017 is where most of VG's
    # COCO-sourced images live, so it is the half that matters here.
    instances = []
    staged_val = pc.COCO_ANNOTATIONS.parent.parent / "annotations" / "instances_val2017.json"
    if staged_val.exists():
        instances.append(staged_val)
    train = anchor / "instances_train2017.json"
    if not train.exists():
        if not fetch:
            raise SystemExit(f"missing {train}; re-run with --fetch")
        zpath = anchor / "annotations_trainval2017.zip"
        _fetch(COCO_ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                if member.endswith(("instances_train2017.json", "instances_val2017.json")):
                    out = anchor / Path(member).name
                    if not out.exists():
                        out.write_bytes(z.read(member))
                        log(f"extracted {out.name}")
    instances.append(train)
    val_extracted = anchor / "instances_val2017.json"
    if not staged_val.exists() and val_extracted.exists():
        instances.append(val_extracted)
    return image_data, instances


def coco_truth(instances: list[Path], classes: set[str]) -> dict[int, dict[str, list[list[float]]]]:
    """``{coco_image_id: {class: [[x0, y0, x1, y1], ...]}}`` over *classes*.

    Images with no instance of any wanted class still get an (empty) entry:
    COCO is exhaustive, so "annotated and absent" is a *fact* about the image,
    and it is the half of the reference that makes VG's false positives
    measurable at all.
    """
    truth: dict[int, dict[str, list[list[float]]]] = {}
    for path in instances:
        if not path.exists():
            continue
        log(f"loading {path.name} ({path.stat().st_size / 1e6:.0f} MB)...")
        with path.open() as fh:
            data = json.load(fh)
        wanted = {c["id"]: c["name"] for c in data["categories"] if c["name"] in classes}
        for img in data["images"]:
            truth.setdefault(int(img["id"]), {})
            # COCO carries the dimensions, so banding an anchored image needs no
            # JPEG header read at all.
            COCO_DIMS[int(img["id"])] = (int(img["width"]), int(img["height"]))
        for ann in data["annotations"]:
            name = wanted.get(ann["category_id"])
            if name is None:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            truth[int(ann["image_id"])].setdefault(name, []).append([x, y, x + w, y + h])
        log(f"  {len(data['images'])} images, {len(wanted)} of our classes in its vocabulary")
    return truth


def vg_truth(classes: set[str]) -> dict[int, dict[str, list[list[float]]]]:
    """The same shape, read from VG's own ``objects.json``."""
    log("loading VG objects.json...")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    out: dict[int, dict[str, list[list[float]]]] = {}
    for rec in records:
        iid = int(rec["image_id"])
        by_name: dict[str, list[list[float]]] = {}
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if name not in classes:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w > 0 and h > 0:
                by_name.setdefault(name, []).append([x, y, x + w, y + h])
        out[iid] = by_name
    log(f"  {len(out)} VG images")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"), help="where the sources are staged")
    ap.add_argument("--fetch", action="store_true", help="download missing sources")
    ap.add_argument("--out", default="", help="write the per-image verdicts as JSON")
    ap.add_argument("--pool", default="", help="restrict to the image ids in this vg_scale pickle")
    ap.add_argument(
        "--bands",
        action="store_true",
        help="also report per-(class, band) supply using COCO's boxes, i.e. what a "
        "dataset built on the anchored half alone could fill",
    )
    args = ap.parse_args()

    anchor = Path(args.anchor_dir)
    anchor.mkdir(parents=True, exist_ok=True)
    classes = set(pc.SCALE_CLASSES)

    image_data, instances = ensure_sources(anchor, args.fetch)
    truth = coco_truth(instances, classes)

    log(f"loading {image_data.name}...")
    with image_data.open() as fh:
        meta = json.load(fh)
    # VG image -> the COCO image it was sourced from, where there is one.
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    log(f"  {len(coco_of)} of {len(meta)} VG images carry a coco_id")

    pool: set[int] | None = None
    if args.pool:
        import sys  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
        from _cells_io import load_medias  # noqa: PLC0415

        pool = set(load_medias(Path(args.pool)))
        log(f"  restricted to the {len(pool)} images of {Path(args.pool).name}")

    vg = vg_truth(classes)

    # Per class: the four cells of the confusion between VG and COCO.
    rows: dict[str, dict[str, int]] = {c: defaultdict(int) for c in pc.SCALE_CLASSES}
    verdicts: dict[str, dict[str, list]] = {}
    anchored = 0
    for vg_id, cid in sorted(coco_of.items()):
        if pool is not None and vg_id not in pool:
            continue
        ref = truth.get(cid)
        if ref is None or vg_id not in vg:
            continue  # COCO image not in the 2017 split, or VG image not scanned
        anchored += 1
        mine = vg[vg_id]
        for c in pc.SCALE_CLASSES:
            in_vg, in_coco = c in mine, c in ref
            key = "both" if (in_vg and in_coco) else "vg_missing" if in_coco else "coco_missing" if in_vg else "neither"
            rows[c][key] += 1
            if key == "vg_missing":
                verdicts.setdefault(c, {}).setdefault("vg_missing", []).append(
                    {"image_id": vg_id, "coco_id": cid, "boxes": ref[c]}
                )
            elif key == "coco_missing":
                verdicts.setdefault(c, {}).setdefault("coco_missing", []).append(
                    {"image_id": vg_id, "coco_id": cid, "boxes": mine[c]}
                )

    print(f"\n{anchored} images have an exhaustive COCO reference\n")
    hdr = f"{'class':<12}{'both':>7}{'vg_missing':>12}{'coco_only':>11}{'neither':>9}{'VG recall':>11}{'neg noise':>11}"
    print(hdr)
    print("-" * len(hdr))
    total = defaultdict(int)
    for c in pc.SCALE_CLASSES:
        r = rows[c]
        true_pos = r["both"] + r["vg_missing"]
        recall = r["both"] / true_pos if true_pos else float("nan")
        # The number that matters most: of the images VG calls negative for this
        # class, what share actually hold one? That is the poison in the ~95%
        # negative pool every cell rests on.
        vg_neg = r["vg_missing"] + r["neither"]
        noise = r["vg_missing"] / vg_neg if vg_neg else float("nan")
        for k, v in r.items():
            total[k] += v
        print(
            f"{c:<12}{r['both']:>7}{r['vg_missing']:>12}{r['coco_missing']:>11}"
            f"{r['neither']:>9}{recall:>11.2f}{noise:>11.4f}"
        )
    tp = total["both"] + total["vg_missing"]
    neg = total["vg_missing"] + total["neither"]
    print("-" * len(hdr))
    print(
        f"{'pooled':<12}{total['both']:>7}{total['vg_missing']:>12}{total['coco_missing']:>11}"
        f"{total['neither']:>9}{total['both'] / tp:>11.2f}{total['vg_missing'] / neg:>11.4f}"
    )
    print(
        "\nVG recall = of the objects COCO annotates, the share VG also annotates."
        "\nneg noise = of the images VG treats as negatives, the share that actually hold one."
    )

    if args.bands:
        # What could a dataset built on the anchored half ALONE hold? Its labels
        # would be exhaustive by construction -- a negative is an image COCO
        # annotated and found none in, not merely one nobody mentioned it on --
        # so it needs no correction pass at all. The question is only supply.
        print(f"\n=== per-band supply on the {anchored} anchored images, COCO boxes ===")
        hdr2 = f"{'class':<12}{'small':>8}{'medium':>8}{'large':>8}{'over':>7}{'scatter':>9}{'min':>7}"
        print(hdr2)
        print("-" * len(hdr2))
        worst = None
        for c in pc.SCALE_CLASSES:
            counts = dict.fromkeys((*pc.BOX_BANDS, "oversize"), 0)
            scattered = 0
            for vg_id, cid in coco_of.items():
                if pool is not None and vg_id not in pool:
                    continue
                ref = truth.get(cid)
                wh = COCO_DIMS.get(cid)
                if not ref or wh is None or c not in ref:
                    continue
                area = float(wh[0] * wh[1])
                bs = ref[c]
                union = (
                    max(0.0, max(b[2] for b in bs) - min(b[0] for b in bs))
                    * max(0.0, max(b[3] for b in bs) - min(b[1] for b in bs))
                    / area
                )
                largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in bs) / area
                if union > largest * pc.BAND_MAX_INFLATION:
                    scattered += 1
                    continue
                for band, (lo, hi) in pc.BOX_BANDS.items():
                    if lo <= union < hi:
                        counts[band] += 1
                        break
                else:
                    counts["oversize"] += 1
            mn = min(counts[b] for b in pc.BOX_BANDS)
            worst = mn if worst is None else min(worst, mn)
            print(
                f"{c:<12}{counts['small']:>8}{counts['medium']:>8}{counts['large']:>8}"
                f"{counts['oversize']:>7}{scattered:>9}{mn:>7}"
            )
        print("-" * len(hdr2))
        print(f"binding per-band supply across C: {worst}  (n_pos must not exceed it)")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {"anchored_images": anchored, "classes": list(pc.SCALE_CLASSES)},
                    "counts": {c: dict(rows[c]) for c in pc.SCALE_CLASSES},
                    "verdicts": verdicts,
                },
                indent=1,
            )
            + "\n"
        )
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
