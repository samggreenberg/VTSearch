#!/usr/bin/env python3
"""Export SOD classes (COCO/LVIS/VG via SodDataset) to CD-ViTO's COCO few-shot layout.

Builds a multi-class COCO-format dataset from our staged data so the *published*
CD-ViTO detector can be run on our own classes — to measure its resolution-floor
(per-size AP) and annotation-scaling (1/5/10-shot) before deciding on any port.

Output (matches ``CDFSOD-benchmark/detectron2/data/datasets/builtin.py::register_all_CD``):

    <out-root>/<name>/
      train/                                  # JPEGs for <name>_train and <name>_Kshot
      test/                                   # JPEGs for <name>_test
      annotations/{train.json, test.json, 1_shot.json, 5_shot.json, 10_shot.json}

COCO json shape (from the shipped clipart1k jsons): images=[{file_name,id,width,height}],
annotations=[{image_id,bbox:[x,y,w,h] abs px,area,category_id(1-idx),id,iscrowd:0}],
categories=[{supercategory:'none',id,name}].  ``split``/``k-shot`` logic mirrors the
repo's ``datasets/split.py`` + ``datasets/kshot_split.py`` (per category: K images, one
annotation each), so the output is drop-in for ``build_prototypes.sh`` + ``train_net.py``.

Runs in the MAIN venv on a login node (reads /exp zips, writes /exp, no GPU).

Example:
    .venv/bin/python scripts/cdfsod/export_sod_cocofsod.py \\
        --dataset coco --name sodcoco \\
        --classes "traffic light,stop sign,car,person,bus" \\
        --max-pos-per-class 200 --k-values 1,5,10 \\
        --out-root /exp/mlucio/projects/cdfsod/datasets
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path


def _load_sod_datasets():
    """Import scripts/sod/datasets.py under a unique name (dodge HF `datasets`)."""
    ds_path = Path(__file__).resolve().parent.parent / "sod" / "datasets.py"
    spec = importlib.util.spec_from_file_location("sod_datasets", ds_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass needs the module registered before exec
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("coco", "lvis", "vg"), default="coco")
    ap.add_argument("--classes", required=True, help="comma-separated category names (order = category_id 1..N)")
    ap.add_argument(
        "--name", default="sodcoco", help="dataset name registered in CD-ViTO (dir + <name>_train/test/Kshot)"
    )
    ap.add_argument("--out-root", type=Path, default=Path("/exp/mlucio/projects/cdfsod/datasets"))
    ap.add_argument("--k-values", default="1,5,10", help="comma-separated K for the k-shot support jsons")
    ap.add_argument(
        "--max-pos-per-class",
        type=int,
        default=200,
        help="cap positives sampled per class (0 = all); keeps eval fast + classes balanced",
    )
    ap.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="fraction of the pooled positive images used for train (rest = test)",
    )
    ap.add_argument(
        "--min-box-frac",
        type=float,
        default=0.0,
        help="drop GT boxes below this fraction on either axis (0 = keep all; the sweep uses 0.01). "
        "Default 0 so the small-object behaviour is fully exposed.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sod = _load_sod_datasets()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    cat_id = {sod._norm(c): i + 1 for i, c in enumerate(classes)}  # 1-indexed, in given order
    cat_name = {i + 1: c for i, c in enumerate(classes)}
    want = set(cat_id)
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    frac = args.min_box_frac

    cfg = sod._CONFIG[args.dataset]
    if cfg["kind"] != "coco_lvis":
        raise SystemExit(f"--dataset {args.dataset}: only coco/lvis supported here (vg has no split/file_name).")

    # ---- stream the extract once: per image, collect target-class boxes (+ locator) ----
    import gzip

    per_image: dict[int, dict] = {}  # iid -> {split, file_name, boxes:[(cat_id, x0,y0,x1,y1)]}
    by_class: dict[int, list[int]] = {cid: [] for cid in cat_name}  # cat_id -> image ids (with a surviving box)
    with gzip.open(cfg["extract"], "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = sod._norm(row.get("name", ""))
            if key not in want:
                continue
            x0, y0, x1, y1 = float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])
            if (x1 - x0) < frac or (y1 - y0) < frac:
                continue
            iid = int(row["image_id"])
            rec = per_image.setdefault(
                iid, {"split": str(row["split"]), "file_name": str(row["file_name"]), "boxes": []}
            )
            cid = cat_id[key]
            rec["boxes"].append((cid, x0, y0, x1, y1))
            if iid not in by_class[cid]:  # note: list-membership; fine at these counts
                by_class[cid].append(iid)

    if not per_image:
        raise SystemExit(f"no images found for classes {classes} in {args.dataset}")

    # ---- pick the pooled positive images: cap per class, then union ----
    rng = random.Random(args.seed)
    pool: set[int] = set()
    for cid, ids in by_class.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        keep = ids if args.max_pos_per_class <= 0 else ids[: args.max_pos_per_class]
        pool.update(keep)
        print(f"  class {cat_name[cid]!r}: {len(ids)} imgs available -> {len(keep)} kept")
    pool_ids = sorted(pool)
    rng.shuffle(pool_ids)
    n_train = int(len(pool_ids) * args.train_size)
    train_ids, test_ids = set(pool_ids[:n_train]), set(pool_ids[n_train:])
    print(f"pooled positive images: {len(pool_ids)}  (train {len(train_ids)} / test {len(test_ids)})")

    # ---- write images + build COCO image/annotation records ----
    out = args.out_root / args.name
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    ds = sod.SodDataset(args.dataset)

    def build_split(ids: set[int], subdir: str):
        images, annotations = [], []
        ann_id = 1
        for iid in sorted(ids):
            rec = per_image[iid]
            img = ds._reader.load(rec["split"], rec["file_name"])  # PIL RGB
            w, h = img.size
            fname = Path(rec["file_name"]).name
            img.save(out / subdir / fname, quality=95)
            images.append({"file_name": fname, "id": iid, "width": w, "height": h})
            for cid, x0, y0, x1, y1 in rec["boxes"]:
                bx, by, bw, bh = x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h
                annotations.append(
                    {
                        "image_id": iid,
                        "bbox": [bx, by, bw, bh],
                        "area": bw * bh,
                        "category_id": cid,
                        "id": ann_id,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        return images, annotations

    categories = [{"supercategory": "none", "id": cid, "name": cat_name[cid]} for cid in sorted(cat_name)]
    tr_imgs, tr_anns = build_split(train_ids, "train")
    te_imgs, te_anns = build_split(test_ids, "test")
    ds.close()

    def dump(name, imgs, anns):
        (out / "annotations" / name).write_text(
            json.dumps({"images": imgs, "annotations": anns, "categories": categories})
        )

    dump("train.json", tr_imgs, tr_anns)
    dump("test.json", te_imgs, te_anns)
    print(
        f"train.json: {len(tr_imgs)} imgs / {len(tr_anns)} anns   test.json: {len(te_imgs)} imgs / {len(te_anns)} anns"
    )

    # ---- k-shot jsons from train (mirror kshot_split.py: per category, K images, 1 ann each) ----
    train_img_by_id = {im["id"]: im for im in tr_imgs}
    cat_img_ann: dict[int, dict[int, dict]] = {}
    for a in tr_anns:
        cat_img_ann.setdefault(a["category_id"], {}).setdefault(a["image_id"], a)  # first ann per (cat,image)
    for k in k_values:
        sel_ids: set[int] = set()
        sel_anns = []
        for cid, img_anns in cat_img_ann.items():
            iids = list(img_anns)
            chosen = iids if len(iids) <= k else rng.sample(iids, k)
            sel_ids.update(chosen)
            sel_anns.extend(img_anns[i] for i in chosen)
        dump(f"{k}_shot.json", [train_img_by_id[i] for i in sel_ids], sel_anns)
        print(f"{k}_shot.json: {len(sel_ids)} imgs / {len(sel_anns)} anns (one per category-image)")

    print(f"\nwrote dataset -> {out}")
    print("next: add", repr(args.name), "to builtin.py datasets_name; build_prototypes.sh; write a config; train_net.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
