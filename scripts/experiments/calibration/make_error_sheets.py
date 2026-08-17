"""Contact sheets: the errors as pictures rather than as filenames.

`error_report.py` lists the score, the file id and the dataset's annotations for
every error worth looking at. That is enough to *count* with and not enough to
*adjudicate* with: deciding whether a false positive is a model error or a
missing label means looking at the image. This renders the same rows as image
grids — score, annotations, and the target's ground-truth box where the dataset
has one — so the judgement can be made from the report instead of from a cluster
login.

Runs where the images are (the GRID), because the dumps carry file ids and
nothing else:

    python make_error_sheets.py --dumps /expscratch/$USER/bench-errors/dumps \\
        --out /expscratch/$USER/bench-errors/sheets

Then copy the PNGs into the study's `figures/` directory. Sheets are declared in
`SHEETS` below, one per claim the report makes about a *kind* of error.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import pickle
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

#: Where each dataset's source images live. VG is split across two directories
#: and COCO is only ever unpacked as a zip, so both cases have to be handled
#: rather than assuming one directory per dataset.
DATA = Path(os.environ.get("VTSEARCH_DATA_DIR", "/expscratch/sgreenberg/vts-cache/datadir"))
IMAGE_ROOTS: dict[str, list[Path]] = {
    "visual_genome_m": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "vg_box_small": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "vg_box_medium": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "vg_box_large": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "coco_val": [Path("/exp/scale26/datasets/external/COCO/images/val2017.zip")],
    "caltech101_m": [DATA / "caltech-101/101_ObjectCategories"],
}
#: Region boxes are a property of the dataset, not of the embedder, so read them
#: from the cheapest pickle that has them (the whole-image SigLIP one: the patch
#: pickles are 3.6 GB and carry the identical `regions`).
BOX_PICKLES = {ds: DATA / f"embeddings/{ds}__siglip.pkl" for ds in IMAGE_ROOTS}

THUMB = 340  # px on the long edge; enough to judge a small object, small on disk

SHEETS: list[dict] = [
    {
        "name": "sky_fp",
        "dump": "vg_siglip_sky",
        "kind": "fp",
        "dataset": "visual_genome_m",
        "n": 8,
        "title": "visual_genome_m / sky — `siglip`'s most confident FALSE positives",
        "note": "Every one has sky in it. VG annotated the clouds, the grass and the road, and left sky off.",
    },
    {
        "name": "sky_fn",
        "dump": "vg_siglip_sky",
        "kind": "fn",
        "dataset": "visual_genome_m",
        "n": 8,
        "title": "visual_genome_m / sky — the false NEGATIVES, which are genuine misses",
        "note": "These do carry a `sky` annotation, and the sky is a thin strip behind a person or a building.",
    },
    {
        "name": "glasses_fp",
        "dump": "vgsmall_siglip_glasses",
        "kind": "fp",
        "dataset": "vg_box_small",
        "n": 8,
        "title": "vg_box_small / glasses — false positives, and the label that should have matched",
        "note": "364 of this arm's false positives are annotated `sunglasses`: one object under two labels.",
    },
    {
        "name": "tip_pos",
        "dump": "vgsmall_siglip_tip",
        "kind": "fn",
        "dataset": "vg_box_small",
        "n": 8,
        "boxes": True,
        "title": "vg_box_small / tip — what the dataset calls a positive (boxes drawn)",
        "note": "The tip of anything: a nose, a horn, a shoe. There is no visual class here to learn.",
    },
    {
        "name": "clock_rescued",
        "compare": ("coco_siglip_clock", "coco_dinov3_clock"),
        "dataset": "coco_val",
        "n": 8,
        "boxes": True,
        "title": "coco_val / clock — clocks the whole-image arm MISSED and the box arm FOUND",
        "note": "Sorted by how much else is in the frame: the more the image holds, the more one pooled vector dilutes the clock.",
    },
    {
        "name": "bus_fp",
        "dump": "vg_siglip_bus",
        "kind": "fp",
        "dataset": "visual_genome_m",
        "n": 8,
        "title": "visual_genome_m / bus — a threshold collapse, not a ranking failure",
        "note": "1,210 of 2,030 negatives pass the cut. These are the most confident of them.",
    },
    {
        "name": "sky_binary_fp",
        "dump": "vg_dinov3_sky_binary",
        "kind": "fp",
        "dataset": "visual_genome_m",
        "n": 8,
        "title": "visual_genome_m / sky — `dinov3_patch` with the BOX TAKEN AWAY",
        "note": "The same cell flags 1,305 of 1,703 negatives without a box against 504 with one; it is scoring 'outdoor street photo'.",
    },
    {
        "name": "text_bear_fp",
        "dump": "text/text__coco_val__siglip__bear",
        "kind": "fp",
        "dataset": "coco_val",
        "n": 8,
        "title": "coco_val / bear — what a TYPED query flags",
        "note": "43 of 626 false positives are annotated `teddy bear`, a different COCO class. A user typing 'bear' would call these hits.",
    },
    {
        "name": "text_airplanes_fp",
        "dump": "text/text__caltech101_m__siglip__airplanes",
        "kind": "fp",
        "dataset": "caltech101_m",
        "n": 8,
        "title": "caltech101_m / airplanes — every false positive a typed query makes",
        "note": "All 13 are helicopters. Nothing is wrong with the labels or the ranking; two Bad votes delete the class.",
    },
]


def load_dump(dumps: Path, name: str) -> list[dict]:
    with (dumps / f"{name}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["score"] = float(r["score"])
        r["label"] = int(r["label"])
        r["threshold"] = float(r["threshold"])
    return rows


def load_boxes(dataset: str) -> dict[str, list[dict]]:
    """`filename -> regions`, read from the dataset's cheapest pickle."""
    path = BOX_PICKLES.get(dataset)
    if path is None or not path.exists():
        return {}
    with path.open("rb") as fh:
        medias = pickle.load(fh)  # noqa: S301 - this study's own dataset pickle
    return {m["filename"]: (m.get("regions") or []) for m in medias.values() if m.get("filename")}


def open_image(dataset: str, filename: str) -> Image.Image | None:
    for root in IMAGE_ROOTS.get(dataset, []):
        if root.suffix == ".zip":
            if not root.exists():
                continue
            with zipfile.ZipFile(root) as zf:
                for member in (filename, f"{root.stem}/{filename}"):
                    try:
                        data = zf.read(member)
                    except KeyError:
                        continue
                    return Image.open(io.BytesIO(data)).convert("RGB")
            continue
        candidate = root / filename
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    return None


def wrap(text: str, width: int = 34, lines: int = 2) -> str:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
            if len(out) == lines:
                break
        else:
            cur = f"{cur} {w}".strip()
    if len(out) < lines and cur:
        out.append(cur)
    joined = "\n".join(out[:lines])
    return joined + (" …" if len(" ".join(words)) > len(joined.replace("\n", " ")) else "")


def pick(rows: list[dict], kind: str, n: int) -> list[dict]:
    thr = rows[0]["threshold"]
    if kind == "fp":  # negatives the model was most confident about
        return sorted((r for r in rows if r["label"] == 0 and r["score"] >= thr), key=lambda r: -r["score"])[:n]
    # positives the model was least confident about
    return sorted((r for r in rows if r["label"] == 1 and r["score"] < thr), key=lambda r: r["score"])[:n]


def pick_rescued(missed: list[dict], found: list[dict], n: int) -> list[dict]:
    """Positives below one arm's threshold and above the other's."""
    thr_m, thr_f = missed[0]["threshold"], found[0]["threshold"]
    by_id = {r["media_id"]: r for r in found}
    out = []
    for r in missed:
        other = by_id.get(r["media_id"])
        if not other or r["label"] != 1:
            continue
        if r["score"] < thr_m and other["score"] >= thr_f:
            out.append({**r, "other_score": other["score"], "other_threshold": thr_f})
    # Most cluttered first: that is the mechanism being illustrated.
    return sorted(out, key=lambda r: -len(r["all_categories"].split("|")))[:n]


def draw_sheet(spec: dict, rows: list[dict], boxes: dict[str, list[dict]], out: Path) -> bool:
    if not rows:
        print(f"  {spec['name']}: no rows matched; skipped")
        return False
    ncols = 4
    nrows = (len(rows) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.5 * nrows), squeeze=False)
    flat = axes.ravel()
    target = rows[0]["target_category"]
    for ax, row in zip(flat, rows, strict=False):
        img = open_image(spec["dataset"], row["filename"])
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=8)
            continue
        img.thumbnail((THUMB, THUMB))
        ax.imshow(img)
        if spec.get("boxes"):
            for region in boxes.get(row["filename"], []):
                if str(region.get("label", "")).lower() != target.lower():
                    continue
                x0, y0, x1, y1 = region["box"]
                w, h = img.size
                ax.add_patch(
                    mpatches.Rectangle(
                        (x0 * w, y0 * h),
                        (x1 - x0) * w,
                        (y1 - y0) * h,
                        fill=False,
                        edgecolor="#F2C744",
                        lw=2.0,
                    )
                )
        head = f"{row['score']:.3f}  {row['filename'].split('/')[-1]}"
        if "other_score" in row:
            head = f"missed {row['score']:.3f} · found {row['other_score']:.3f}\n{row['filename'].split('/')[-1]}"
        ax.set_title(head, fontsize=8)
        cats = row["all_categories"].replace("|", ", ") or "(no annotations)"
        ax.set_xlabel(wrap(cats), fontsize=7, color="#444")
    for ax in flat[len(rows) :]:
        ax.axis("off")
    fig.suptitle(spec["title"], fontsize=11, y=1.0)
    fig.text(0.5, -0.01, spec["note"], ha="center", va="top", fontsize=8.5, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size // 1024} KB, {len(rows)} images)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated sheet names")
    args = ap.parse_args()

    dumps, out = Path(args.dumps), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    box_cache: dict[str, dict[str, list[dict]]] = {}

    for spec in SHEETS:
        if wanted and spec["name"] not in wanted:
            continue
        print(f"{spec['name']}:")
        if "compare" in spec:
            missed, found = (load_dump(dumps, n) for n in spec["compare"])
            rows = pick_rescued(missed, found, spec["n"])
        else:
            rows = pick(load_dump(dumps, spec["dump"]), spec["kind"], spec["n"])
        boxes = {}
        if spec.get("boxes"):
            ds = spec["dataset"]
            if ds not in box_cache:
                box_cache[ds] = load_boxes(ds)
            boxes = box_cache[ds]
        draw_sheet(spec, rows, boxes, out / f"examples_{spec['name']}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
