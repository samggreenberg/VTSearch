#!/usr/bin/env python3
"""State of the App (#4159): thumbnails for the images the influence analysis names.

    python thumbs.py --analysis <exp>/analysis --out <report dir>/images --n 12

Writes one small JPEG per named image (whole image, shrunk to ``--px``) and
``images.md``, a markdown table the report includes: the images with an effect
of their own (``analyze.py``'s ``resid_z``: credit net of cell, label and
phase), most helpful and most harmful, with how many times they were clicked,
by how many detectors, and early vs late. Images past |z| > 3 are marked; the
rest of the ``--n`` slots fill with the next strongest.

Kept deliberately light: a review page with dozens of full-size images goes
black on a laptop (the review-sheet weight budget), so each thumbnail is ~10 KB.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pile"))

import pile_config as pc  # noqa: E402


def _filenames(ids: set[int]) -> dict[int, str]:
    import json  # noqa: PLC0415

    out: dict[int, str] = {}
    for split in ("val2017", "train2017"):
        with (pc.COCO_ANCHOR_DIR / f"instances_{split}.json").open() as fh:
            for im in json.load(fh)["images"]:
                if int(im["id"]) in ids:
                    out[int(im["id"])] = im["file_name"]
    return out


def _thumb(zips: list[zipfile.ZipFile], name: str, px: int) -> bytes | None:
    from PIL import Image  # noqa: PLC0415

    for zf in zips:
        for member in (f"{Path(zf.filename).stem}/{name}", name):
            try:
                data = zf.read(member)
            except KeyError:
                continue
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.thumbnail((px, px))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=72, optimize=True)
            return buf.getvalue()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--px", type=int, default=180)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img = pd.read_csv(args.analysis / "images.csv")
    seen = img[img["resid_z"].notna()].sort_values("resid_z", ascending=False)
    picks = {"Most helpful": seen.head(args.n), "Most harmful": seen.tail(args.n).iloc[::-1]}
    ids = {int(i) for df in picks.values() for i in df["image_id"]}
    names = _filenames(ids)
    zips = [zipfile.ZipFile(z) for z in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP)]

    flagged = int((seen["resid_z"].abs() > 3).sum())
    lines = [
        f"Images tested (clicked 10+ times): {len(seen)} of {len(img)}; {flagged} past |z| > 3. "
        "*Net* is the mean cost removed per click beyond what an average click with the same "
        "cell, label and phase removed; *z* is that in standard errors.",
        "",
    ]
    for title, df in picks.items():
        lines += [f"### {title}", ""]
        lines += [
            "| image | id | clicks | detectors | as positive | net / click | z | raw / click | early | late |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in df.itertuples():
            jpg = _thumb(zips, names.get(int(r.image_id), ""), args.px)
            cell = ""
            if jpg is not None:
                (args.out / f"{int(r.image_id)}.jpg").write_bytes(jpg)
                cell = f"![{int(r.image_id)}]({args.out.name}/{int(r.image_id)}.jpg)"
            fmt = lambda x: "" if pd.isna(x) else f"{x:+.3f}"  # noqa: E731
            lines.append(
                f"| {cell} | {int(r.image_id)} | {int(r.n_obs)} | {int(r.n_detectors)} | {int(r.clicked_as_positive)} "
                f"| {fmt(r.resid)} | {r.resid_z:+.1f} | {fmt(r.help_cost)} | {fmt(r.help_early)} | {fmt(r.help_late)} |"
            )
        lines.append("")
    (args.out.parent / "images.md").write_text("\n".join(lines))
    kb = sum(p.stat().st_size for p in args.out.glob("*.jpg")) / 1024
    print(f"{len(list(args.out.glob('*.jpg')))} thumbnails, {kb:.0f} KB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
