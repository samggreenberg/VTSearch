"""Tile a class's negative slate into contact sheets for a triage pass.

Reviewing thousands of images one at a time is not practical for a model any
more than for a person, so the pass is triage-then-confirm: scan tiles, flag
anything that might hold the class, then look at each flag at full resolution.

The tiles are ordered by the same text-query score the slate was ranked on, so
the most likely hidden positives arrive first and a partial pass is still a
useful pass.

Triage recall is **measured, not assumed**: run it over a class a human has
already reviewed and compare the flags against their findings. A sheet pass will
miss sub-patch objects — the exam put full-resolution small-object recall at
0.50 — so the number that matters is how much *more* is lost by tiling.

Usage::

    python make_contact_sheets.py --class bicycle --out /expscratch/$USER/vgscale-3156/sheets_neg
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pile_config as pc

pc.setup_env()

COLS, ROWS = 5, 4
TILE = 260
LABEL = 22


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="klass", required=True)
    ap.add_argument("--slates", default=str(pc.PILE.parent / "vgscale-3156" / "slates"))
    ap.add_argument("--out", default=str(pc.PILE.parent / "vgscale-3156" / "sheets_neg"))
    args = ap.parse_args()

    from PIL import Image, ImageDraw  # noqa: PLC0415

    from build_pile import _vg_image_paths  # noqa: PLC0415

    folder = args.klass.replace(" ", "_")
    man = Path(args.slates) / folder / "manifest.csv"
    rows = [r for r in csv.DictReader(man.open()) if r["stratum"] in ("boundary", "random")]
    rows.sort(key=lambda r: -float(r["text_score"]))
    paths = _vg_image_paths()

    out = Path(args.out) / folder
    out.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    index = []
    for s in range((len(rows) + per - 1) // per):
        chunk = rows[s * per : (s + 1) * per]
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + LABEL)), (16, 18, 19))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            src = paths.get(int(r["image_id"]))
            if src is None:
                continue
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((TILE - 6, TILE - 6), Image.LANCZOS)
            ox = (k % COLS) * TILE
            oy = (k // COLS) * (TILE + LABEL)
            sheet.paste(im, (ox + (TILE - im.width) // 2, oy + LABEL + (TILE - LABEL - im.height) // 2))
            d.text((ox + 6, oy + 5), f"{k + 1:02d}", fill=(255, 214, 92))
            index.append(
                {
                    "sheet": s + 1,
                    "tile": k + 1,
                    "image_id": int(r["image_id"]),
                    "stratum": r["stratum"],
                    "exhaustive": r["exhaustive"],
                }
            )
        sheet.save(out / f"sheet{s + 1:02d}.jpg", quality=88)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(rows)} images -> {(len(rows) + per - 1) // per} sheets in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
