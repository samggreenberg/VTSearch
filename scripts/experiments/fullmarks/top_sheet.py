"""Look at what a structural ranker put on top (#3911).

For each named class: the query crop, then the pool's top-N pages by inlier count
from an ``eval_*_rank.py`` run's ``inliers.json``, each whole page shrunk, labelled
POS/neg with its inlier count.  A near-perfect AP is only believable if these are
the mark and not near-duplicate pages or shared layout.

    python top_sheet.py --run <dir> --classes a,b,c --out sheet.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
from sources._common import read_manifest  # noqa: E402

W, H = 230, 300


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inliers = json.loads((args.run / "inliers.json").read_text(encoding="utf-8"))
    classes = json.loads((cfg.OUT / "classes.json").read_text(encoding="utf-8"))
    paths = {p.page_id: p.path for p in read_manifest(cfg.OUT / "corpus.jsonl") if p.meta.get("tier") == "s"}
    wanted = args.classes.split(",")
    sheet = Image.new("L", ((args.top + 1) * (W + 6), len(wanted) * (H + 20)), 255)
    draw = ImageDraw.Draw(sheet)
    for r, cid in enumerate(wanted):
        meta = classes[cid]
        positives = set(meta["page_ids"]) - {meta["query_page_id"]}
        ranked = sorted(inliers[cid].items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))[: args.top]
        cells = [(f"{cid.split('/')[-1][:24]} query", meta["query_crop"])]
        cells += [(f"{'POS' if p in positives else 'neg'} {v[0]} {p.split('/')[-1][:14]}", paths[p]) for p, v in ranked]
        y = r * (H + 20)
        for c, (label, path) in enumerate(cells):
            x = c * (W + 6)
            with Image.open(path) as img:
                img = img.convert("L")
                img.thumbnail((W, H))
                sheet.paste(img, (x, y + 18))
            draw.text((x + 2, y + 3), label, fill=0)
    sheet.save(args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
