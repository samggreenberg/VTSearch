"""Look at what the contamination rule removed (#3904).

For each Tobacco800 class: its query crop, then the top excluded UCSF pages
SigLIP ranked highest over the naive pool, each shown as the top 35% of the
page.  Whether those pages carry the class's mark unlabelled is the question the
rule exists for, and only looking answers it.

    python excluded_sheet.py <eval-out>/m/excluded_hits.json m <out.png>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import read_manifest  # noqa: E402

PER_CLASS = 4
CELL_W, CELL_H = 420, 190


def main(hits_path: Path, tier: str, out: Path) -> int:
    hits = json.loads(hits_path.read_text(encoding="utf-8"))[tier]
    classes = json.loads((cfg.OUT / "classes.json").read_text(encoding="utf-8"))
    paths = {p.page_id: p.path for p in read_manifest(cfg.OUT / "corpus.jsonl") if p.source == "ucsf"}
    rows = sorted(c for c in hits if c.startswith("tobacco800/"))
    sheet = Image.new("L", ((PER_CLASS + 1) * (CELL_W + 6), len(rows) * (CELL_H + 20)), 255)
    draw = ImageDraw.Draw(sheet)
    for r, cid in enumerate(rows):
        y = r * (CELL_H + 20)
        cells = [(f"{cid} query", classes[cid]["query_crop"], False)]
        ucsf = [h for h in hits[cid]["top"]["siglip"] if h["page_id"].startswith("ucsf/")][:PER_CLASS]
        cells += [(f"{h['page_id']} rank {h['rank_naive']}", paths[h["page_id"]], True) for h in ucsf]
        for c, (label, path, band) in enumerate(cells):
            x = c * (CELL_W + 6)
            with Image.open(path) as img:
                img = img.convert("L")
                if band:
                    img = img.crop((0, 0, img.width, int(img.height * 0.35)))
                img.thumbnail((CELL_W, CELL_H))
                sheet.paste(img, (x, y + 18))
            draw.text((x + 2, y + 3), label, fill=0)
    sheet.save(out)
    print(f"wrote {out} ({len(rows)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])))
