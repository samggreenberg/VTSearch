"""Whole-page contact sheets of the pages ``meta_scan.py`` found (#3914).

python meta_sheets.py measurements/meta_hits.json <out-dir>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COLS, W, H, PER = 5, 420, 560, 15


def main(hits_path: Path, out: Path) -> int:
    corpus = Path(os.environ.get("VTS_FULLMARKS_OUT", "/expscratch/sgreenberg/fullmarks/corpus"))
    paths = {}
    for line in (corpus / "corpus.jsonl").open(encoding="utf-8"):
        page = json.loads(line)
        if page["source"] == "ucsf":
            paths[page["page_id"]] = (page["path"], page["meta"].get("author"))
    hits = json.loads(hits_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for company, ids in hits.items():
        for start in range(0, len(ids), PER):
            chunk = ids[start : start + PER]
            rows = (len(chunk) + COLS - 1) // COLS
            sheet = Image.new("L", (COLS * (W + 8), rows * (H + 22)), 255)
            draw = ImageDraw.Draw(sheet)
            for i, page_id in enumerate(chunk):
                x, y = (i % COLS) * (W + 8), (i // COLS) * (H + 22)
                with Image.open(paths[page_id][0]) as img:
                    img = img.convert("L")
                    img.thumbnail((W, H))
                    sheet.paste(img, (x, y + 20))
                draw.text((x + 2, y + 3), f"[{start + i + 1}] {page_id} {str(paths[page_id][1])[:28]}", fill=0)
            name = f"meta_{company.replace(' ', '_').replace('&', 'and')}_{start // PER + 1}.png"
            sheet.save(out / name)
            print(name, len(chunk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
