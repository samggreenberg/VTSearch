"""Contact sheets of UCSF letterhead bands for the #3901 count.

A fixed-seed sample of N bands per candidate author, the top
``LETTERHEAD_BAND_FRAC`` of each page, numbered on one sheet per author.  The
count itself is made by eye from the sheets; ``sample.json`` maps each number
back to its page so every call can be checked.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
from sources._common import read_manifest  # noqa: E402

N = 40
COLS = 4
THUMB_W = 460


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    by_author: dict[str, list] = {}
    for page in read_manifest(cfg.OUT / "corpus.jsonl"):
        author = page.meta.get("letterhead_author")
        if author:
            by_author.setdefault(author, []).append(page)
    rng = random.Random(3901)
    sample = {}
    for si, author in enumerate(sorted(by_author)):
        pages = sorted(by_author[author], key=lambda p: p.page_id)
        picked = rng.sample(pages, min(N, len(pages)))
        thumbs = []
        for page in picked:
            with Image.open(page.path) as img:
                img = img.convert("L")
                band = img.crop((0, 0, img.width, int(img.height * cfg.LETTERHEAD_BAND_FRAC)))
                h = int(band.height * THUMB_W / band.width)
                thumbs.append(band.resize((THUMB_W, h)))
        cell_h = max(t.height for t in thumbs) + 18
        rows = (len(thumbs) + COLS - 1) // COLS
        sheet = Image.new("L", (COLS * (THUMB_W + 8), rows * cell_h), 255)
        draw = ImageDraw.Draw(sheet)
        for i, t in enumerate(thumbs):
            x, y = (i % COLS) * (THUMB_W + 8), (i // COLS) * cell_h
            sheet.paste(t, (x, y + 16))
            draw.rectangle((x, y + 16, x + THUMB_W - 1, y + 16 + t.height - 1), outline=0)
            draw.text((x + 2, y + 2), f"[{i + 1}] {picked[i].page_id}", fill=0)
        name = f"sheet{si + 1}.png"
        sheet.save(out / name)
        sample[author] = {"sheet": name, "pool": len(pages), "pages": [p.page_id for p in picked]}
        print(f"{name}: {author} ({len(picked)} of {len(pages)})", flush=True)
    (out / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
