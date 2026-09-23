#!/usr/bin/env python3
"""Ink coverage of every top-hit page (#4089): how many of SigLIP's best "negatives" are near-blank.

cd scripts/experiments/docmarks && python ../../../docs/experiments/2026-09-22-docmarks-review-4088-4089/measure_blank.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "experiments" / "docmarks"))
import docmarks_config as cfg  # noqa: E402
from sources._common import read_manifest  # noqa: E402

rows = [json.loads(line) for line in (HERE / "measurements" / "tophit_verdicts.jsonl").read_text().splitlines()]
want = {r["page_id"] for r in rows if r["arm"] == "hit"}
pages = {p.page_id: p for p in read_manifest(cfg.OUT / "corpus.jsonl") if p.page_id in want}
out = {}
for pid, page in sorted(pages.items()):
    path = Path(page.path)
    with Image.open(path if path.is_absolute() else cfg.OUT / path) as im:
        a = np.asarray(im.convert("L"))
    # Ink inside the page, ignoring a 3% border where scan frames and edges live.
    h, w = a.shape
    inner = a[int(0.03 * h) : int(0.97 * h), int(0.03 * w) : int(0.97 * w)]
    out[pid] = round(float((inner < 128).mean()), 5)
(HERE / "measurements" / "tophit_ink.json").write_text(json.dumps(out, indent=1) + "\n")
v = np.array(list(out.values()))
print(
    len(v),
    "pages; ink share median",
    np.median(v),
    "; under 0.2%:",
    int((v < 0.002).sum()),
    "; under 0.5%:",
    int((v < 0.005).sum()),
)
