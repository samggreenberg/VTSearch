"""SAM3's zero-shot scores per cell, from ``sam3_score_5213.py``'s shards (#5213, #5241).

Shared by ``text_baseline.py --sam3`` (the zero-shot sort graded as a vote-0 anchor) and ``run_cells.py`` under
``CALIB_SEED_SAM3`` (the sort a session opens on), so both read one image's score for one class the same way.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import common


@lru_cache(maxsize=4)
def load_sam3(root: Path, which: str):
    """``(cat, medias) -> ({id: score}, query)`` from ``sam3_score_5213.py``'s shards (#5213).

    An image's score for a class is its best detection over the class's prompts in *which*
    (``typed``, ``noun``, or ``both``): a merged class ("car", "truck") takes the max over its members.
    """
    from vtscore.eval import scale_bands  # noqa: PLC0415

    table = json.loads((root / "prompts.json").read_text())["classes"]
    shards = sorted(p for p in root.glob("shard_*_of_*.npz") if "limit" not in p.name)
    if not shards:
        raise SystemExit(f"no full shards in {root}")
    n = int(shards[0].name.split("_of_")[1].split(".")[0])
    if len(shards) != n:
        raise SystemExit(f"{root}: {len(shards)} of {n} shards present")
    ids, top1, prompts = [], [], None
    for p in shards:
        z = np.load(p, allow_pickle=True)
        prompts = list(z["prompts"]) if prompts is None else prompts
        if list(z["prompts"]) != prompts:
            raise SystemExit(f"{p}: prompt list differs from the first shard's")
        ids.extend(z["ids"].tolist())
        top1.append(z["top1"].astype(np.float32))
    mat = np.concatenate(top1)
    row = {i: k for k, i in enumerate(ids)}
    col = {q: j for j, q in enumerate(prompts)}
    common.log(f"SAM3 scores: {mat.shape[0]} images x {mat.shape[1]} prompts from {n} shards ({which})")

    def score(cat: str, medias: dict):
        cls, _ = scale_bands.parse_cell(cat)
        kinds = ("typed", "noun") if which == "both" else (which,)
        qs = [q for k in kinds for q in table[cls][k]]
        best = mat[:, [col[q] for q in qs]].max(axis=1)
        missing = [i for i in medias if i not in row]
        if missing:
            raise SystemExit(f"{cat}: {len(missing)} pool images have no SAM3 score (e.g. {missing[:3]})")
        return {i: float(best[row[i]]) for i in medias}, " | ".join(qs)

    return score
