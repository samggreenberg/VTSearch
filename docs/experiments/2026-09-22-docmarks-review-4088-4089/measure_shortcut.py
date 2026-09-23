#!/usr/bin/env python3
"""The mark-blind "UCSF Tobacco first" control for the four UCSF classes, before and after #4088.

    cd scripts/experiments/docmarks
    python ../../../docs/experiments/2026-09-22-docmarks-review-4088-4089/measure_shortcut.py

"Before" is the pre-apply backup of ``classes.json``; "after" is the live corpus.
Writes ``measurements/shortcut.json``: per tier and class, the ``own_verified``
pool size, its positives, its reviewed Tobacco negatives and the control's AP.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "experiments" / "docmarks"))

import docmarks_config as cfg  # noqa: E402
import eval_retrieval as ev  # noqa: E402
from sources._common import read_manifest  # noqa: E402

BEFORE = cfg.OUT / "backup-pre4088-20260922" / "classes.json"


def control_ap(pool, positives, score, reps: int = 20) -> float:
    """AP of a mark-blind ranking, ties broken at random and averaged."""
    total = 0.0
    for r in range(reps):
        rng = random.Random(r)
        total += ev.average_precision(sorted(pool, key=lambda p: (-score(p), rng.random())), positives)
    return total / reps


def main() -> int:
    after = json.loads((cfg.OUT / "classes.json").read_text(encoding="utf-8"))
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    pages = list(read_manifest(cfg.OUT / "corpus.jsonl"))
    industry = {p.page_id: (p.meta or {}).get("industry") for p in pages}
    source = {p.page_id: p.source for p in pages}

    def tobacco_first(p: str) -> int:
        return (source[p] == "ucsf") + (industry.get(p) == "Tobacco")

    out = []
    for tier in cfg.TIER_ORDER:
        want = set(cfg.TIER_ORDER[: cfg.TIER_ORDER.index(tier) + 1])
        by_source = collections.defaultdict(list)
        for p in pages:
            if (p.meta or {}).get("tier") in want:
                by_source[p.source].append(p.page_id)
        for cid in sorted(c for c in after if c.startswith("ucsf/")):
            row = {"tier": tier, "class_id": cid}
            for tag, meta in (("before", before[cid]), ("after", after[cid])):
                pools = ev.class_pools(meta, by_source, industry)
                pool, pos = pools["own_verified"], pools["positives"]
                row[tag] = {
                    "pool": len(pool),
                    "positives": len(pos),
                    "tobacco_negatives": sum(1 for p in pool - pos if industry.get(p) == "Tobacco"),
                    "industry_prior_ap": round(control_ap(pool, pos, tobacco_first), 4),
                }
            out.append(row)
            print(tier, cid, row["before"], "->", row["after"], flush=True)
    (HERE / "measurements" / "shortcut.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
