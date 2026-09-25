#!/usr/bin/env python3
"""Measurements for the UCSF un-banded contamination check (#3922).

    cd scripts/experiments/fullmarks && python ../../../docs/experiments/2026-09-22-fullmarks-ucsf-contamination/measure.py

Reads the corpus and the banked verdicts, writes ``measurements/``:

* ``verdicts.jsonl`` -- every answer, joined to the page's UCSF industry.
* ``pools.json`` -- each UCSF class's ``own_verified`` pool under the v4.0 and
  v4.1 rules, per tier, and two mark-blind controls scored on it.
"""

from __future__ import annotations

import collections
import copy
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "experiments" / "fullmarks"))

import fullmarks_config as cfg  # noqa: E402
import eval_retrieval as ev  # noqa: E402
from sources._common import read_manifest  # noqa: E402

VERDICTS = Path("/expscratch/sgreenberg/fullmarks/contam-ucsf/verdicts.final.jsonl")
V40 = frozenset({"ucsf"})
V41 = frozenset({"ucsf:Tobacco"})


def control_ap(pool, positives, score, reps=20):
    """AP of a mark-blind ranking, ties broken at random and averaged."""
    out = []
    for r in range(reps):
        rng = random.Random(r)
        order = sorted(pool, key=lambda p: (-score(p), rng.random()))
        out.append(ev.average_precision(order, positives))
    return sum(out) / len(out)


def main() -> int:
    classes = json.loads((cfg.OUT / "classes.json").read_text(encoding="utf-8"))
    pages = list(read_manifest(cfg.OUT / "corpus.jsonl"))
    industry = {p.page_id: (p.meta or {}).get("industry") for p in pages}
    source = {p.page_id: p.source for p in pages}

    meas = HERE / "measurements"
    meas.mkdir(exist_ok=True)
    with open(meas / "verdicts.jsonl", "w", encoding="utf-8") as f:
        for line in VERDICTS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["industry"] = industry.get(row["page_id"])
            f.write(json.dumps(row, sort_keys=True) + "\n")

    out = []
    for tier in cfg.TIER_ORDER:
        want = set(cfg.TIER_ORDER[: cfg.TIER_ORDER.index(tier) + 1])
        by_source = collections.defaultdict(list)
        for p in pages:
            if (p.meta or {}).get("tier") in want:
                by_source[p.source].append(p.page_id)
        for cid, meta in sorted(classes.items()):
            if not cid.startswith("ucsf/"):
                continue
            row = {"tier": tier, "class_id": cid}
            for tag, rule in (("v4.0", V40), ("v4.1", V41)):
                m = copy.deepcopy(meta)
                m["eligible_distractor_sources"] = sorted(
                    s for s in ("spods", "staver", "tobacco800", "ucsf", "synth") if s != "ucsf" or rule == V41
                )
                saved = cfg.CONTAMINATES["ucsf"]
                cfg.CONTAMINATES["ucsf"] = rule
                try:
                    pools = ev.class_pools(m, by_source, industry)
                finally:
                    cfg.CONTAMINATES["ucsf"] = saved
                pool, pos = pools["own_verified"], pools["positives"]
                row[tag] = {
                    "pool": len(pool),
                    "positives": len(pos),
                    "tobacco_negatives": sum(1 for p in pool - pos if industry.get(p) == "Tobacco"),
                }
                if tier == "m":
                    row[tag]["source_prior_ap"] = control_ap(pool, pos, lambda p: source[p] == "ucsf")
                    row[tag]["industry_prior_ap"] = control_ap(
                        pool, pos, lambda p: (source[p] == "ucsf") + (industry.get(p) == "Tobacco")
                    )
            out.append(row)
    (meas / "pools.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
