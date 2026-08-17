"""What a precision change actually buys, end to end (issue #3143).

    python report_timings.py

The issue quotes **4.2x** for `siglip2_l`, measured on the GPU forward alone. That
number is real and it is not what a user gets, for two reasons:

* PR #3151 already overlapped image decode with the forward, so the stage fp16
  speeds up is no longer the whole cost. Quoting a forward-only speedup after
  that landed would over-promise.
* The arms here also pay dataset load, the CPU-side processor, pickling, and the
  text pass. Those are identical across arms, which is exactly why the honest
  figure is the *difference between arms*, not a ratio of a stage in isolation.

So this reports per-cell wall time from each arm's ``provenance.json`` — same
dataset, same code, same commit, one card per row — and the speedup against the
same-card fp32 arm. Rows are only compared within a card: comparing fp16 on an
L40S against fp32 on a V100 would fold the #3144 GPU effect into the precision
claim, which is the confound this study's control arm exists to keep separate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import precision_config as pcfg  # noqa: E402


def main() -> int:
    rows = []
    missing = []
    for arm, spec in pcfg.ARMS.items():
        path = pcfg.provenance_path(arm)
        if not path.exists():
            missing.append(arm)
            continue
        prov = json.loads(path.read_text())
        per_cell = {c["embedder"]: c for c in prov.get("cells", [])}
        rows.append(
            {
                "arm": arm,
                "precision": spec["precision"],
                "card": spec["gpu"],
                "gpu_name": prov.get("probe", {}).get("gpu_name", "?"),
                "wall": prov.get("wall_seconds"),
                "cells": per_cell,
            }
        )

    if missing:
        print(f"MISSING provenance for {len(missing)} arm(s): {', '.join(missing)}")
        print("(reported rather than silently excluded)")
        print()

    embedders = pcfg.EMBEDDERS
    # Same-card fp32 is the only legitimate baseline for a precision claim.
    base = {r["card"]: r for r in rows if r["precision"] == "fp32"}

    header = f"{'arm':16s} {'precision':14s} {'card':6s} {'arm wall':>9s}"
    for emb in embedders:
        header += f" | {emb + ' s':>12s} {'med/s':>7s} {'vs fp32':>8s}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["card"], r["precision"])):
        line = f"{r['arm']:16s} {r['precision']:14s} {r['card']:6s} {r['wall'] or 0:8.0f}s"
        for emb in embedders:
            cell = r["cells"].get(emb, {})
            secs = cell.get("wall_seconds")
            rate = cell.get("medias_per_second")
            ref = (base.get(r["card"], {}).get("cells", {}) or {}).get(emb, {}).get("wall_seconds")
            if secs is None:
                line += f" | {'—':>12s} {'—':>7s} {'—':>8s}"
                continue
            speedup = f"{ref / secs:.2f}x" if ref and secs else "—"
            if r["precision"] == "fp32":
                speedup = "(base)"
            line += f" | {secs:11.0f}s {rate or 0:7.1f} {speedup:>8s}"
        print(line)

    print()
    print("arm wall includes model load, the dataset load, both cells and the text pass;")
    print("per-cell seconds are wall time around one (dataset, embedder) build.")
    print()
    print("Speedups are WITHIN a card. Across cards is the #3144 effect, not this one:")
    for emb in embedders:
        per_card = {}
        for card, r in base.items():
            secs = (r["cells"].get(emb) or {}).get("wall_seconds")
            if secs:
                per_card[card] = secs
        if len(per_card) == 2:
            (c1, s1), (c2, s2) = sorted(per_card.items(), key=lambda kv: -kv[1])
            print(f"  {emb:12s} fp32 {c1} {s1:.0f}s vs fp32 {c2} {s2:.0f}s -> {s1 / s2:.2f}x from the card alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
