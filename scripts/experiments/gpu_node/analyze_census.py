"""#3160: read the per-node census and say how many *devices* each type label covers.

    python analyze_census.py --census <dir> [--reference rack7n03] [--csv]

Three tables, in the order the argument needs them:

1. **What the label covers** -- every node grouped by the device torch reports,
   with SM count and driver.  This is the fact `--gres` cannot express.
2. **Do the vectors agree** -- median 1-cos against a reference node, per
   embedder.  The reference defaults to the node that built the published pile
   cell, because "does this reproduce the pile" is the question a rebuild asks.
3. **Which op moves** -- the bare GEMM / conv / attention fingerprints, grouped.
   If nodes that disagree on the tower already disagree on a plain GEMM, the
   cause is arithmetic order, not a model-level backend choice.

Nodes whose job failed are listed as missing rather than dropped: a census with
holes in it is a different claim from a census without.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def sig2(x: float) -> str:
    if x == 0:
        return "0"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.2g}"


def load(census: Path) -> dict[str, dict]:
    out = {}
    for d in sorted(census.iterdir()):
        f = d / "device.json"
        if f.is_file():
            try:
                out[d.name] = json.loads(f.read_text())
            except json.JSONDecodeError as exc:
                log(f"  unreadable {f}: {exc}")
    return out


def vectors(census: Path, node: str, embedder: str) -> np.ndarray | None:
    p = census / node / f"vectors_{embedder}.npy"
    if not p.is_file():
        return None
    v = np.asarray(np.load(p), dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", required=True)
    ap.add_argument("--reference", default="rack7n03", help="node the drift is measured against")
    ap.add_argument("--embedders", default="siglip,siglip2_l")
    ap.add_argument("--csv", default="", help="write the per-node table here")
    args = ap.parse_args(argv)

    census = Path(args.census)
    recs = load(census)
    if not recs:
        raise SystemExit(f"no device.json under {census}")
    embedders = [e for e in args.embedders.split(",") if e]

    # ---- 1. what the type label covers ------------------------------------
    by_device: dict[tuple, list[str]] = defaultdict(list)
    for node, r in recs.items():
        d = r.get("device", r)
        by_device[(d.get("gpu_name"), d.get("gpu_capability"), d.get("multi_processor_count"), d.get("driver"))].append(
            node
        )

    log("=== devices behind the type labels ===")
    log(f"{'device':<28} {'cap':<7} {'SMs':>4} {'driver':<10} {'n':>3}  nodes")
    for (name, cap, sms, driver), nodes in sorted(by_device.items(), key=lambda kv: str(kv[0][0])):
        log(f"{str(name):<28} {str(cap):<7} {str(sms):>4} {str(driver):<10} {len(nodes):>3}  {' '.join(sorted(nodes))}")

    # ---- 2. vector agreement ----------------------------------------------
    ref = args.reference
    if ref not in recs:
        raise SystemExit(f"reference node {ref} is not in the census ({', '.join(sorted(recs))})")
    log(f"\n=== median 1-cos vs {ref} ({recs[ref].get('device', {}).get('gpu_name')}) ===")
    refs = {e: vectors(census, ref, e) for e in embedders}
    rows = []
    header = f"{'node':<12} {'device':<26} {'SMs':>4} " + " ".join(f"{e:>12}" for e in embedders) + "   ops"
    log(header)
    for node in sorted(recs):
        d = recs[node].get("device", {})
        cells, drifts = [], {}
        for e in embedders:
            a, b = refs[e], vectors(census, node, e)
            if a is None or b is None or a.shape != b.shape:
                cells.append(f"{'--':>12}")
                continue
            med = float(np.median(1.0 - (a * b).sum(1)))
            drifts[e] = med
            cells.append(f"{sig2(med):>12}")
        ops = recs[node].get("ops", {})
        ops_tag = ",".join(sorted({v.get("sha256", "?")[:6] for v in ops.values()})) if ops else "-"
        log(
            f"{node:<12} {str(d.get('gpu_name')):<26} {str(d.get('multi_processor_count')):>4} "
            + " ".join(cells)
            + f"   {ops_tag}"
        )
        rows.append(
            {
                "node": node,
                "gpu_name": d.get("gpu_name"),
                "capability": d.get("gpu_capability"),
                "sms": d.get("multi_processor_count"),
                "driver": d.get("driver"),
                **{f"drift_{e}": drifts.get(e) for e in embedders},
            }
        )

    # ---- 3. which bare op moves -------------------------------------------
    log("\n=== bare-op fingerprints (nodes sharing a hash agree bit-for-bit) ===")
    op_names = sorted({k for r in recs.values() for k in (r.get("ops") or {})})
    for op in op_names:
        groups: dict[str, list[str]] = defaultdict(list)
        for node, r in recs.items():
            h = (r.get("ops") or {}).get(op, {}).get("sha256")
            if h:
                groups[h].append(node)
        log(f"  {op:<24} {len(groups)} distinct result(s)")
        for h, nodes in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            log(f"    {h}  {len(nodes):>2} node(s): {' '.join(sorted(nodes))}")

    if args.csv:
        import csv

        with Path(args.csv).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"\nwrote {args.csv}")

    # A census with holes is a different claim from one without.
    log(f"\n{len(recs)} nodes reporting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
