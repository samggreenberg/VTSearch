"""#4930: tabulate probe_numpy_pins.py's output, one row per (numpy, pin).

For each: whether numpy warned about the pin, which dispatch targets above AVX2
stayed on (per CPU model), which ufunc targets ran, and which ops still differ
between nodes.  An op "differs" when its hash is not the same on every node probed.

    python collate_numpy_pins.py --probes /expscratch/$USER/det-4930/probes [--out table.md] [--csv pins.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# What an AVX2-only node (EPYC 7713, Xeon E5-2698 v4) can dispatch to.
AVX2_LEVEL = {"SSSE3", "SSE41", "POPCNT", "SSE42", "AVX", "F16C", "FMA3", "AVX2", "X86_V2", "X86_V3"}
PIN_ORDER = ["base", "old", "old_icl", "old_all", "v4", "v4_icl_spr"]


def short_cpu(cpu: str) -> str:
    for key in ("Silver 4114", "Gold 5218R", "E5-2698 v4", "EPYC 7713", "EPYC 9534"):
        if key in cpu:
            return key
    return cpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--csv", type=Path, help="the same table, one row per (numpy, pin)")
    args = ap.parse_args()

    runs = [json.loads(p.read_text()) for p in sorted(args.probes.glob("*__*__*.json"))]
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        _, pin = r["tag"].split("__", 1)
        by_key[(r["np"], pin)].append(r)

    lines = [
        f"{len({r['node'] for r in runs})} nodes: "
        + ", ".join(sorted({f"{r['node']} ({short_cpu(r['cpu'])})" for r in runs})),
        "",
        "| numpy | pin | warns | above-AVX2 targets still on (CPU: targets) | ufunc targets run | ops differing across nodes |",
        "|---|---|---|---|---|---|",
    ]
    records = []
    for np_ver in sorted({k[0] for k in by_key}, key=lambda v: tuple(int(x) for x in v.split("."))):
        for pin in PIN_ORDER:
            rs = by_key.get((np_ver, pin))
            if not rs:
                continue
            warns = "yes" if any(r["warnings"] for r in rs) else "no"
            on: dict[str, set[str]] = defaultdict(set)
            for r in rs:
                extra = [t for t in r["dispatch_on"] if t not in AVX2_LEVEL]
                if extra:
                    on[short_cpu(r["cpu"])].add(" ".join(extra))
            on_txt = "; ".join(f"{c}: {', '.join(sorted(v))}" for c, v in sorted(on.items())) or "none"
            ran = sorted({t for r in rs for t in r["ufunc_current"]})
            ops = sorted(rs[0]["hash"])
            differ = [op for op in ops if len({r["hash"][op] for r in rs}) > 1]
            lines.append(
                f"| {np_ver} | `{pin}` | {warns} | {on_txt} | {', '.join(ran)} | {', '.join(differ) or '**none**'} |"
            )
            records.append(
                {
                    "numpy": np_ver,
                    "pin": pin,
                    "value": rs[0]["pin"] if rs[0]["pin"] is not None else "",
                    "nodes": len(rs),
                    "warns": warns,
                    "above_avx2_on": on_txt,
                    "ufunc_targets_run": " ".join(ran),
                    "ops_differing": " ".join(differ),
                }
            )
    lines += ["", "Pins:"]
    seen = {}
    for r in runs:
        seen.setdefault(r["tag"].split("__", 1)[1], r["pin"])
    lines += [f"- `{k}`: `{seen[k]}`" for k in PIN_ORDER if k in seen]
    lines += ["", "Dispatch targets by numpy:"]
    for np_ver in sorted({r["np"] for r in runs}, key=lambda v: tuple(int(x) for x in v.split("."))):
        r = next(x for x in runs if x["np"] == np_ver)
        lines.append(f"- {np_ver}: baseline {' '.join(r['baseline'])}; dispatch {' '.join(r['dispatch'])}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)


if __name__ == "__main__":
    main()
