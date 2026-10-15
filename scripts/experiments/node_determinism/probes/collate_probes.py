"""Collate #4110's probe outputs into the report's tables.

    python collate_probes.py --ops DIR.. --mkl DIR.. --text DIR.. --cost DIR.. --out DIR

Each DIR holds ``<node>__<variant>.json`` files from one probe (``probes.sh``).
Writes ``ops_by_cpu.csv`` (per probe, variant and op: how many different
outputs the CPU models gave, and which models agreed), ``text.csv`` and
``cost.csv``.  CPU models come from the ops probe, which records them.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd


def _load(dirs: list[Path]) -> dict[tuple[str, str], dict]:
    out = {}
    for d in dirs:
        for f in sorted(d.glob("*__*.json")):
            j = json.loads(f.read_text())
            out[(j["node"].split(".")[0], j["variant"])] = j
    return out


def _short(cpu: str) -> str:
    for word in ("Silver 4114", "Gold 5218R", "E5-2698 v4", "EPYC 7713", "EPYC 9534"):
        if word in cpu:
            return word
    return cpu


def ops_table(runs: dict, probe: str, cpu: dict[str, str]) -> list[dict]:
    rows = []
    for variant in sorted({v for _, v in runs}):
        nodes = sorted(n for n, v in runs if v == variant)
        keys = sorted({k for n in nodes for k in runs[(n, variant)]["r"]})
        for k in keys:
            groups = collections.defaultdict(set)
            for n in nodes:
                groups[str(runs[(n, variant)]["r"].get(k))].add(_short(cpu.get(n, n)))
            rows.append(
                {
                    "probe": probe,
                    "variant": variant,
                    "op": k,
                    "cpu_models": len({_short(cpu.get(n, n)) for n in nodes}),
                    "distinct_outputs": len(groups),
                    "agree": " | ".join(sorted(", ".join(sorted(g)) for g in groups.values())),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for kind in ("ops", "mkl", "text", "cost"):
        ap.add_argument(f"--{kind}", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ops = _load(args.ops)
    cpu = {n: j["cpu"] for (n, _), j in ops.items() if "cpu" in j}
    rows = (
        ops_table(ops, "ops", cpu) + ops_table(_load(args.mkl), "mkl", cpu) + ops_table(_load(args.text), "text", cpu)
    )
    pd.DataFrame(rows).to_csv(args.out / "ops_by_cpu.csv", index=False)
    text = [
        {"node": n, "cpu": _short(cpu.get(n, n)), "variant": v, "warm_query_ms": j.get("warm_query_ms")}
        for (n, v), j in sorted(_load(args.text).items())
    ]
    pd.DataFrame(text).to_csv(args.out / "text.csv", index=False)
    cost = [
        {"node": n, "cpu": _short(cpu.get(n, n)), "variant": v} | {k: x for k, x in j.items() if k.endswith("_ms")}
        for (n, v), j in sorted(_load(args.cost).items())
    ]
    pd.DataFrame(cost).to_csv(args.out / "cost.csv", index=False)
    print(f"tables -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
