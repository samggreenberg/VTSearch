"""#4938: tabulate openblas_l2.sh's output.

Two questions, per numpy (and so per bundled OpenBLAS):

* across nodes, under ``OPENBLAS_CORETYPE=Haswell`` with and without ``OPENBLAS_L2_SIZE=2048``:
  which ops crash where, and which ops' results are not the same on every node;
* on each node, does adding ``OPENBLAS_L2_SIZE=2048`` change any op's result?  It must not,
  except where it stops the Zen 4 override, or it moves a baseline.

    python collate_openblas_l2.py --probes /expscratch/$USER/det-4938/l2 \
        --np-dir /expscratch/$USER/det-4930/np [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--np-dir", type=Path, required=True, help="openblas_l2.sh's NP_DIR, to name a crash row's numpy")
    ap.add_argument("--csv", type=Path, help="one row per (numpy, L2 setting, op): crashes and distinct results")
    args = ap.parse_args()

    # (numpy, l2, op) -> {node: hash or CRASH(..)}
    res: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for p in sorted(args.probes.glob("*.tsv")):
        for line in p.read_text().splitlines():
            node, np_v, env, op, m, k, digest = line.split("\t")
            np_v = np_v.split("=", 1)[1]
            l2 = env.split("OPENBLAS_L2_SIZE=", 1)[1]
            res[(np_v, l2, f"{op} m={m} k={k}")][node] = digest
    nodes = sorted({n for v in res.values() for n in v})
    # A crash row carries the numpy directory's name ("2.5.x"), a result row numpy's own version.
    for key in [k for k in res if (args.np_dir / f"np-{k[0]}").is_dir()]:
        (info,) = (args.np_dir / f"np-{key[0]}").glob("numpy-*.dist-info")
        version = info.name.removeprefix("numpy-").removesuffix(".dist-info")
        if version != key[0]:
            res[(version, key[1], key[2])].update(res.pop(key))
    rows = []
    for (np_v, l2, op), per in sorted(res.items()):
        crashed = sorted(n for n, d in per.items() if d.startswith("CRASH"))
        ok = {n: d for n, d in per.items() if not d.startswith("CRASH")}
        rows.append(
            {
                "numpy": np_v,
                "l2": l2,
                "op": op,
                "nodes": len(per),
                "crashed_on": " ".join(crashed),
                "distinct_results": len(set(ok.values())),
                "odd_nodes_out": " ".join(_minority(ok)),
            }
        )
    print(f"nodes: {' '.join(nodes)}\n")
    print("| numpy | OPENBLAS_L2_SIZE | ops | ops crashing (nodes) | ops differing across nodes (nodes out) |")
    print("|---|---|---|---|---|")
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["numpy"], r["l2"])].append(r)
    for (np_v, l2), rs in sorted(by.items()):
        crash = [f"{r['op']} ({r['crashed_on']})" for r in rs if r["crashed_on"]]
        differ = [f"{r['op']} ({r['odd_nodes_out']})" for r in rs if r["distinct_results"] > 1]
        print(f"| {np_v} | {l2} | {len(rs)} | {'; '.join(crash) or 'none'} | {'; '.join(differ) or 'none'} |")

    # Same node, same numpy: does the L2 setting change a result?
    moved = defaultdict(list)
    for (np_v, l2, op), per in res.items():
        if l2 != "2048":
            continue
        base = res.get((np_v, "-", op), {})
        for node, d in per.items():
            if node in base and base[node] != d:
                moved[np_v].append(f"{op}@{node}")
    print("\nResults that OPENBLAS_L2_SIZE=2048 changes on the same node:")
    for np_v in sorted({k[0] for k in res}):
        print(f"- numpy {np_v}: {', '.join(sorted(moved[np_v])) or 'none'}")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def _minority(ok: dict[str, str]) -> list[str]:
    """The nodes whose result is not the most common one."""
    counts: dict[str, int] = defaultdict(int)
    for d in ok.values():
        counts[d] += 1
    if len(counts) <= 1:
        return []
    top = max(counts, key=lambda d: counts[d])
    return sorted(n for n, d in ok.items() if d != top)


if __name__ == "__main__":
    main()
