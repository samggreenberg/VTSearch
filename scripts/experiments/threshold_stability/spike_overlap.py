"""Do the same media items cause spikes across two runs? (#2790 robustness)

Aggregates each run's spikes per (class, image_id) and, for the classes both runs
cover, measures how much the **offender sets overlap** — i.e. whether the images that
stress the cut are a property of the *image* (shared across a different proposal path
or a different region embedder) or an artifact of one setup.

For each shared class it reports, over items that spiked in >= ``--min-spikes`` seeds
in either run: the offenders unique to A, unique to B, shared by both, and the Jaccard.
A high shared fraction => the hard cases are intrinsic to the images.

Usage: ``python spike_overlap.py --a <trace_root> --b <trace_root>
        [--a-name whole] [--b-name hac-dinov3] [--min-spikes 3]``
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from spike_analysis import collect_spikes


def _per_item(root: Path, thresh: float) -> dict[str, dict[int, int]]:
    """(class -> {image_id -> n_spikes}) for a trace root."""
    out: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in collect_spikes(root, thresh):
        out[r["cls"]][int(r["culprit_id"])] += 1
    return {c: dict(m) for c, m in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spike-item overlap across two runs (#2790).")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--min-spikes", type=int, default=3, help="offender = spiked in >= this many seeds")
    args = ap.parse_args(argv)

    A = _per_item(Path(args.a), args.thresh)
    B = _per_item(Path(args.b), args.thresh)
    shared_classes = sorted(set(A) & set(B))
    k = args.min_spikes

    print(f"Offender = an image that spiked in >= {k} seeds. Runs: {args.a_name} vs {args.b_name}.")
    print(
        f"{'class':<16} {args.a_name + '_off':>10} {args.b_name + '_off':>10} {'shared':>7} {'jaccard':>8}  shared image ids"
    )
    tot_a = tot_b = tot_shared = tot_union = 0
    for c in shared_classes:
        oa = {i for i, n in A[c].items() if n >= k}
        ob = {i for i, n in B[c].items() if n >= k}
        if not (oa or ob):
            continue
        sh = oa & ob
        un = oa | ob
        tot_a += len(oa)
        tot_b += len(ob)
        tot_shared += len(sh)
        tot_union += len(un)
        j = len(sh) / len(un) if un else 0.0
        ids = " ".join(str(i) for i in sorted(sh)[:6])
        print(f"{c:<16} {len(oa):>10} {len(ob):>10} {len(sh):>7} {j:>8.2f}  {ids}")
    jac = tot_shared / tot_union if tot_union else 0.0
    print(
        f"\nTOTAL offenders: {args.a_name}={tot_a}  {args.b_name}={tot_b}  "
        f"shared={tot_shared}  overall Jaccard={jac:.2f}  "
        f"(of {args.a_name}'s offenders, {tot_shared}/{tot_a or 1} = {tot_shared / (tot_a or 1):.0%} also offend in {args.b_name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
