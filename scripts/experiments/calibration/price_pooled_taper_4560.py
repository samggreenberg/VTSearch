#!/usr/bin/env python3
"""Price a taper on the pooled prior above the line, by replaying #4523's snapshots (#4560).

Since #4539 every band above the line takes its prior from the pooled share
of every pick above it, at one round's weight (``POOLED_WEIGHT``).  That fixed
the line's precision range on big sparse lines, but it pulls the rich top
bands toward the line's average, so *Lean the Threshold*'s farthest-shallow
preset (a beta-4 session's beta-1/4 option, about a quarter of the line)
under-reads its precision by 0.11-0.17 (#4540's report).

This swaps :func:`vtscore.training.thresholds.line_test.pooled_weight` for one
of :data:`TAPERS`, sets the pool's reach (``POOL_RADIUS``, ``--radius``) and
whether the budget waits for the first pass (``FIRST_PASS_BEFORE_BUDGET``,
``--first-pass``), and runs ``analyze_line_test_4523.py`` with the rest of the
command line, so every taper is replayed on the same snapshots, seeds and
grid point.  The worker processes are forked, so they inherit the swap.

    python price_pooled_taper_4560.py --taper linear -- --world ... --out ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_line_test_4523 as analysis  # noqa: E402

from vtscore.training.thresholds import line_test  # noqa: E402

W = line_test.POOLED_WEIGHT


def _flat(depth: int, n_above: int) -> float:
    """Today's prior: one round's weight on every band (#4539)."""
    return W


def _linear(depth: int, n_above: int) -> float:
    """Full weight on the band holding the line, falling linearly to W / n_above on the top band."""
    return W * (n_above - depth) / max(n_above, 1)


def _half(depth: int, n_above: int) -> float:
    """Full weight on the half of the bands nearest the line, one pick's worth on the top half."""
    return W if depth < (n_above + 1) // 2 else 1.0


def _weak(depth: int, n_above: int) -> float:
    """One pick's worth on every band: the pool as a light hint."""
    return 1.0


TAPERS = {"flat": _flat, "linear": _linear, "half": _half, "weak": _weak}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--taper", choices=sorted(TAPERS), required=True)
    ap.add_argument("--radius", type=int, default=None, help="pool only the bands within this many of each band")
    ap.add_argument("--first-pass", action="store_true", help="the budget waits until every band above has a round")
    args = ap.parse_args(argv)
    line_test.pooled_weight = TAPERS[args.taper]
    line_test.POOL_RADIUS = args.radius
    line_test.FIRST_PASS_BEFORE_BUDGET = args.first_pass
    print(f"radius={args.radius} first_pass={args.first_pass}")
    print(f"taper={args.taper}: depth 0 (the line) .. 4 -> {[round(TAPERS[args.taper](d, 5), 2) for d in range(5)]}")
    return analysis.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
