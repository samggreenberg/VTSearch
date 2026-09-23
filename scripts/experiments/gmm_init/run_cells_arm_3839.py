#!/usr/bin/env python
"""Run one cell with a named #3839 anchored-refit rule installed (the trajectory A/B).

    ANCHORED_RULE_ARM=cap2000 python run_cells_arm_3839.py --index 0

``run_cells_arm_3825.py`` with ``arms_3839`` in place of ``arms_3825``.  The
A/B pairs this against a grid of the *shipped* rule on the same code and the
same cells - #3840's validation grid ``v_ll1e-8`` is exactly that, so no
second baseline grid is run.  A raised cap only changes folds that reach 200
iterations, so most cells should pair to a Δ of exactly zero; that share is
the check that the two grids are the same harness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3839 as A  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    arm = os.environ.get("ANCHORED_RULE_ARM", "shipped")
    if arm not in A.ARMS:
        print(f"ANCHORED_RULE_ARM={arm!r} is not one of {sorted(A.ARMS)}", file=sys.stderr)
        return 2

    import run_cells  # noqa: PLC0415

    rule, desc = A.ARMS[arm]
    with A.swap_anchored_stop(rule):
        A.assert_swap_installed()
        print(f"ANCHORED_RULE_ARM={arm}: {desc}", flush=True)
        return run_cells.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
