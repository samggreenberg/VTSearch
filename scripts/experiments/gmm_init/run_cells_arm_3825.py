#!/usr/bin/env python
"""Run one cell with a named anchored-EM stopping rule installed (#3825, the A/B).

    ANCHORED_STOP_ARM=ll1e-3 python run_cells_arm_3825.py --index 0

``gate_3825.py`` replays the arms over captured inputs, which holds everything
else fixed - the same haystacks, the same votes, the same folds.  That is the
right instrument for "does the admitted set change" and the wrong one for "does
the detector end up better or worse", because the threshold is not output-only:
Autopilot's Hard phase picks the unlabelled item nearest the decision
threshold, so a run whose threshold moves is asked to vote on different items
from its second Hard pick onward and the two trajectories diverge.  Only two
whole runs can price that, and on this issue it matters more than it did on
#3585 - the gate says the admitted set moves on nearly every case, so the
divergence is not hypothetical.

Pair the output dirs with ``analyze_ab.py`` (``CALIB_AB_ON`` = the candidate,
``CALIB_AB_OFF`` = the shipped rule).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3825 as A  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    arm = os.environ.get("ANCHORED_STOP_ARM", "baseline")
    if arm not in A.ARMS:
        print(f"ANCHORED_STOP_ARM={arm!r} is not one of {sorted(A.ARMS)}", file=sys.stderr)
        return 2

    import run_cells  # noqa: PLC0415

    stop, desc = A.ARMS[arm]
    with A.swap_anchored_stop(stop):
        # Assert the swap took before the cell spends an hour measuring the
        # wrong arm.  A patched-name study that silently ran the default twice
        # produces two identical result dirs and a clean, wrong "no difference".
        A.assert_swap_installed()
        print(f"ANCHORED_STOP_ARM={arm}: {desc}", flush=True)
        return run_cells.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
