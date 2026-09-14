#!/usr/bin/env python
"""Run one cell with a named unanchored-GMM fit installed (#3585, the A/B).

    GMM_FIT_ARM=baseline python run_cells_arm_3585.py --index 0 --outdir <dir>

The gate (``gate_3585.py``) replays candidate fits over captured inputs, which
holds everything else fixed - the same haystacks, the same votes.  That is the
right instrument for "does the admitted set change", and the wrong one for
"does the detector end up better or worse", because the threshold is **not**
output-only: Autopilot's Hard phase picks the unlabelled item nearest the
decision threshold, so a run whose threshold moves is asked to vote on
different items from its second Hard pick onward, and the two trajectories
diverge.  Only two whole runs can measure that.

So this is the trajectory arm: an ordinary cell, with one function swapped for
the whole run.  Pair the two output dirs with ``analyze_ab.py``
(``CALIB_AB_ON`` = the native run, ``CALIB_AB_OFF`` = the sklearn one).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3585 as A  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    arm = os.environ.get("GMM_FIT_ARM", "native")
    if arm not in A.ARMS:
        print(f"GMM_FIT_ARM={arm!r} is not one of {sorted(A.ARMS)}", file=sys.stderr)
        return 2

    import run_cells  # noqa: PLC0415

    fit_fn = A.ARMS[arm][0]
    with A.swap_fit(fit_fn):
        # Assert the swap took before the cell spends an hour measuring the
        # wrong arm.  A patched-name study that silently ran the default twice
        # produces two identical result dirs and a clean, wrong "no difference".
        from vtscore.training.thresholds import fit_score_gmm  # noqa: PLC0415

        assert fit_score_gmm is fit_fn, "the fit swap did not reach the package binding"
        print(f"GMM_FIT_ARM={arm}: {A.ARMS[arm][1]}", flush=True)
        return run_cells.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
