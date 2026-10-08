#!/usr/bin/env python
"""Run one calibration cell under one text-sort line rule (the #3826 trajectory A/B).

    TEXTCUT_ARM=guarded_tail python run_cells_ab_3826.py --index 0

The arm is ``VTSEARCH_TEXT_SORT_CUT``, the same flag the app reads, so a cell
here draws its opening's line exactly as the app would.  When this A/B ran the
line was not output-only: Autopilot's Bad phase voted the unlabelled items
nearest it (``al_strategies._pick_bad_phase``), so the two arms voted on
different items from the first Bad pick onward, and only two whole runs could
price that.  Since #4136 the flag draws the *display* line only and the Bad
phase samples at the midpoint under both arms, so a re-run of this A/B
measures no trajectory difference by construction
(``tests_lib/sorting/test_text_sort_threshold.py`` pins the opening's picks
against the rule).

The wrapper refuses to run unless the rule the thresholds module actually
resolved is the arm named.  A flag that failed to reach the module would
produce two identical grids and a clean, wrong "no difference".
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "calibration"))

arm = os.environ.get("TEXTCUT_ARM", "")
if arm not in ("gmm_midpoint", "guarded_tail"):
    raise SystemExit(f"TEXTCUT_ARM={arm!r} is not gmm_midpoint or guarded_tail")
# Set before anything imports the thresholds package, which reads it at import.
os.environ["VTSEARCH_TEXT_SORT_CUT"] = arm

import common  # noqa: E402

common.setup_env()


def main(argv: "list[str] | None" = None) -> int:
    from vtscore.training.thresholds import gmm as G  # noqa: PLC0415

    if G.TEXT_SORT_CUT_RULE != arm:
        raise SystemExit(f"TEXTCUT_ARM={arm} but the thresholds module resolved {G.TEXT_SORT_CUT_RULE!r}")
    print(f"TEXTCUT_ARM={arm}: VTSEARCH_TEXT_SORT_CUT resolved to {G.TEXT_SORT_CUT_RULE}", flush=True)

    import run_cells  # noqa: PLC0415

    return run_cells.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
