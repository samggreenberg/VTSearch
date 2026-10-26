"""#5215: the head's count line gated on how far the head's top stands out, closed loop on Binary Photo.

Reads ``head_tail_5215.sh``'s arms (200 clicks plus the closing check, paired by cell and seed), which differ only in
the head's count line at beta <= 1:

* ``ctl``: the app, 0.71 beta times what stands out on every head;
* ``tz61``: #4603's ``beta ** 0.708`` where the top logit stands under 6.1 bulk spreads.

The scoring and the tables are ``head_bulk_5084_ab.py``'s, pointed at this study's run dirs.

    python head_tail_5215_ab.py --root <state-of-the-app dir> --date 2026-10-25 --seeds 5 --path binary \\
        --baseline <text_baseline.csv> --out <dir> [--docs <report dir>/tables] [--tag smoke] [--betas 0.25,1]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_bulk_5084_ab as hb  # noqa: E402
import head_count_5039_ab as hc  # noqa: E402

hb.ARMS = ("ctl", "tz61")
hb.NAME = {"ctl": "the app (0.71 beta x the count)", "tz61": "beta ** 0.708 where the top is under 6.1 spreads"}
hb.COLOR = {"ctl": "#8a8985", "tz61": "#2a78d6"}


def run_dir(root: Path, date: str, tag: str, arm: str, path: str, beta: float) -> Path:
    return root / f"{date}-headtail5215{tag}-{arm}-{path}-b{hc.btag(beta)}" / "results" / "cells"


hb.run_dir = run_dir

if __name__ == "__main__":
    raise SystemExit(hb.main())
