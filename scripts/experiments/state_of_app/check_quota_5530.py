#!/usr/bin/env python3
"""#5530: does any session's first prompted spot check start before its labels meet the quota?

The 2026-11-03 Binary Photo review (#5525) found the first prompted check at a median of click 15, inside #4736's
30-label hold, before the app showed a detector. #5556 then set the hold to 0, so a typed-query detector gives the
trained head from the label quota (3 Goods and 4 Bads, or a Good and 16 Bads). This reads every session's pick log
(``results/cells/task_NNNN__picks.csv.gz``: one row a pick, ``check_kind`` ``weak`` / ``done`` / ``early`` on a
prompted check's picks) and reports, per run, where the first prompted check pick falls against the first pick at
which the counts meet the quota. The hold changes what Find shows, never what Autopilot asks (#4736, #5556), so the
review's picks are the hold-0 app's.

  check_quota_5530.py RESULTS_CELLS OUT.csv [PROCS]

Writes one row a session (``quota_t``, ``check_t``, ``kind``, ``phase`` and the counts at the check's first pick;
-1 where none) and prints the summary the report quotes.
"""

from __future__ import annotations

import glob
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

COLS = ["seed", "category", "t", "phase", "check_kind", "n_good", "n_bad"]
PROMPTED = ["weak", "done", "early"]


def meets_quota(n_good, n_bad):
    """``label_quota.LabelQuota.tier == "trained"``, on columns: 3 Goods and 4 Bads, or a Good and 16 Bads."""
    return ((n_good >= 3) & (n_bad >= 4)) | ((n_good >= 1) & (n_bad >= 16))


def one(path: str) -> list[tuple]:
    df = pd.read_csv(path, usecols=COLS)
    out = []
    for (seed, cat), g in df.groupby(["seed", "category"], sort=False):
        g = g.sort_values("t", kind="stable")
        q = g[meets_quota(g.n_good, g.n_bad)]
        q_t = int(q.t.iloc[0]) if len(q) else -1
        c = g[g.check_kind.isin(PROMPTED)]
        if not len(c):
            out.append((seed, cat, q_t, -1, "", "", -1, -1))
            continue
        r = c.iloc[0]
        out.append((seed, cat, q_t, int(r.t), r.check_kind, r.phase, int(r.n_good), int(r.n_bad)))
    return out


def main() -> None:
    root, out = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    files = sorted(glob.glob(f"{root}/task_*__picks.csv.gz"))
    rows: list[tuple] = []
    with ProcessPoolExecutor(procs) as ex:
        for r in ex.map(one, files, chunksize=16):
            rows.extend(r)
    d = pd.DataFrame(rows, columns=["seed", "category", "quota_t", "check_t", "kind", "phase", "n_good", "n_bad"])
    d.to_csv(out, index=False)
    c = d[d.check_t >= 0]
    labels = c.n_good + c.n_bad
    print(root)
    print(f"  sessions {len(d)}, prompted {len(c)} ({len(c) / len(d):.0%}), kinds {c.kind.value_counts().to_dict()}")
    before = int(((c.check_t < c.quota_t) | (c.quota_t < 0)).sum())
    print(
        f"  first check pick before the quota: {before}; under the quota at it: {int((~meets_quota(c.n_good, c.n_bad)).sum())}"
    )
    print(f"  phase at it: {c.phase.value_counts().to_dict()}")
    print(
        f"  first check click quartiles {c.check_t.quantile([0.25, 0.5, 0.75]).tolist()};"
        f" quota click quartiles {d[d.quota_t >= 0].quota_t.quantile([0.25, 0.5, 0.75]).tolist()};"
        f" never met {int((d.quota_t < 0).sum())}"
    )
    print(
        f"  clicks from the quota to the check: min {int((c.check_t - c.quota_t).min())}, median {(c.check_t - c.quota_t).median():.0f}"
    )
    print(
        f"  labels at first check under 30 (inside #4736's hold): {int((labels < 30).sum())} ({(labels < 30).mean():.0%})"
    )


if __name__ == "__main__":
    main()
