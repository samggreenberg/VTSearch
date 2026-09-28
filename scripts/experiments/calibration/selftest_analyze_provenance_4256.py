#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_provenance_4256.py``.

Plants precision frames whose calibration votes are named (``fold_cal_vote``)
beside a pick log, and checks the three mechanics every verdict rests on:

* **the time split** - "learned" evidence keeps exactly the calibration votes
  picked at or after the first learned-phase pick, and drops the rest, even
  when a learned-looking phase name (``good``) appears before it;
* **the consistent pool** - removes exactly one reference entry per voted item;
* **no-op when there is no opening** - when every vote was picked by a learned
  phase, "learned" and "all" give identical rows.

    python selftest_analyze_provenance_4256.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_provenance_4256 as A  # noqa: E402
import selftest_analyze_pframes_4220 as F  # noqa: E402

N_VOTES = 60
FIRST_LEARNED_T = 11  # votes 1..10 are the opening


def build(cells: Path, idx: int, seed: int, opening: bool) -> None:
    F.build_cell(cells, idx, seed=seed, pool_prev=0.05, test_prev=0.05, n_votes=N_VOTES)
    z = np.load(cells / f"task_{idx:04d}__pframes.npz")
    packed = {k: z[k] for k in z.files}
    # Vote ids 0..N-1, picked at t = id + 1; the calibration votes are the frame's votes.
    packed["t150/fold_cal_vote"] = np.arange(N_VOTES, dtype=np.int64)
    packed["t150/fold_cal_phase"] = np.array(["x"] * N_VOTES)
    np.savez_compressed(cells / f"task_{idx:04d}__pframes.npz", **packed)
    t = np.arange(1, N_VOTES + 1)
    phase = np.where(t < FIRST_LEARNED_T, "good", "hard") if opening else np.full(N_VOTES, "hard")
    pd.DataFrame({"t": t, "phase": phase, "picked_id": t - 1}).to_csv(cells / f"task_{idx:04d}__picks.csv", index=False)


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    root = Path(tempfile.mkdtemp(prefix="selftest-provenance-4256-"))
    with_opening = root / "open" / "cells"
    no_opening = root / "none" / "cells"
    for d in (with_opening, no_opening):
        d.mkdir(parents=True)
    build(with_opening, 0, seed=1, opening=True)
    build(no_opening, 0, seed=1, opening=False)
    out = root / "out"
    rc = A.main(["--arm", f"open={root / 'open'}", "--arm", f"none={root / 'none'}", "--out", str(out)])
    check(rc == 0, "the analyzer runs on planted frames", failures)
    rows = pd.read_csv(out / "provenance_rows.csv.gz")

    z = np.load(with_opening / "task_0000__pframes.npz")
    labels = z["t150/fold_cal_labels"]
    want_learned_pos = int(labels[FIRST_LEARNED_T - 1 :].sum())
    got = rows[(rows.arm == "open") & (rows.evidence == "learned")]["n_cal_pos"].iloc[0]
    check(
        got == want_learned_pos, f"learned evidence keeps votes from t >= {FIRST_LEARNED_T} ({got} positives)", failures
    )
    n_open = rows[(rows.arm == "open") & (rows.evidence == "all")]["n_cal_opening"].iloc[0]
    check(n_open == FIRST_LEARNED_T - 1, f"the opening counts exactly its {FIRST_LEARNED_T - 1} votes", failures)

    fr = {k.split("/", 1)[1]: z[k] for k in z.files}
    shipped = A.reference_pool(fr, "shipped")
    consistent = A.reference_pool(fr, "consistent")
    check(len(shipped) - len(consistent) == N_VOTES, "the consistent pool drops one entry per voted item", failures)

    none = rows[rows.arm == "none"]
    cols = ["pool", "X", "status", "returned", "n_cal_pos"]
    a = none[none.evidence == "all"][cols].reset_index(drop=True)
    b = none[none.evidence == "learned"][cols].reset_index(drop=True)
    check(a.equals(b), "with no opening, learned evidence equals all evidence", failures)

    print()
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
