"""#5084: the head's count multiplier at a balance of 1 and below, closed loop, as curves over the click a user has reached.

Reads ``head_bulk_5084.sh``'s arms (Binary Photo to 200 clicks, Region Photo to 150, each plus the closing check,
paired by cell and seed), which differ only in the head's count line at beta <= 1:

* ``ctl``: the app, ``beta ** 0.708`` times what stands out;
* ``s71e1``: ``0.71 * beta`` times what stands out;
* ``s50e1``: ``0.5 * beta``.

The scoring, the pairing check, Done and the splits are ``head_count_5039_ab.py``'s (its ``per_run``,
``departure``, ``analyse_beta``), run per path on that path's embedder and horizon.

    python head_bulk_5084_ab.py --root <state-of-the-app dir> --date 2026-10-23 --seeds 5 --path binary \\
        --baseline <text_baseline.csv> --out <dir> [--docs <report dir>/tables] [--tag smoke] [--betas 0.25,1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_count_5039_ab as hc  # noqa: E402
from early_check_4664 import STEP_COLS, _headline, fbeta  # noqa: E402

ARMS = ("ctl", "s71e1", "s50e1")
NAME = {
    "ctl": "the app (beta ** 0.708 x the count)",
    "s71e1": "0.71 beta x the count",
    "s50e1": "0.5 beta x the count",
}
COLOR = {"ctl": "#8a8985", "s71e1": "#2a78d6", "s50e1": "#eb6834"}
EMBEDDER = {"binary": "siglip", "region": "siglip+dinov3_patch"}
HORIZON = {"binary": 200, "region": 100}  # Region runs 100 clicks on the cpu partition (the V100s are taken)
_EMB = "siglip"


def read_cell(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``head_count_5039_ab.read_cell`` on this path's embedder."""
    head = pd.read_csv(path, nrows=0).columns
    m = pd.read_csv(path, usecols=[c for c in STEP_COLS if c in head], low_memory=False)
    m = _headline(m[m["embedder"] == _EMB])
    stem = path[: -len(".csv.gz")] if path.endswith(".gz") else path[: -len(".csv")]
    ext = ".csv.gz" if path.endswith(".gz") else ".csv"
    try:
        pk = pd.read_csv(f"{stem}__picks{ext}", usecols=lambda c: c in hc.PICK_COLS, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pk = pd.DataFrame(columns=hc.PICK_COLS)
    return m, pk[pk["embedder"] == _EMB]


def set_path(path: str) -> None:
    """Point #5039's analysis at *path*: its embedder, its horizon, this study's arms."""
    global _EMB
    _EMB = EMBEDDER[path]
    hc.T = HORIZON[path]
    hc.GRID_T = np.arange(hc.T + 1)
    hc.CLICKS = tuple(c for c in (10, 25, 35, 50, 75, 100, 125, 150, 175, 200) if c <= hc.T)
    hc.AREAS = tuple((0, hi) for hi in (25, 50, 100, 150, 200) if hi <= hc.T)
    hc.ARMS, hc.NAME, hc.COLOR = ARMS, NAME, COLOR
    hc.read_cell = read_cell


def run_dir(root: Path, date: str, tag: str, arm: str, path: str, beta: float) -> Path:
    return root / f"{date}-headbulk5084{tag}-{arm}-{path}-b{hc.btag(beta)}" / "results" / "cells"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-23")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--path", required=True, choices=sorted(EMBEDDER))
    ap.add_argument("--betas", default="0.25,1")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path, help="the report's tables dir; figures go beside it")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--arms", default=",".join(ARMS), help="the arms to read; the control is always one")
    a = ap.parse_args()
    set_path(a.path)
    arms = [x for x in ARMS if x in {"ctl", *a.arms.split(",")}]
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == _EMB) & (base["seed"] < a.seeds)]
    res: dict[float, dict] = {}
    counts = {}
    for beta in (float(b) for b in a.betas.split(",")):
        t = hc.btag(beta)
        tq = {f"{c}|{int(s)}": float(fbeta([p], [r], beta)[0]) for c, s, p, r in
              zip(base["category"], base["seed"], base[f"text_line_precision_b{t}"], base[f"text_line_recall_b{t}"])}  # fmt: skip
        runs = {}
        for arm in arms:
            steps, picks = hc.load_arm(run_dir(a.root, a.date, a.tag, arm, a.path, beta), a.seeds, a.procs)
            runs[arm] = hc.per_run(steps, picks, tq, beta)
            counts[f"{arm}-{a.path}-b{t}"] = len(runs[arm])
            print(f"{a.path} beta {beta:g} {arm}: {len(runs[arm])} runs", flush=True)
        if min(len(r) for r in runs.values()) == 0:
            print(f"{a.path} beta {beta:g}: an arm has no runs yet; skipped")
            continue
        res[beta] = hc.analyse_beta(runs, beta)
    if not res:
        return 1
    stem = f"ab_{a.path}"
    for o in [a.out] + ([a.docs] if a.docs else []):
        o.mkdir(parents=True, exist_ok=True)
        for name in ("summary", "splits", "examples", "curves", "gains"):
            pd.concat([r[name] for r in res.values()]).to_csv(o / f"{stem}_{name}.csv", index=False)
        (o / f"{stem}_counts.json").write_text(json.dumps(counts, indent=1))
    for o in [a.out] + ([a.docs.parent / "figures"] if a.docs else []):
        o.mkdir(parents=True, exist_ok=True)
        hc.figure(res, o / f"{stem}_over_clicks.png")
    s = pd.concat([r["summary"] for r in res.values()])
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(s[["beta", "arm", "runs", "departed", "departure_median", "departed_on_the_line", "departed_on_a_phase",
                 "departed_otherwise", "f_prefix_mismatch"]
                + [f"meanF_gain_{lo}_{hi}" for lo, hi in hc.AREAS] + [f"meanF_se_{lo}_{hi}" for lo, hi in hc.AREAS]
                + ["after", "after_gain", "after_se"]].round(4).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm"] + [f"gain_at_{c}" for c in hc.CLICKS]].round(4).to_string(index=False))
        print(s[["beta", "arm", "check_clicks", "done_reached", "done_median", "done_shift_mean", "stop_clicks_gain",
                 "stop_f", "stop_f_gain", "stop_f_se"]].round(4).to_string(index=False))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
