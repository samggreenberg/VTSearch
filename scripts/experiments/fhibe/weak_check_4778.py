#!/usr/bin/env python
"""The weak-separation check on faces, priced: today's sessions against the same sessions without it (#4778).

The face review (#4762) found the check (#4496) prompted in 96-98% of FHIBE face sessions, first at vote 15,
and spending 32-48 of the 150 votes. This prices it as two session sets on the same people, starting photos
and seeds (``sota_face.sh`` under two tags, one with ``CALIB_SPOT_CHECK=end``): today's app, whose prompted
rounds are cast inside the 150 votes, and the same app with the end-of-run check only. The sessions differ
from the first prompt on, so they pair run for run.

Per K x beta (one size):

* the owner's number, the **session mean** of the objective (F-beta of the withheld set at the app's line, at
  the run's own beta) over votes 1-25, 1-50, 51-150 and 1-150, filled as ``by_click.py`` fills it (the
  example sort until the app shows a detector, then the last value carried); arm minus control, mean, SE over
  people, and the share of people the arm wins and loses;
* the objective after the end-of-run check, final AP, Goods found, and failed sessions (final AP below 0.1
  while the person's full-label ceiling reaches 0.5, the #4777 bar); precision, recall and the count Find
  returns at votes 25 and 50, where a check's picks would move the line;
* what the check does in each set, from the pick log: the share of sessions prompted, the first prompted vote,
  the prompted votes cast inside the 150, and the Goods those votes found;
* a drift check: a pair's picks must agree up to the control's first prompt (or to vote 150 when it never
  prompts), since nothing else differs between the sets.

``per_run.csv`` names FHIBE subject ids: it stays in the owner-only analysis directory.

    python weak_check_4778.py --runs <runs> --control <set>::<analysis> --arm <set>::<analysis> \\
        --faces fhibe_faces_roll --size 1024 --out <dir>

``<set>`` is a ``sota_face.sh`` run-set prefix under ``--runs`` (its cells at ``<set>-k<K>-b<tag>``, its
click-0 baseline at ``<set>-baseline``); ``<analysis>`` is ``sota_face_analyze.sh``'s output for it.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "state_of_app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
from _cells_paths import main_frame_files  # noqa: E402
from by_click import RUN, filled_runs, typed_query  # noqa: E402

KS = (1, 4)
BETAS = {0.25: "025", 1.0: "1", 4.0: "4"}
HORIZON = 150
SPANS = ((1, 25), (1, 50), (51, 150), (1, 150))
FAIL_AP, FAIL_CEILING = 0.1, 0.5


def _se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def read_picks(path: str) -> dict | None:
    """One run's pick log: the pick at every vote, and what its prompted checks did inside the horizon.

    A check's round of picks is logged on one row each at the round's last vote (``t``): votes 11-15 all
    read ``t = 15``. So the votes a round cast are ``t - n + 1 .. t`` for its *n* rows.
    """
    try:
        pk = pd.read_csv(
            path, usecols=["dataset", "category", "seed", "t", "phase", "picked_id", "picked_label", "n_good"]
        )
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if pk.empty:
        return None
    pk["phase"] = pk["phase"].fillna("").astype(str)
    inside = pk[pk["t"] <= HORIZON].copy()
    inside["vote"] = inside["t"] - inside.groupby("t")["t"].transform("size") + inside.groupby("t").cumcount() + 1
    prompt = inside[inside["phase"] == "prompt"]
    votes = np.sort(prompt["vote"].to_numpy(dtype=int))
    # A check runs rounds back to back until it ends; a gap between prompted votes starts the next check.
    checks = int(len(votes) > 0) + int((np.diff(votes) > 1).sum()) if len(votes) else 0
    first = int(votes[0]) if len(votes) else None
    goods_before = None
    if first is not None:
        goods_before = int((inside.loc[inside["vote"] < first, "picked_label"] == 1).sum())
    return {
        "dataset": str(pk["dataset"].iloc[0]),
        "category": str(pk["category"].iloc[0]),
        "seed": int(pk["seed"].iloc[0]),
        "prompted": bool(len(votes)),
        "first_prompt": first,
        "prompt_votes": int(len(votes)),
        "prompt_checks": checks,
        "prompt_goods": int((prompt["picked_label"] == 1).sum()),
        "goods_before_prompt": goods_before,
        "prompt_vote_list": votes.tolist(),
        "ids": inside.set_index("vote")["picked_id"].to_dict(),
    }


def load_picks(cells: Path, dataset: str, workers: int) -> dict:
    files = [str(f).replace(".csv", "__picks.csv") for f in main_frame_files(cells)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = [g for g in pool.map(read_picks, files, chunksize=8) if g is not None and g["dataset"] == dataset]
    return {(g["category"], g["seed"]): g for g in got}


def session_set(analysis: Path, baseline: Path, k: int, size: str, beta: float, faces: str):
    """One session set's runs (cells.csv, indexed by run) and their filled objective at every vote 1..150."""
    d = analysis / f"k{k}-b{BETAS[beta]}-{size}"
    cells = pd.read_csv(d / "cells.csv")
    curves = pd.read_csv(d / "curves.csv")
    base = pd.read_csv(baseline / f"example_baseline_k{k}.csv")
    base = base[base["dataset"] == f"{faces}_{size}"]
    filled, _ = filled_runs(curves, cells, typed_query(base, "face", beta))
    filled = filled.reindex(columns=range(1, HORIZON + 1)).ffill(axis=1)
    return cells.set_index(RUN), filled


def first_divergence(a: dict, b: dict) -> int | None:
    """The first vote whose pick differs between two runs' logs, None if they agree to the horizon."""
    for t in range(1, HORIZON + 1):
        if a.get(t) != b.get(t):
            return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--control", required=True, help="SET::ANALYSIS, today's app")
    ap.add_argument("--arm", required=True, help="SET::ANALYSIS, CALIB_SPOT_CHECK=end")
    ap.add_argument("--faces", default="fhibe_faces_roll")
    ap.add_argument("--size", default="1024")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name, spec in (("control", args.control), ("arm", args.arm)):
        s, _, a = spec.partition("::")
        sets[name] = (s, Path(a))
    dataset = f"{args.faces}_{args.size}"
    rows, curves, per_run, goods_at_prompt = [], [], [], []
    for k in KS:
        for beta, tag in BETAS.items():
            key = {"K": k, "beta": beta}
            got = {}
            for name, (s, analysis) in sets.items():
                cells, filled = session_set(analysis, args.runs / f"{s}-baseline", k, args.size, beta, args.faces)
                picks = load_picks(args.runs / f"{s}-k{k}-b{tag}" / "results" / "cells", dataset, args.workers)
                got[name] = (cells, filled, picks)
            (cc, cf, cp), (ac, af, apk) = got["control"], got["arm"]
            runs = sorted(set(cf.index) & set(af.index) & set(cp) & set(apk))
            if not runs:
                print(f"K{k} beta {beta}: no paired runs", file=sys.stderr)
                continue
            C, A = cf.loc[runs].to_numpy(float), af.loc[runs].to_numpy(float)
            row: dict = {
                **key,
                "runs": len(runs),
                "control_only": len(set(cf.index) - set(af.index)),
                "arm_only": len(set(af.index) - set(cf.index)),
            }
            pr = [{**key, "category": r[0], "seed": r[1]} for r in runs]
            for lo, hi in SPANS:
                c, a = C[:, lo - 1 : hi].mean(axis=1), A[:, lo - 1 : hi].mean(axis=1)
                d = a - c
                row[f"ctl_{lo}_{hi}"], row[f"arm_{lo}_{hi}"] = float(c.mean()), float(a.mean())
                row[f"d_{lo}_{hi}"], row[f"d_{lo}_{hi}_se"] = float(d.mean()), _se(d)
                row[f"win_{lo}_{hi}"], row[f"loss_{lo}_{hi}"] = float((d > 1e-9).mean()), float((d < -1e-9).mean())
                for r, v, vc in zip(pr, d, c, strict=True):
                    r[f"d_{lo}_{hi}"], r[f"ctl_{lo}_{hi}"] = float(v), float(vc)
            for col, label in (
                ("thr_fbeta_final", "after_check"),
                ("final_ap", "final_ap"),
                ("positives_found", "goods_found"),
                ("goods_25", "goods_25"),
                ("goods_50", "goods_50"),
                ("thr_precision_25", "precision_25"),
                ("thr_recall_25", "recall_25"),
                ("thr_returned_25", "returned_25"),
                ("thr_precision_50", "precision_50"),
                ("thr_recall_50", "recall_50"),
                ("thr_returned_50", "returned_50"),
                ("check_votes", "end_check_votes"),
            ):
                c, a = cc.loc[runs, col].to_numpy(float), ac.loc[runs, col].to_numpy(float)
                row[f"ctl_{label}"], row[f"arm_{label}"] = float(np.nanmean(c)), float(np.nanmean(a))
                row[f"d_{label}"], row[f"d_{label}_se"] = float(np.nanmean(a - c)), _se(a - c)
                for r, v in zip(pr, a - c, strict=True):
                    r[f"d_{label}"] = float(v)
            ceiling = cc.loc[runs, "ceiling_ap"].to_numpy(float)
            for name, cells_ in (("ctl", cc), ("arm", ac)):
                fap = cells_.loc[runs, "final_ap"].to_numpy(float)
                row[f"{name}_failed"] = int(((fap < FAIL_AP) & (ceiling >= FAIL_CEILING)).sum())
            # What the check does, in each set (the arm's sessions should never prompt).
            for name, picks in (("ctl", cp), ("arm", apk)):
                g = pd.DataFrame(
                    [
                        {
                            kk: picks[r][kk]
                            for kk in ("prompted", "first_prompt", "prompt_votes", "prompt_checks", "prompt_goods")
                        }
                        for r in runs
                    ]
                )
                row[f"{name}_prompted"] = float(g["prompted"].mean())
                row[f"{name}_first_prompt_median"] = float(g["first_prompt"].median()) if g["prompted"].any() else None
                row[f"{name}_prompt_votes_mean"] = float(g["prompt_votes"].mean())
                row[f"{name}_prompt_votes_median"] = float(g["prompt_votes"].median())
                row[f"{name}_prompt_checks_mean"] = float(g["prompt_checks"].mean())
                row[f"{name}_prompt_goods_mean"] = float(g["prompt_goods"].mean())
            for r in pr:
                rk = (r["category"], r["seed"])
                for kk in (
                    "prompted",
                    "first_prompt",
                    "prompt_votes",
                    "prompt_checks",
                    "prompt_goods",
                    "goods_before_prompt",
                ):
                    r[kk] = cp[rk][kk]
                if cp[rk]["prompted"]:
                    goods_at_prompt.append(
                        {
                            **key,
                            "goods_before_prompt": cp[rk]["goods_before_prompt"],
                            "first_prompt": cp[rk]["first_prompt"],
                        }
                    )
            # Drift: nothing differs before the control's first prompt, so the picks must agree to there.
            agree, never_same = [], []
            for r in pr:
                rk = (r["category"], r["seed"])
                t_div = first_divergence(cp[rk]["ids"], apk[rk]["ids"])
                r["first_divergence"] = t_div
                limit = cp[rk]["first_prompt"]
                if limit is None:
                    never_same.append(t_div is None)
                else:
                    agree.append(t_div is None or t_div >= limit)
            row["pairs_agree_to_prompt"] = float(np.mean(agree)) if agree else None
            row["never_prompted"] = len(never_same)
            row["never_prompted_identical"] = float(np.mean(never_same)) if never_same else None
            # Per vote: each set's mean objective, the paired difference, and where the control's checks are.
            # A prompted vote is one the arm spends acquiring instead.
            casting = np.zeros((len(runs), HORIZON), dtype=bool)
            for i, r in enumerate(runs):
                casting[i, [t - 1 for t in cp[r]["prompt_vote_list"]]] = True
            entered = np.maximum.accumulate(casting, axis=1)
            for t in range(HORIZON):
                d = A[:, t] - C[:, t]
                curves.append(
                    {
                        **key,
                        "vote": t + 1,
                        "control": float(C[:, t].mean()),
                        "arm": float(A[:, t].mean()),
                        "diff": float(d.mean()),
                        "diff_se": _se(d),
                        "share_casting_prompt": float(casting[:, t].mean()),
                        "share_prompted_by": float(entered[:, t].mean()),
                    }
                )
            rows.append(row)
            per_run += pr
            print(
                f"K{k} beta {beta}: {len(runs)} paired runs, d(1-50) {row['d_1_50']:+.3f} +- {row['d_1_50_se']:.3f}, "
                f"d(1-150) {row['d_1_150']:+.3f} +- {row['d_1_150_se']:.3f}",
                file=sys.stderr,
            )
    pd.DataFrame(rows).round(4).to_csv(args.out / "ab.csv", index=False)
    pd.DataFrame(curves).round(4).to_csv(args.out / "curves.csv", index=False)
    pd.DataFrame(goods_at_prompt).to_csv(args.out / "goods_at_prompt.csv", index=False)
    # Per run: names FHIBE subject ids, so it stays in the owner-only analysis directory.
    pd.DataFrame(per_run).round(4).to_csv(args.out / "per_run.csv", index=False)
    print(f"wrote ab.csv, curves.csv, goods_at_prompt.csv, per_run.csv -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
