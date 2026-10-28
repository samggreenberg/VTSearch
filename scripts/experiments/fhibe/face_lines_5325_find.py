#!/usr/bin/env python
"""Find's line on imagekit's face scores (#5325): what the head's count line keeps, and what other bars would.

A face session's class model never holds :data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_GOODS`
held-out Goods (a person has about four photos in the voting half), so Find's line is the head's count line all
session (:func:`~vtscore.training.thresholds.labels_line.head_count_cut`, #4786 / #5039 / #5084): a multiple of
#4603's count of what stands out (the logits above ``median + z spreads`` less what a Gaussian bulk puts there,
``z`` = :data:`~vtscore.training.thresholds.gmm.TEXT_SORT_COUNT_Z`), never fewer than the top's widest drop.

This script reads a face review's score snapshots (``CALIB_SAVE_TEST_SCORES=1`` + ``CALIB_SAVE_TEST_SCORES_AT``:
the withheld half's scores at each listed click, at the last vote and after the check, with the calibration
folds' held-out votes) and draws, through the library on each, the line of every candidate in
:data:`CANDIDATES` at the run's balance: the app's, the count read at a higher bar, the drop alone, a smaller
multiplier at a balance of 1 and below. The app's line must keep what the session's line-count frame
(``__linecounts.csv``, ``k_head_count``) says it kept at the same click.

It also writes, per snapshot, where the bulk's tail stands: the negatives above ``z`` spreads against a Gaussian
bulk's share there, and the positives' z.

  face_lines_5325_find.py price --runs DIR [--runs DIR ...] --out FILE.csv.gz [--procs N] [--limit N]
  face_lines_5325_find.py report --priced FILE.csv.gz --runs-root DIR --strata person_strata.csv --out FILE.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import common  # noqa: E402

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: name -> (count's bar z, multiplier scale at a balance of 1 and below); ``None`` keeps the app's.
CANDIDATES = {
    "app": (None, None),
    "z4.5": (4.5, None),
    "z5": (5.0, None),
    "z5.5": (5.5, None),
    "z6": (6.0, None),
    "z7": (7.0, None),
    "drop": (99.0, None),  # no logit stands 99 spreads up: the count is 1, so the drop keeps
    "s0.5": (None, 0.5),
    "s0.35": (None, 0.35),
    "z5s0.5": (5.0, 0.5),
}
ZS = (3.0, 4.0, 5.0, 6.0)


@contextmanager
def candidate(z: float | None, scale: float | None):
    from vtscore.training.thresholds import gmm, labels_line

    z0, s0 = gmm.TEXT_SORT_COUNT_Z, labels_line.HEAD_COUNT_PRECISION_SCALE
    try:
        if z is not None:
            gmm.TEXT_SORT_COUNT_Z = float(z)
        if scale is not None:
            labels_line.HEAD_COUNT_PRECISION_SCALE = float(scale)
        yield
    finally:
        gmm.TEXT_SORT_COUNT_Z, labels_line.HEAD_COUNT_PRECISION_SCALE = z0, s0


def _snapshots(npz: str) -> dict[str, dict]:
    z = np.load(npz, allow_pickle=False)
    snaps: dict[str, dict] = {}
    for name in z.files:
        key, field = name.rsplit("/", 1)
        snaps.setdefault(key, {})[field] = z[name]
    for snap in snaps.values():
        snap.update(json.loads(str(snap.pop("meta"))))
    return snaps


def _cell(args) -> list[dict]:
    from vtscore.training.thresholds.labels_line import (
        HEAD_COUNT_GOODS,
        _finite_unit,
        _logit,
        fit_labels_line,
        head_count_cut,
    )

    npz, ident = args
    lc_path = npz.replace("__testscores.npz", "__linecounts.csv.gz")
    lc = pd.read_csv(lc_path) if os.path.exists(lc_path) else None
    lc_k = {} if lc is None else dict(zip(lc["t"].astype(int), lc["k_head_count"].astype(int)))
    rows = []
    for key, snap in sorted(_snapshots(npz).items()):
        label = key.rsplit("/", 1)[1]
        s = _finite_unit(np.asarray(snap["scores"], dtype=np.float64))
        y = np.asarray(snap["labels"], dtype=np.int8)
        npos = int(y.sum())
        if npos == 0:
            continue
        order = np.argsort(-s, kind="stable")
        s, y = s[order], y[order]
        fl = np.asarray(snap["fold_labels"], dtype=np.int8)
        fs = np.asarray(snap["fold_scores"], dtype=np.float64)
        line = fit_labels_line([(fs, fl.astype(np.float64))], s) if fl.size else None
        if line is not None and int(line.model.n_pos) >= HEAD_COUNT_GOODS:
            continue  # the class model's cut serves, not the head's count
        held = None if line is None else line.model.pos_logits
        beta = float(ident["beta"])
        t = int(snap.get("t", -1))
        b2 = beta * beta
        tp = np.cumsum(y)
        f_at = (1 + b2) * tp / (b2 * npos + np.arange(1, y.size + 1))
        x = _logit(s)
        med = float(np.median(x))
        sd = 1.4826 * float(np.median(np.abs(x - med)))
        r = {**ident, "snap": label, "t": t, "n_good": int(snap.get("n_good", -1)), "n_test": int(s.size),
             "n_pos": npos, "best_f": float(f_at.max()), "best_k": int(np.argmax(f_at)) + 1,
             "ap": float(np.mean(tp[y == 1] / (np.flatnonzero(y == 1) + 1))), "lc_k": lc_k.get(t, -1),
             "held_goods": int(0 if held is None else held.size)}  # fmt: skip
        if sd > 0:
            zx = (x - med) / sd
            r["pos_z_min"], r["pos_z_med"] = float(zx[y == 1].min()), float(np.median(zx[y == 1]))
            for zz in ZS:
                r[f"neg_above_z{zz:g}"] = int(np.count_nonzero(zx[y == 0] > zz))
                r[f"pos_above_z{zz:g}"] = int(np.count_nonzero(zx[y == 1] > zz))
                r[f"gauss_above_z{zz:g}"] = s.size * 0.5 * math.erfc(zz / math.sqrt(2.0))
        empty = s[y == 0]
        for name, (z, scale) in CANDIDATES.items():
            with candidate(z, scale):
                k = int(np.count_nonzero(s >= head_count_cut(s, beta, held)))
                k_empty = int(np.count_nonzero(empty >= head_count_cut(empty, beta, held)))
            r[f"k_{name}"] = k
            r[f"F_{name}"] = float(f_at[k - 1]) if k > 0 else 0.0
            r[f"P_{name}"] = float(tp[k - 1] / k) if k > 0 else 0.0
            r[f"R_{name}"] = float(tp[k - 1] / npos) if k > 0 else 0.0
            r[f"kempty_{name}"] = k_empty
        rows.append(r)
    return rows


def _ident(npz: str, run: str) -> dict:
    idx = int(re.search(r"task_(\d+)__", os.path.basename(npz)).group(1))
    m = re.search(r"-k(\d+)-b(\d+)$", run)
    beta = {"025": 0.25, "1": 1.0, "4": 4.0}[m.group(2)]
    return {"run": run, "idx": idx, "k": int(m.group(1)), "beta": beta, "size": 1024 if idx < 562 else 640,
            "person": idx % 562}  # fmt: skip


def price(a: argparse.Namespace) -> int:
    jobs = []
    for run_dir in a.runs:
        run = os.path.basename(run_dir.rstrip("/"))
        files = sorted(glob.glob(os.path.join(run_dir, "results", "cells", "task_*__testscores.npz")))
        jobs.extend((f, _ident(f, run)) for f in files)
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)][: a.limit]
    common.log(f"{len(jobs)} cells")
    rows = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for i, out in enumerate(ex.map(_cell, jobs, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 500 == 0:
                common.log(f"{i + 1}/{len(jobs)}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    chk = df[df["lc_k"] >= 0]
    bad = int((chk["k_app"] != chk["lc_k"]).sum())
    common.log(f"{len(df)} snapshots -> {a.out}; app vs line-count frame: {bad} mismatches of {len(chk)}")
    return 1 if bad else 0


def _category(run_dir: str, idx: int) -> str:
    import gzip

    with gzip.open(os.path.join(run_dir, "results", "cells", f"task_{idx:04d}.csv.gz"), "rt") as fh:
        head = fh.readline().rstrip("\n").split(",")
        return fh.readline().split(",")[head.index("category")]


def report(a: argparse.Namespace) -> int:
    """Per snapshot, each candidate's F-beta paired against the app's, SE on the person; the pronoun gap; failures.

    A failed session (#5336) ends with AP below 0.1 while the person's full-label ceiling reaches 0.5: its starting
    photos' vectors miss the person, which no line can mend, so it is flagged and left out of the over-return count.
    """
    d = pd.read_csv(a.priced)
    cats = {(r, i): _category(os.path.join(a.runs_root, r), i) for r, i in d[["run", "idx"]].drop_duplicates().values}
    d["category"] = [cats[(r, i)] for r, i in zip(d["run"], d["idx"])]
    strata = pd.read_csv(a.strata)
    strata["pronoun"] = strata["pronoun"].str.lower().str.extract(r"^(she/her|he/him)", expand=False)
    d = d.merge(strata[["category", "pronoun"]], on="category", how="left")
    cell = ["run", "idx"]
    ap = d.pivot_table(index=cell, columns="snap", values="ap")
    failed = ((ap["final"] < 0.1) & (ap["ceiling"] >= 0.5)).rename("failed").reset_index()
    d = d.merge(failed, on=cell, how="left")
    key = ["size", "k", "beta", "snap"]
    rows = []
    for g, h in d[d["snap"] != "ceiling"].groupby(key):
        r = dict(zip(key, g), sessions=len(h), failed=int(h["failed"].sum()), pos=h["n_pos"].mean(),
                 k_app=h["k_app"].mean(), F_app=h["F_app"].mean())  # fmt: skip
        for c in CANDIDATES:
            if c == "app":
                continue
            diff = (h[f"F_{c}"] - h["F_app"]).groupby(h["category"]).mean()
            r[f"d_{c}"], r[f"se_{c}"] = diff.mean(), diff.std(ddof=1) / math.sqrt(len(diff))
            r[f"k_{c}"] = h[f"k_{c}"].mean()
        ok = h[~h["failed"]]
        r["over_return"] = int(((ok["ap"] >= 0.95) & (ok["F_app"] < 0.7) & (ok["k_app"] > ok["n_pos"])).sum())
        for c in ("app", *a.gap):
            by = ok.groupby("pronoun")[f"F_{c}"]
            m, s, n = by.mean(), by.std(ddof=1), by.size()
            r[f"gap_{c}"] = m["she/her"] - m["he/him"]
            r[f"gap_se_{c}"] = math.sqrt(s["she/her"] ** 2 / n["she/her"] + s["he/him"] ** 2 / n["he/him"])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(out[out["snap"].isin(["t8", "t20", "t50", "final"])].round(4).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price")
    p.add_argument("--runs", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--priced", required=True, help="price's output")
    r.add_argument("--runs-root", required=True, help="the directory holding the priced runs")
    r.add_argument("--strata", required=True, help="person_strata.py's csv")
    r.add_argument("--gap", nargs="*", default=["z5", "z5.5"], help="candidates whose pronoun gap to report")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    return price(a) if a.cmd == "price" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
