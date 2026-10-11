#!/usr/bin/env python
"""#4776: price hand-over rules on sessions run with the hand-over shadow (``CALIB_HANDOVER_SHADOW=1``).

Below the label quota Test gives the Goods' centroid; past it, the trained head. A rule may keep
the centroid past the quota. Neither picks, so a rule changes only what Test shows, and every rule
is priced on the same sessions: at each click past the quota the cells hold the head's row (the
untagged row, what Test gives today), the centroid's (tagged ``handover:centroid``) and the
evidence a rule may read (``__handover.csv``: each Good held out in turn, its rank in the unvoted
pool under the centroid of the other Goods and under the head fitted without it).

What a Find returns at every vote 1..150 is read as the face review reads it (#4762,
``sota_face_tables.read_find``): Test's rows, the end-of-run check's left out, carried between
rows, 0 before the first. Every run is in every average (#4631). Per session set and rule:

* the session mean of Find's F-beta over votes 1-50 and 1-150, paired against today's (the head
  from the quota), mean and SE over runs; AP's the same way;
* the per-vote curve of both, and the share of trained clicks the rule shows the centroid on.

Rules (each reads only what Test could read at that click):

* ``today``     the head from the quota (the app);
* ``never``     the centroid at every click (no hand-over);
* ``oracle``    whichever of the two scores the higher F-beta on the withheld half, per click
                (not a rule: the ceiling of every rule here);
* ``goods>=G``  the head once the labels hold G Goods; ``bads>=B`` once they hold B Bads;
* ``judge:<s>`` the head when its held-out Goods rank better than the centroid's by statistic
                ``s`` (``mrr``, ``logrank``, ``median``, ``wins``: more Goods ranked higher by the
                head than by the centroid); with fewer than two Goods there is nothing to hold
                out, and the centroid stays (``+g1`` hands those clicks to the head instead).
                ``sticky`` keeps the head once it has won.

The centroid-prior heads (``prior_ap_s<S>``) are ranking-only, so they are read as AP curves.

Usage::

    python handover_4776.py --set k1-b1-1024=<cells dir>::fhibe_faces_1024 ... --out <dir> [--beta-of k1-b1-1024=1]

A set is ``NAME=CELLS_DIR[::DATASET]``; DATASET keeps one dataset's runs (two share a FHIBE cell dir).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import _cells_io  # noqa: E402
from _cells_paths import main_frame_files  # noqa: E402

HORIZON = 150
TAG = "handover:centroid"
COLS = [
    "t",
    "phase",
    "dataset",
    "category",
    "seed",
    "gmm_variant",
    "schedule",
    "pool_variant",
    "detector_tier",
    "fbeta",
    "average_precision",
    "n_flagged",
    "n_good",
    "n_bad",
]
PRIOR = ["prior_ap_s1", "prior_ap_s3", "prior_ap_s10"]
STATS = ("mrr", "logrank", "median", "wins")


def _ranks(s: object) -> np.ndarray:
    if not isinstance(s, str) or not s.strip():
        return np.empty(0)
    return np.array([float(x) for x in s.split()])


def judge_prefers_head(c: np.ndarray, h: np.ndarray, stat: str) -> bool | None:
    """Whether the held-out ranks say the head ranks the person better; ``None`` with nothing held out."""
    if not len(c) or len(c) != len(h):
        return None
    if stat == "mrr":
        return float(np.mean(1.0 / h)) > float(np.mean(1.0 / c))
    if stat == "logrank":
        return float(np.mean(np.log2(h))) < float(np.mean(np.log2(c)))
    if stat == "median":
        return float(np.median(h)) < float(np.median(c))
    if stat == "wins":
        return int(np.sum(h < c)) > int(np.sum(h > c))
    raise ValueError(stat)


def rule_names() -> list[str]:
    names = ["today", "never", "oracle"]
    names += [f"goods>={g}" for g in (4, 5, 6, 8)]
    names += [f"bads>={b}" for b in (8, 16, 32, 64)]
    for s in STATS:
        names += [f"judge:{s}", f"judge:{s}+g1", f"judge:{s} sticky"]
    return names


def _carry(ts: np.ndarray, vals: np.ndarray, fill: float) -> np.ndarray:
    """Per vote 1..HORIZON: the value of the last row at or before it, *fill* before the first."""
    out = np.full(HORIZON, np.nan)
    for t, v in zip(ts, vals, strict=True):
        if 1 <= t <= HORIZON:
            out[int(t) - 1] = v
    s = pd.Series(out).ffill()
    return s.fillna(fill).to_numpy() if fill == fill else s.to_numpy()


def read_run(path: str, dataset: str | None) -> list[dict] | None:
    """Every rule's Find F-beta and AP at every vote, per run in this cell (one per dataset/style)."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if df.empty:
        return None
    df = df[(df["phase"].fillna("").astype(str) != "ceiling") & (df["t"] > 0)]
    df = df[~_cells_io.check_rows(df)]
    hpath = Path(path.replace(".csv", "__handover.csv"))
    ev = pd.read_csv(hpath) if hpath.exists() and hpath.stat().st_size else pd.DataFrame()
    out = []
    for (ds, cat, seed), run in df.groupby(["dataset", "category", "seed"], sort=False):
        if dataset and ds != dataset:
            continue
        base = _cells_io._base_rows(run).groupby("t").last()
        shadow = run[run["gmm_variant"].fillna("") == TAG].groupby("t").last()
        if base.empty:
            continue
        tiers = base["detector_tier"].fillna("").astype(str)
        trained_t = base.index[tiers == "trained"].to_numpy()
        # The centroid at every row: its own rows under the quota, the shadow's past it.
        cen = base[["fbeta", "average_precision", "n_flagged"]].copy()
        if len(trained_t):
            missing = [t for t in trained_t if t not in shadow.index]
            if missing:
                raise RuntimeError(
                    f"{path}: {len(missing)} trained clicks have no shadow row (run without the shadow?)"
                )
            cen.loc[trained_t, ["fbeta", "average_precision", "n_flagged"]] = shadow.loc[
                trained_t, ["fbeta", "average_precision", "n_flagged"]
            ].to_numpy()
        e = ev[(ev["dataset"] == ds) & (ev["category"] == cat) & (ev["seed"] == seed)] if len(ev) else ev
        e = e.groupby("t").last() if len(e) else e
        ts = base.index.to_numpy()
        hf, ha = base["fbeta"].to_numpy(float), base["average_precision"].to_numpy(float)
        cf, ca = cen["fbeta"].to_numpy(float), cen["average_precision"].to_numpy(float)
        is_tr = (tiers == "trained").to_numpy()
        n_good = base["n_good"].to_numpy(float)
        n_bad = base["n_bad"].to_numpy(float)
        # The evidence at each trained row.
        prefer: dict[str, list[bool | None]] = {s: [] for s in STATS}
        for t in ts:
            if t in getattr(e, "index", []) and len(e):
                c, h = _ranks(e.at[t, "centroid_ranks"]), _ranks(e.at[t, "head_ranks"])
            else:
                c = h = np.empty(0)
            for s in STATS:
                prefer[s].append(judge_prefers_head(c, h, s))

        def pick(show_head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            f = np.where(show_head, hf, cf)
            a = np.where(show_head, ha, ca)
            return _carry(ts, f, 0.0), _carry(ts, a, float("nan"))

        rules: dict[str, np.ndarray] = {}
        rules["today"] = is_tr.copy()
        rules["never"] = np.zeros_like(is_tr)
        rules["oracle"] = is_tr & (hf >= cf)
        for g in (4, 5, 6, 8):
            rules[f"goods>={g}"] = is_tr & (n_good >= g)
        for b in (8, 16, 32, 64):
            rules[f"bads>={b}"] = is_tr & (n_bad >= b)
        for s in STATS:
            p = prefer[s]
            plain = np.array([bool(x) for x in p]) & is_tr
            g1 = np.array([True if x is None else bool(x) for x in p]) & is_tr
            sticky = np.maximum.accumulate(plain.astype(int)).astype(bool) & is_tr
            rules[f"judge:{s}"] = plain
            rules[f"judge:{s}+g1"] = g1
            rules[f"judge:{s} sticky"] = sticky
        rec = {"dataset": ds, "category": cat, "seed": seed, "F": {}, "AP": {}, "share_centroid": {}}
        n_tr = int(is_tr.sum())
        for name, show in rules.items():
            f, a = pick(show)
            rec["F"][name], rec["AP"][name] = f, a
            rec["share_centroid"][name] = float((is_tr & ~show).sum() / n_tr) if n_tr else float("nan")
        # The ranking-only heads: AP at every trained row, the centroid's elsewhere.
        if len(e):
            for col in PRIOR:
                v = np.array([e.at[t, col] if (is_tr[i] and t in e.index) else ca[i] for i, t in enumerate(ts)], float)
                rec["AP"][col] = _carry(ts, v, float("nan"))
        rec["first_trained"] = int(trained_t.min()) if len(trained_t) else -1
        rec["n_good_at_handover"] = float(n_good[is_tr][0]) if n_tr else float("nan")
        out.append(rec)
    return out


def _se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(name: str, runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per rule: session means over 1-50 and 1-150, paired against today; per vote: the curves."""
    rows, curves = [], []
    today_f = np.stack([r["F"]["today"] for r in runs])
    today_a = np.stack([r["AP"]["today"] for r in runs])
    for rule in [*rule_names(), *PRIOR]:
        have_f = rule in runs[0]["F"]
        a = np.stack([r["AP"].get(rule, np.full(HORIZON, np.nan)) for r in runs])
        row: dict = {"set": name, "rule": rule, "runs": len(runs)}
        if have_f:
            f = np.stack([r["F"][rule] for r in runs])
            for lo, hi in ((1, 50), (1, 150), (51, 150)):
                m, m0 = f[:, lo - 1 : hi].mean(axis=1), today_f[:, lo - 1 : hi].mean(axis=1)
                row[f"F_{lo}_{hi}"] = float(m.mean())
                row[f"dF_{lo}_{hi}"] = float((m - m0).mean())
                row[f"dF_{lo}_{hi}_se"] = _se(m - m0)
                row[f"win_{lo}_{hi}"] = float(((m - m0) > 1e-9).mean())
                row[f"loss_{lo}_{hi}"] = float(((m - m0) < -1e-9).mean())
            row["share_centroid"] = float(np.nanmean([r["share_centroid"][rule] for r in runs]))
        for lo, hi in ((1, 50), (1, 150)):
            m, m0 = np.nanmean(a[:, lo - 1 : hi], axis=1), np.nanmean(today_a[:, lo - 1 : hi], axis=1)
            row[f"AP_{lo}_{hi}"] = float(np.nanmean(m))
            row[f"dAP_{lo}_{hi}"] = float(np.nanmean(m - m0))
            row[f"dAP_{lo}_{hi}_se"] = _se(m - m0)
        rows.append(row)
        for t in range(HORIZON):
            curves.append(
                {
                    "set": name,
                    "rule": rule,
                    "vote": t + 1,
                    "F": float(np.stack([r["F"][rule] for r in runs])[:, t].mean()) if have_f else float("nan"),
                    "AP": float(np.nanmean(a[:, t])),
                }
            )
    return rows, curves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="NAME=CELLS_DIR[::DATASET]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_curves, per_run = [], [], []
    for spec in args.set:
        name, _, rest = spec.partition("=")
        cells, _, dataset = rest.partition("::")
        files = [str(f) for f in main_frame_files(cells)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            got = [
                r for rs in pool.map(partial(read_run, dataset=dataset or None), files, chunksize=8) if rs for r in rs
            ]
        if not got:
            print(f"{name}: no runs in {cells}", file=sys.stderr)
            continue
        rows, curves = summarize(name, got)
        all_rows += rows
        all_curves += curves
        for r in got:
            per_run.append(
                {
                    "set": name,
                    "category": r["category"],
                    "seed": r["seed"],
                    "first_trained": r["first_trained"],
                    "n_good_at_handover": r["n_good_at_handover"],
                    **{f"F150:{k}": float(v.mean()) for k, v in r["F"].items()},
                    **{f"F50:{k}": float(v[:50].mean()) for k, v in r["F"].items()},
                }
            )
        print(f"{name}: {len(got)} runs from {len(files)} cells", file=sys.stderr)
    pd.DataFrame(all_rows).round(5).to_csv(args.out / "rules.csv", index=False)
    pd.DataFrame(all_curves).round(5).to_csv(args.out / "curves.csv", index=False)
    pd.DataFrame(per_run).round(5).to_csv(args.out / "per_run.csv", index=False)
    print(f"wrote rules.csv, curves.csv, per_run.csv -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
