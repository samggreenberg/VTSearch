#!/usr/bin/env python
"""#4798: price fixes to the labels line's positives total on sessions run with ``CALIB_LINE_COUNTS=1``.

At beta 4 the class model's cut sometimes keeps a whole corpus: ``corpus_cut``'s total is the
larger of the 2-part estimate and the sum of every 3-part chance, and diffuse chances run a
recall-leaning cut to the corpus's end. Each candidate fix (``vtscore.eval.arms_cut_total``) is
recorded at every click Test gives the trained head as ``k_cut_<name>`` in ``__linecounts.csv``,
the class model's cut whether or not #4786's head count serves there. The app draws the head's
count while the class model is thin, so a candidate changes only the clicks past that gate:

* ``today``        what Test keeps (the head's count while thin, the class model's cut after);
* ``cut:<name>``   the head's count while thin, candidate ``<name>``'s cut after;
* ``oracle``       the best count on the withheld half, per click (a bound, not a rule);
* ``headgoods``    not a fix to the total: #4786's head count past its Bads gate too, while the
                   class model holds fewer than ``HEAD_COUNT_GOODS`` Goods (any Bads), today's
                   cut after. Priced because the faces' deep cuts sit there;
* ``floorhead``    ``headgoods``'s first half: there, never fewer than the head's count (today's cut
                   when it keeps more);
* ``caphead``      its second half: there, never more than the head's count (the deep-cut fix);
* ``caphead-dry``  ``caphead`` only once the session has gone ``DRY_RUN_TARGET`` votes without a
                   Good (#4488's dry run): the walk's own sign that few positives are left to find.

``cut:class`` is today by construction; the analyzer counts the clicks where it is not (none, if
the frame is right). Scoring as #4786's ``line_4786.py``: Find's F-beta at every vote, carried,
every run in every average; per set and rule the session mean over votes 1-50, 1-150 and 51-150,
paired against today.

Usage::

    python cut_4798.py --set k4-b4-1024=<cells dir>::fhibe_faces_roll_1024 ... --set coco-b4=<cells> --out <dir>

``per_run.csv`` and ``clicks.csv`` name FHIBE subjects: owner-only.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import _cells_io  # noqa: E402
from _cells_paths import main_frame_files  # noqa: E402
from line_4786 import COLS, HORIZON, _carry, _ranks, fbeta_at, oracle_fbeta  # noqa: E402

from vtscore.detectors.labeling_progress import DRY_RUN_TARGET  # noqa: E402
from vtscore.eval.arms_cut_total import CUT_VARIANTS  # noqa: E402
from vtscore.training.thresholds.labels_line import HEAD_COUNT_BADS, HEAD_COUNT_GOODS  # noqa: E402

RULES = ["today", "oracle", *[f"cut:{v}" for v in CUT_VARIANTS], "headgoods", "floorhead", "caphead", "caphead-dry"]


def _linecounts_path(path: str) -> Path:
    p = Path(path)
    name = p.name.replace(".csv", "__linecounts.csv", 1)
    return p.with_name(name)


def _dry(ts: np.ndarray, n_good: np.ndarray) -> np.ndarray:
    """At each row, the votes since the session's Good count last rose (since the start if it never has)."""
    out, last = np.zeros(len(ts)), 0
    for i in range(len(ts)):
        if i and n_good[i] > n_good[i - 1]:
            last = ts[i]
        out[i] = ts[i] - last
    return out


def read_run(path: str, dataset: str | None) -> list[dict] | None:
    """Every rule's Find F-beta at every vote, per run in this cell, and the clicks where a candidate moves."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if df.empty:
        return None
    df = df[(df["phase"].fillna("").astype(str) != "ceiling") & (df["t"] > 0)]
    df = df[~_cells_io.check_rows(df)]
    lpath = _linecounts_path(path)
    lc = pd.read_csv(lpath) if lpath.exists() and lpath.stat().st_size else pd.DataFrame()
    out = []
    for (ds, cat, seed), run in df.groupby(["dataset", "category", "seed"], sort=False):
        if dataset and ds != dataset:
            continue
        base = _cells_io._base_rows(run).groupby("t").last()
        if base.empty:
            continue
        e = lc[(lc["dataset"] == ds) & (lc["category"] == cat) & (lc["seed"] == seed)] if len(lc) else lc
        e = e.groupby("t").last() if len(e) else e
        ts = base.index.to_numpy()
        is_tr = (base["detector_tier"].fillna("").astype(str) == "trained").to_numpy()
        trained_t = ts[is_tr]
        missing = [t for t in trained_t if not len(e) or t not in e.index]
        if missing:
            raise RuntimeError(f"{path}: {len(missing)} trained clicks have no line counts (run without them?)")
        f_today = base["fbeta"].to_numpy(float)
        dry = _dry(ts, base["n_good"].to_numpy(float))
        beta = float(e["beta"].iloc[0]) if len(e) else float("nan")
        f = {r: f_today.copy() for r in RULES}
        mismatch = 0
        clicks: list[dict] = []
        for i, t in enumerate(ts):
            if not is_tr[i]:
                continue
            rec = e.loc[t]
            ranks, n_pos = _ranks(rec["test_pos_ranks"]), int(rec["n_test_pos"])
            f["oracle"][i] = oracle_fbeta(ranks, n_pos, beta)
            n_neg, n_pos_m = float(rec["model_n_neg"]), float(rec["model_n_pos"])
            if not np.isfinite(n_neg):
                continue  # no class model: the retrain's fallback, the same under every candidate
            if n_neg < HEAD_COUNT_BADS and n_pos_m < HEAD_COUNT_GOODS:
                continue  # the head's count serves: no candidate moves it
            if n_pos_m < HEAD_COUNT_GOODS:
                k_head, k_class = int(rec["k_head_count"]), int(rec["k_cut_class"])
                f["headgoods"][i] = fbeta_at(k_head, ranks, n_pos, beta)
                f["floorhead"][i] = fbeta_at(max(k_class, k_head), ranks, n_pos, beta)
                f["caphead"][i] = fbeta_at(min(k_class, k_head), ranks, n_pos, beta)
                if dry[i] >= DRY_RUN_TARGET:
                    f["caphead-dry"][i] = f["caphead"][i]
            if int(rec["k_cut_class"]) != int(rec["k_test"]):
                mismatch += 1
            ks = {v: int(rec[f"k_cut_{v}"]) for v in CUT_VARIANTS}
            fs = {v: fbeta_at(k, ranks, n_pos, beta) for v, k in ks.items()}
            for v in CUT_VARIANTS:
                f[f"cut:{v}"][i] = fs[v]
            if len({ks[v] for v in CUT_VARIANTS} | {int(rec["k_head_count"])}) > 1:
                clicks.append(
                    {
                        "t": int(t),
                        "n_good": int(rec["n_good"]),
                        "n_bad": int(rec["n_bad"]),
                        "n_test_pos": n_pos,
                        "separation": float(rec["separation"]),
                        "est_positives": float(rec["est_positives"]),
                        "chances_sum": float(rec["chances_sum"]),
                        "model_n_pos": n_pos_m,
                        "model_n_neg": n_neg,
                        "dry": float(dry[i]),
                        "k_head_count": int(rec["k_head_count"]),
                        "f_head_count": fbeta_at(int(rec["k_head_count"]), ranks, n_pos, beta),
                        **{f"k_{v}": ks[v] for v in CUT_VARIANTS},
                        **{f"f_{v}": fs[v] for v in CUT_VARIANTS},
                        "f_oracle": f["oracle"][i],
                    }
                )
        out.append(
            {
                "dataset": ds,
                "category": cat,
                "seed": seed,
                "F": {r: _carry(ts, v, 0.0) for r, v in f.items()},
                "mismatch": mismatch,
                "n_past_gate": int(
                    sum(
                        1
                        for t in trained_t
                        if np.isfinite(float(e.at[t, "model_n_neg"]))
                        and not (
                            float(e.at[t, "model_n_neg"]) < HEAD_COUNT_BADS
                            and float(e.at[t, "model_n_pos"]) < HEAD_COUNT_GOODS
                        )
                    )
                ),
                "clicks": clicks,
            }
        )
    return out


def _se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(name: str, runs: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, curves = [], []
    today = np.stack([r["F"]["today"] for r in runs])
    for rule in RULES:
        f = np.stack([r["F"][rule] for r in runs])
        row: dict = {"set": name, "rule": rule, "runs": len(runs)}
        for lo, hi in ((1, 50), (1, 150), (51, 150)):
            m, m0 = f[:, lo - 1 : hi].mean(axis=1), today[:, lo - 1 : hi].mean(axis=1)
            d = m - m0
            row[f"F_{lo}_{hi}"] = float(np.nanmean(m))
            row[f"dF_{lo}_{hi}"] = float(np.nanmean(d))
            row[f"dF_{lo}_{hi}_se"] = _se(d)
            row[f"win_{lo}_{hi}"] = float((d > 1e-6).mean())
            row[f"loss_{lo}_{hi}"] = float((d < -1e-6).mean())
        rows.append(row)
        mean_curve = np.nanmean(f, axis=0)
        curves += [{"set": name, "rule": rule, "vote": t + 1, "F": float(mean_curve[t])} for t in range(HORIZON)]
    return rows, curves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="NAME=CELLS_DIR[::DATASET]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_curves, per_run, clicks = [], [], [], []
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
            clicks += [{"set": name, "category": r["category"], "seed": r["seed"], **c} for c in r["clicks"]]
            per_run.append(
                {
                    "set": name,
                    "category": r["category"],
                    "seed": r["seed"],
                    **{f"F150:{k}": float(v.mean()) for k, v in r["F"].items()},
                    **{f"F50:{k}": float(v[:50].mean()) for k, v in r["F"].items()},
                }
            )
        print(
            f"{name}: {len(got)} runs from {len(files)} cells; {sum(r['n_past_gate'] for r in got)} clicks past the "
            f"head's count, {sum(r['mismatch'] for r in got)} where cut:class is not today",
            file=sys.stderr,
        )
    pd.DataFrame(all_rows).round(5).to_csv(args.out / "rules.csv", index=False)
    pd.DataFrame(all_curves).round(5).to_csv(args.out / "curves.csv", index=False)
    pd.DataFrame(per_run).round(5).to_csv(args.out / "per_run.csv", index=False)
    pd.DataFrame(clicks).round(5).to_csv(args.out / "clicks.csv", index=False)
    print(f"wrote rules.csv, curves.csv, per_run.csv, clicks.csv -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
