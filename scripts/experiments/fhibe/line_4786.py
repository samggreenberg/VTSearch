#!/usr/bin/env python
"""#4786: price count rules for the first trained clicks' line on sessions run with ``CALIB_LINE_COUNTS=1``.

At beta 4 the labels line on the first trained head keeps too many faces. A rule for those clicks
moves the line, not the ranking, so every count rule is priced on the same sessions: at each
click Test gives the trained head, ``__linecounts.csv`` holds where the withheld half's positives
sit in the head's ranking and the counts a rule may read (the labels line at each preset, the
Goods' centroid's count line, the count rule on the head's own scores, the Goods in hand). A rule
is a count ``k`` per click; keeping the head's top ``k`` scores exactly (``(1 + b^2) TP / (b^2 P +
k)``). Clicks the rule does not touch keep Test's own row.

What a Find returns at every vote 1..150 is read as the face review reads it (#4762): Test's rows,
the end-of-run check's left out, carried between rows, 0 before the first. Every run is in every
average (#4631). Per session set and rule: the session mean of Find's F-beta over votes 1-50 and
1-150, paired against today's line, mean and SE over runs, wins and losses.

Rules (each reads only what Test could read at that click; ``@N``: while the labels hold fewer
than N Bads, today's line after; ``@inf``: at every trained click; ``@negN``: while the labels
line's class model holds fewer than N Bads (its held-out ones, about half to two thirds of the
Bads in hand: what a rule inside the line can read, in Train and in Find alike); ``@sepD``: while
the labels line's separation d' is below D, at any vote; ``@negNposP``: while the class model holds fewer
than N Bads and fewer than P Goods, held out):

* ``today``           the labels line (the app);
* ``oracle``          the best count on the withheld half, per click (a bound, not a rule);
* ``carry@N``         the Goods' centroid's count line (the app's rule at the run's beta);
* ``carrymin@N``      the fewer of today's and the centroid's;
* ``cap<m>g@N``       today's, at most m times the Goods in hand;
* ``scaled@N``        today's, at most beta^0.708 times what the beta-1 labels line keeps
                      (the count line's own scaling, #4603);
* ``b1line@N``        the beta-1 labels line;
* ``headcount@N``     the centroid's count rule drawn on the head's own logit scores;
                      ``headcountmin@N`` the fewer of that and today's; ``headcountb1@N`` the more
                      of that and the beta-1 labels line's, so beta 4 never keeps less than beta 1;
* ``centroid@N``      the Goods' centroid's own Test (its ranking and line): #4776's "head from
                      N Bads", on today's head.

Usage::

    python line_4786.py --set k4-b4-1024=<cells dir>::fhibe_faces_roll_1024 ... --out <dir>

A set is ``NAME=CELLS_DIR[::DATASET]``; DATASET keeps one dataset's runs (two share a FHIBE cell dir).
``per_run.csv`` names FHIBE subjects: owner-only.
"""

from __future__ import annotations

import argparse
import math
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
GATES: tuple[float, ...] = (8, 12, 16, 24, 32, 48, 64, math.inf)
#: The class-model gates: a rule applies while the labels line's model holds fewer than N held-out Bads.
NEG_GATES: tuple[int, ...] = (4, 6, 8, 12, 16, 20, 24, 32)
#: The thin-model gates: fewer than N held-out Bads and fewer than P held-out Goods.
THIN_GATES: tuple[tuple[int, int], ...] = tuple((n, p) for n in (12, 20, 32, 1000) for p in (3, 4, 5, 6, 8))
#: The separation gates: a rule applies while the labels line's d' is below D (#4496's weak check reads 1.5).
SEP_GATES: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0)
CAPS: tuple[float, ...] = (1, 1.5, 2, 3, 4)
#: The count rules, each a function of one click's record (a row of ``__linecounts.csv``).
COUNT_RULES = (
    "carry",
    "carrymin",
    *(f"cap{m:g}g" for m in CAPS),
    "scaled",
    "b1line",
    "headcount",
    "headcountmin",
    "headcountb1",
)


def gate_tag(n: float) -> str:
    return "inf" if math.isinf(n) else f"{n:g}"


def gate_tags() -> list[str]:
    return (
        [gate_tag(n) for n in GATES]
        + [f"neg{n}" for n in NEG_GATES]
        + [f"neg{n}pos{p}" for n, p in THIN_GATES]
        + [f"sep{d:g}" for d in SEP_GATES]
    )


def gate_open(
    tag: str, n_bad: int, separation: float, model_n_neg: float = float("nan"), model_n_pos: float = float("nan")
) -> bool:
    """Whether a rule gated by *tag* applies at a click: its Bads, the class model's Bads and Goods, the line's d'."""
    if tag.startswith("sep"):
        return bool(np.isfinite(separation)) and separation < float(tag[3:])
    if tag.startswith("neg") and "pos" in tag:
        n, p = tag[3:].split("pos")
        return bool(np.isfinite(model_n_neg)) and model_n_neg < float(n) and model_n_pos < float(p)
    if tag.startswith("neg"):
        return bool(np.isfinite(model_n_neg)) and model_n_neg < float(tag[3:])
    return tag == "inf" or n_bad < float(tag)


def rule_names() -> list[str]:
    names = ["today", "oracle"]
    for g in gate_tags():
        names += [f"{r}@{g}" for r in (*COUNT_RULES, "centroid")]
    return names


def rule_count(rule: str, rec: pd.Series, beta: float) -> int:
    """The count *rule* keeps at the click *rec* records (the rule's own count, gate aside)."""
    today = int(rec["k_test"])
    if rule == "carry":
        return int(rec["k_centroid"])
    if rule == "carrymin":
        return min(today, int(rec["k_centroid"]))
    if rule.startswith("cap") and rule.endswith("g"):
        return min(today, int(math.floor(float(rule[3:-1]) * int(rec["n_good"]))))
    if rule == "scaled":
        b1 = int(rec["k_labels_b1"])
        return today if b1 < 0 else min(today, int(round(beta**0.708 * b1)))
    if rule == "b1line":
        b1 = int(rec["k_labels_b1"])
        return today if b1 < 0 else b1
    if rule == "headcount":
        hc = int(rec["k_head_count"])
        return today if hc < 0 else hc
    if rule == "headcountmin":
        hc = int(rec["k_head_count"])
        return today if hc < 0 else min(today, hc)
    if rule == "headcountb1":
        # Never fewer than the beta-1 line keeps: a recall-leaning balance keeps at least a balanced one's set.
        hc = int(rec["k_head_count"])
        return today if hc < 0 else max(hc, int(rec["k_labels_b1"]))
    raise ValueError(rule)


def fbeta_at(k: int, ranks: np.ndarray, n_pos: int, beta: float) -> float:
    """F-beta of keeping the top *k* of a ranking whose positives sit at sorted 0-based *ranks*."""
    if n_pos <= 0:
        return float("nan")
    tp = int(np.searchsorted(ranks, k, side="left"))
    if k <= 0 or tp == 0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * tp / (b2 * n_pos + k)


def oracle_fbeta(ranks: np.ndarray, n_pos: int, beta: float) -> float:
    """The best F-beta any count gets on this ranking: a cut just below one of its positives."""
    if n_pos <= 0:
        return float("nan")
    if not ranks.size:
        return 0.0
    b2 = beta * beta
    j = np.arange(1, ranks.size + 1, dtype=np.float64)
    return float(((1.0 + b2) * j / (b2 * n_pos + ranks + 1.0)).max())


def _ranks(s: object) -> np.ndarray:
    if not isinstance(s, str) or not s.strip():
        return np.empty(0, dtype=np.int64)
    return np.sort(np.array([int(x) for x in s.split()], dtype=np.int64))


def _carry(ts: np.ndarray, vals: np.ndarray, fill: float) -> np.ndarray:
    """Per vote 1..HORIZON: the value of the last row at or before it, *fill* before the first."""
    out = np.full(HORIZON, np.nan)
    for t, v in zip(ts, vals, strict=True):
        if 1 <= t <= HORIZON:
            out[int(t) - 1] = v
    s = pd.Series(out).ffill()
    return s.fillna(fill).to_numpy() if fill == fill else s.to_numpy()


def read_run(path: str, dataset: str | None) -> list[dict] | None:
    """Every rule's Find F-beta at every vote, per run in this cell, and the clicks' diagnostics."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if df.empty:
        return None
    df = df[(df["phase"].fillna("").astype(str) != "ceiling") & (df["t"] > 0)]
    df = df[~_cells_io.check_rows(df)]
    lpath = Path(path.replace(".csv", "__linecounts.csv"))
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
        tiers = base["detector_tier"].fillna("").astype(str).to_numpy()
        is_tr = tiers == "trained"
        trained_t = ts[is_tr]
        missing = [t for t in trained_t if not len(e) or t not in e.index]
        if missing:
            raise RuntimeError(f"{path}: {len(missing)} trained clicks have no line counts (run without them?)")
        f_today = base["fbeta"].to_numpy(float)
        beta = float(e["beta"].iloc[0]) if len(e) else float("nan")
        rules = rule_names()
        f = {r: f_today.copy() for r in rules}
        mismatch = 0
        diag: list[dict] = []
        for i, t in enumerate(ts):
            if not is_tr[i]:
                continue
            rec = e.loc[t]
            ranks, n_pos = _ranks(rec["test_pos_ranks"]), int(rec["n_test_pos"])
            got = fbeta_at(int(rec["k_test"]), ranks, n_pos, beta)
            if not (got == f_today[i] or (np.isnan(got) and np.isnan(f_today[i])) or abs(got - f_today[i]) < 2e-6):
                mismatch += 1
            f["oracle"][i] = oracle_fbeta(ranks, n_pos, beta)
            n_bad = int(rec["n_bad"])
            counts = {r: rule_count(r, rec, beta) for r in COUNT_RULES}
            fk = {r: fbeta_at(k, ranks, n_pos, beta) for r, k in counts.items()}
            sep = float(rec["separation"])
            for g in gate_tags():
                if not gate_open(g, n_bad, sep, float(rec["model_n_neg"]), float(rec["model_n_pos"])):
                    continue
                for r in COUNT_RULES:
                    f[f"{r}@{g}"][i] = fk[r]
                f[f"centroid@{g}"][i] = float(rec["centroid_fbeta"])
            diag.append(
                {
                    "t": int(t),
                    "n_good": int(rec["n_good"]),
                    "n_bad": n_bad,
                    "model_n_neg": float(rec["model_n_neg"]),
                    "model_n_pos": float(rec["model_n_pos"]),
                    "n_test_pos": n_pos,
                    "k_test": int(rec["k_test"]),
                    "k_centroid": int(rec["k_centroid"]),
                    "k_labels_b1": int(rec["k_labels_b1"]),
                    "k_head_count": int(rec["k_head_count"]),
                    "est_positives": float(rec["est_positives"]),
                    "separation": float(rec["separation"]),
                    "f_today": f_today[i],
                    "f_oracle": f["oracle"][i],
                    "f_carry": fk["carry"],
                    "f_headcount": fk["headcount"],
                    "f_centroid": float(rec["centroid_fbeta"]),
                }
            )
        rec_out = {
            "dataset": ds,
            "category": cat,
            "seed": seed,
            "beta": beta,
            "F": {r: _carry(ts, v, 0.0) for r, v in f.items()},
            "first_trained": int(trained_t.min()) if len(trained_t) else -1,
            "last_centroid_f": float(f_today[~is_tr & (ts < trained_t.min())][-1])
            if len(trained_t) and (~is_tr & (ts < trained_t.min())).any()
            else float("nan"),
            "mismatch": mismatch,
            "n_trained": int(is_tr.sum()),
            "clicks": diag,
        }
        out.append(rec_out)
    return out


def _se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(name: str, runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per rule: session means over 1-50, 1-150 and 51-150, paired against today; per vote: the curves."""
    rows, curves = [], []
    today = np.stack([r["F"]["today"] for r in runs])
    for rule in rule_names():
        f = np.stack([r["F"][rule] for r in runs])
        row: dict = {"set": name, "rule": rule, "runs": len(runs)}
        for lo, hi in ((1, 50), (1, 150), (51, 150)):
            m, m0 = f[:, lo - 1 : hi].mean(axis=1), today[:, lo - 1 : hi].mean(axis=1)
            d = m - m0
            row[f"F_{lo}_{hi}"] = float(np.nanmean(m))
            row[f"dF_{lo}_{hi}"] = float(np.nanmean(d))
            row[f"dF_{lo}_{hi}_se"] = _se(d)
            row[f"win_{lo}_{hi}"] = float((d > 1e-9).mean())
            row[f"loss_{lo}_{hi}"] = float((d < -1e-9).mean())
        rows.append(row)
        mean_curve = np.nanmean(f, axis=0)
        curves += [{"set": name, "rule": rule, "vote": t + 1, "F": float(mean_curve[t])} for t in range(HORIZON)]
    return rows, curves


def first_clicks(name: str, runs: list[dict]) -> list[dict]:
    """The first trained click of each run: what each line keeps there and what it scores."""
    out = []
    for r in runs:
        if not r["clicks"]:
            continue
        c = r["clicks"][0]
        out.append(
            {"set": name, "category": r["category"], "seed": r["seed"], "last_centroid_f": r["last_centroid_f"], **c}
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="NAME=CELLS_DIR[::DATASET]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_curves, per_run, firsts, clicks = [], [], [], [], []
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
        mism = sum(r["mismatch"] for r in got)
        rows, curves = summarize(name, got)
        all_rows += rows
        all_curves += curves
        firsts += first_clicks(name, got)
        for r in got:
            clicks += [{"set": name, "category": r["category"], "seed": r["seed"], **c} for c in r["clicks"]]
            per_run.append(
                {
                    "set": name,
                    "category": r["category"],
                    "seed": r["seed"],
                    "first_trained": r["first_trained"],
                    **{f"F150:{k}": float(v.mean()) for k, v in r["F"].items()},
                    **{f"F50:{k}": float(v[:50].mean()) for k, v in r["F"].items()},
                }
            )
        n_tr = sum(r["n_trained"] for r in got)
        print(
            f"{name}: {len(got)} runs from {len(files)} cells; {n_tr} trained clicks, "
            f"{mism} where the counts do not reproduce Test's F-beta",
            file=sys.stderr,
        )
    pd.DataFrame(all_rows).round(5).to_csv(args.out / "rules.csv", index=False)
    pd.DataFrame(all_curves).round(5).to_csv(args.out / "curves.csv", index=False)
    pd.DataFrame(per_run).round(5).to_csv(args.out / "per_run.csv", index=False)
    pd.DataFrame(firsts).round(5).to_csv(args.out / "first_clicks.csv", index=False)
    pd.DataFrame(clicks).round(5).to_csv(args.out / "clicks.csv", index=False)
    print(f"wrote rules.csv, curves.csv, per_run.csv, first_clicks.csv, clicks.csv -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
