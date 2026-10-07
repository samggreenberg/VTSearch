#!/usr/bin/env python3
"""#4583: the calibration split and fold count under the labels line, at the three presets.

Reads the twelve arms ``launch_calsplit_4583.sh`` writes (fraction {0.3, 0.5} x
calibrate count {2, 4} x beta {0.25, 1, 4}) and scores **the objective** (#4427):
the F-beta, at the arm's own beta, of the withheld half above the threshold the
app holds.  Two readings, as the State of the App gives them:

* **unchecked**, per vote: the filled curve.  Until a run shows a detector the
  user still sees the typed query's own set, so those votes score the text
  sort's line at the arm's preset (``text_fbeta_b*`` in the click-0 baseline),
  and every curve leaves the same dot.
* **after the check**: the run's last ``phase == "check"`` row - the line the
  weak-separation check (#4496) leaves the user with.  A run the check never
  touched keeps its vote-150 value.

Contrasts, within each beta, paired on (category, seed):

* ``split``  f05k2 - f03k2   (0.5 held out against the shipped 0.3)
* ``count``  f03k4 - f03k2   (four folds against the shipped two)
* ``both``   f05k4 - f03k2
* ``interaction`` (f05k4 - f05k2) - (f03k4 - f03k2)
* ``one fold``  f03k1 - f03k2   (beta 1/4 and 1 only; skipped where not run)

Each is reported as the mean paired difference with an SE clustered on the
category (five seeds of one class are not five independent runs), by window
and by object-size band, beside AP (a split should move the cut, not the
ranking) and the returned set's precision and size.

**The σ table (#4584).**  The per-run SD of each paired Δ-objective is what an
A/B on today's app is sized with; ``sigma.csv`` writes it per beta and window,
so ``preflight.sh``'s placeholder σ per beta can be replaced by a measured one.

    python analyze_calsplit_4583.py --base /expscratch/$USER/calsplit-4583 \\
        --baseline /expscratch/$USER/progression-4184-h0.01/text_baseline.csv --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = ["category", "seed"]
BETAS: dict[str, float] = {"b025": 0.25, "b1": 1.0, "b4": 4.0}
ARMS = [f"f{f}k{k}_{b}" for f in ("03", "05") for k in ("2", "4") for b in BETAS]
#: One fold against two, added after the grid because #4582 flagged it at beta 1/4
#: (above the old line); run at 1/4 and 1 only.
ARMS += ["f03k1_b025", "f03k1_b1"]
CONTRASTS: dict[str, tuple[str, str]] = {
    "split": ("f05k2", "f03k2"),
    "count": ("f03k4", "f03k2"),
    "both": ("f05k4", "f03k2"),
    "one fold": ("f03k1", "f03k2"),
}
WINDOWS: dict[str, tuple[int, int]] = {
    "votes 1-30": (1, 30),
    "votes 31-100": (31, 100),
    "votes 101-150": (101, 150),
    "votes 1-150": (1, 150),
    "vote 150": (150, 150),
}
HORIZON = 150


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """The objective's rule (``objective.py`` / ``fbeta_from_rates``): nothing right scores 0."""
    p = np.nan_to_num(np.asarray(precision, dtype=float))
    r = np.asarray(recall, dtype=float)
    b2 = beta * beta
    den = b2 * p + r
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    return np.where(np.isnan(r), np.nan, out)


def band_of(category: pd.Series) -> pd.Series:
    return category.astype(str).str.rsplit("@", n=1).str[-1]


def arm_tables(
    frame: pd.DataFrame, anchor: pd.Series, beta: float, complete: bool
) -> dict[str, pd.DataFrame | pd.Series]:
    """One arm's per-run matrices: the filled objective, AP and the returned set, plus the after-check row.

    *complete*: the arm wrote every cell with none lost, so a run with no row
    never found a positive (starved) and scores the typed query's line
    throughout.  Otherwise a run with no row may simply not have been written
    yet, and is left out: filling it would put a Δ of exactly 0 into every
    contrast and shrink both the effect and its SD.
    """
    is_check = frame["phase"].fillna("").astype(str).str.strip() == "check" if "phase" in frame else False
    votes = frame[~is_check]
    if "app_trained" in votes.columns:
        votes = votes[votes["app_trained"].fillna(0).astype(int) == 1]
    votes = votes[votes["t"] <= HORIZON].copy()
    votes["F"] = fbeta(votes["precision"], votes["recall"], beta)
    runs = pd.MultiIndex.from_frame(frame[RUN].drop_duplicates())
    if complete:
        runs = runs.union(anchor.index)
    out: dict[str, pd.DataFrame | pd.Series] = {}
    shown = None
    for key, col in (("F", "F"), ("ap", "average_precision"), ("precision", "precision"), ("k", "n_flagged")):
        wide = votes.pivot_table(index=RUN, columns="t", values=col, aggfunc="last")
        wide = wide.reindex(index=runs, columns=range(HORIZON + 1))
        if shown is None:
            shown = wide.notna()
        if key == "F":
            wide[0] = anchor.reindex(runs).to_numpy()
            wide = wide.ffill(axis=1)
        out[key] = wide
    out["shown"] = shown
    # After the check: the last check row per run, else the run's own vote-150 value.
    checked = frame[is_check] if isinstance(is_check, pd.Series) else frame.iloc[0:0]
    after = out["F"][HORIZON].copy()
    k_after = out["k"].ffill(axis=1)[HORIZON].copy()
    p_after = out["precision"].ffill(axis=1)[HORIZON].copy()
    touched = pd.Series(False, index=runs)
    if len(checked):
        last = checked.sort_values("t").groupby(RUN).tail(1).set_index(RUN)
        f_last = pd.Series(fbeta(last["precision"], last["recall"], beta), index=last.index)
        after.loc[f_last.index] = f_last
        k_after.loc[last.index] = last["n_flagged"].astype(float)
        p_after.loc[last.index] = last["precision"].astype(float)
        touched.loc[last.index] = True
    out["after"], out["k_after"], out["p_after"], out["touched"] = after, k_after, p_after, touched
    return out


def window_mean(m: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    return m[[c for c in m.columns if lo <= c <= hi]].mean(axis=1)


def clustered(d: pd.Series) -> tuple[float, float, float, int]:
    """Mean, SE clustered on category, per-run SD, n."""
    d = d.dropna()
    if d.empty:
        return np.nan, np.nan, np.nan, 0
    g = d.groupby(level="category").mean()
    se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
    return float(d.mean()), se, float(d.std(ddof=1)), int(d.size)


#: Each run's paired Δ-objective after the check, per (beta, contrast): the
#: report's per-run figure.  Filled by :func:`contrasts`.
RUN_DELTAS: list[pd.DataFrame] = []


def contrasts(tables: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, sig = [], []
    RUN_DELTAS.clear()
    for btag in BETAS:
        for name, (a, b) in {**CONTRASTS, "interaction": ("", "")}.items():
            if name == "interaction":
                need = [f"f05k4_{btag}", f"f05k2_{btag}", f"f03k4_{btag}", f"f03k2_{btag}"]
                if not all(x in tables for x in need):
                    continue

                def diff(key, fn, need=need):
                    v = [fn(tables[x][key]) for x in need]
                    return (v[0] - v[1]) - (v[2] - v[3])
            else:
                aa, bb = f"{a}_{btag}", f"{b}_{btag}"
                if aa not in tables or bb not in tables:
                    continue

                def diff(key, fn, aa=aa, bb=bb):
                    return fn(tables[aa][key]) - fn(tables[bb][key])

            reads: dict[str, pd.Series] = {}
            for wname, (lo, hi) in WINDOWS.items():
                reads[f"objective, {wname}"] = diff("F", lambda m, lo=lo, hi=hi: window_mean(m, lo, hi))
                reads[f"AP, {wname}"] = diff("ap", lambda m, lo=lo, hi=hi: window_mean(m, lo, hi))
            reads["objective, after the check"] = diff("after", lambda s: s)
            RUN_DELTAS.append(
                reads["objective, after the check"]
                .rename("delta")
                .reset_index()
                .assign(beta=BETAS[btag], contrast=name)
            )
            reads["returned set size, after the check"] = diff("k_after", lambda s: s)
            reads["precision, after the check"] = diff("p_after", lambda s: s)
            for read, d in reads.items():
                bands = {"all": d}
                bands.update(
                    {
                        f"band={bnd}": d[band_of(d.index.get_level_values("category").to_series()).to_numpy() == bnd]
                        for bnd in ("small", "medium", "large")
                    }
                )
                for stratum, dd in bands.items():
                    mean, se, sd, n = clustered(dd)
                    rows.append(
                        {
                            "beta": BETAS[btag],
                            "contrast": name,
                            "read": read,
                            "stratum": stratum,
                            "n": n,
                            "delta": mean,
                            "se": se,
                            "sd": sd,
                        }
                    )
                if read.startswith("objective") and name != "interaction":
                    mean, se, sd, n = clustered(d)
                    sig.append({"beta": BETAS[btag], "contrast": name, "read": read, "n": n, "sd_paired": sd})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["resolved"] = out["delta"].abs() > 2 * out["se"]
    return out, pd.DataFrame(sig)


def levels(tables: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for arm, t in tables.items():
        rows.append(
            {
                "arm": arm,
                "runs": int(t["F"].shape[0]),
                "objective, votes 1-150": float(window_mean(t["F"], 1, 150).mean()),
                "objective, vote 150": float(t["F"][HORIZON].mean()),
                "objective, after the check": float(t["after"].mean()),
                "AP, vote 150": float(t["ap"].ffill(axis=1)[HORIZON].mean()),
                "precision, after the check": float(t["p_after"].mean()),
                "returned median, after the check": float(t["k_after"].median()),
                "returned over 200, after the check": float((t["k_after"] > 200).mean()),
                "check touched": float(t["touched"].mean()),
                "shown by vote 30": float(t["shown"].loc[:, 1:30].any(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def premise(arm: str, frame: pd.DataFrame) -> list[str]:
    """The arm ran what its name says: read back off its own rows."""
    f = {"03": 0.3, "05": 0.5}[arm[1:3]]
    k = int(arm[4])
    b = BETAS[arm.split("_")[1]]
    bad = []
    for col, want in (("calibration_fraction", f), ("calibrate_count", k), ("beta", b)):
        if col not in frame.columns:
            bad.append(f"{arm}: no `{col}` column")
            continue
        got = sorted({round(float(x), 6) for x in pd.to_numeric(frame[col], errors="coerce").dropna().unique()})
        if got != [round(float(want), 6)]:
            bad.append(f"{arm}: `{col}` is {got}, expected {want}")
    if "prevalence_arm" in frame.columns:
        pools = sorted({str(x) for x in frame["prevalence_arm"].dropna().unique()})
        if pools != ["haystack_0.01"]:
            bad.append(f"{arm}: pool {pools}, expected haystack_0.01")
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    import _cells_io  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)

    base = _cells_io.legacy_datasets(pd.read_csv(args.baseline)).groupby(RUN).mean(numeric_only=True)
    tables: dict[str, dict] = {}
    prov: dict[str, dict] = {}
    problems: list[str] = []
    for arm in [a for a in args.arms.split(",") if a]:
        d = args.base / arm / "results"
        if not (d / "cells").exists():
            problems.append(f"{arm}: no cells")
            continue
        frame, p = _cells_io.load_arm(d, keep_check=True)
        prov[arm] = {"load": _cells_io.describe_load(p), "n_files": p.get("n_files")}
        if frame.empty:
            problems.append(f"{arm}: no rows")
            continue
        problems += premise(arm, frame)
        btag = arm.split("_")[1]
        anchor = base[f"text_fbeta_{btag}"]
        lost = len(p.get("zero_byte") or []) + len(p.get("unreadable") or [])
        complete = p.get("n_files") == len(anchor) and lost == 0
        prov[arm]["complete"] = bool(complete)
        if not complete:
            problems.append(f"{arm}: {p.get('n_files')} of {len(anchor)} cells, {lost} lost - starved runs not filled")
        tables[arm] = arm_tables(frame, anchor, BETAS[btag], complete)
        print(f"{arm}: {prov[arm]['load']}", flush=True)
    paired, sigma = contrasts(tables)
    # The region phase (the launcher's ``region`` mode): the shipped 0.5 of the
    # patch space against 0.3, at beta 1, on #4219's 72 classes x 2 seeds.  Its
    # runs are a subset of the baseline's, so starved runs are never filled.
    region = args.base / "region"
    if (region / "r_f05_b1" / "results" / "cells").exists() and (region / "r_f03_b1" / "results" / "cells").exists():
        for arm in ("r_f05_b1", "r_f03_b1"):
            frame, p = _cells_io.load_arm(region / arm / "results", keep_check=True)
            prov[arm] = {"load": _cells_io.describe_load(p), "n_files": p.get("n_files"), "complete": False}
            tables[arm] = arm_tables(frame, base["text_fbeta_b1"], 1.0, complete=False)
            print(f"{arm}: {prov[arm]['load']}", flush=True)
        a, b = tables["r_f03_b1"], tables["r_f05_b1"]
        reads = {
            f"objective, {w}": window_mean(a["F"], lo, hi) - window_mean(b["F"], lo, hi)
            for w, (lo, hi) in WINDOWS.items()
        }
        reads.update(
            {f"AP, {w}": window_mean(a["ap"], lo, hi) - window_mean(b["ap"], lo, hi) for w, (lo, hi) in WINDOWS.items()}
        )
        reads["objective, after the check"] = a["after"] - b["after"]
        reads["returned set size, after the check"] = a["k_after"] - b["k_after"]
        reads["precision, after the check"] = a["p_after"] - b["p_after"]
        rrows = []
        for read, d in reads.items():
            mean, se, sd, n = clustered(d)
            rrows.append(
                {
                    "beta": 1.0,
                    "contrast": "region: 0.3 - 0.5",
                    "read": read,
                    "stratum": "all",
                    "n": n,
                    "delta": mean,
                    "se": se,
                    "sd": sd,
                }
            )
        rr = pd.DataFrame(rrows)
        rr["resolved"] = rr["delta"].abs() > 2 * rr["se"]
        paired = pd.concat([paired, rr], ignore_index=True)
    lv = levels(tables)
    if RUN_DELTAS:
        pd.concat(RUN_DELTAS, ignore_index=True).to_csv(args.out / "run_deltas.csv", index=False, float_format="%.4g")
    paired.to_csv(args.out / "paired.csv", index=False, float_format="%.5g")
    sigma.to_csv(args.out / "sigma.csv", index=False, float_format="%.4g")
    lv.to_csv(args.out / "levels.csv", index=False, float_format="%.4g")
    # Per-click means, for the figures.
    curves = pd.DataFrame({arm: t["F"].mean(axis=0) for arm, t in tables.items()})
    curves.index.name = "t"
    curves.to_csv(args.out / "curves.csv", float_format="%.4g")
    (args.out / "provenance.json").write_text(json.dumps({"arms": prov, "problems": problems}, indent=2))

    pd.set_option("display.width", 250)
    lines = ["# #4583 - machine summary", ""]
    lines += [f"- `{a}`: {v['load']}" for a, v in prov.items()]
    if problems:
        lines += ["", "## Premise failures", "", *[f"- {x}" for x in problems]]
    lines += ["", "## Levels", "", lv.round(3).to_string(index=False), ""]
    head = paired[
        (paired.stratum == "all")
        & paired.read.isin(
            [
                "objective, votes 1-150",
                "objective, vote 150",
                "objective, after the check",
                "AP, vote 150",
                "returned set size, after the check",
            ]
        )
    ]
    lines += ["## Contrasts (all bands)", "", head.round(4).to_string(index=False), ""]
    if not sigma.empty:
        s = sigma.pivot_table(index=["beta", "read"], columns="contrast", values="sd_paired")
        lines += ["## Per-run SD of the paired Δ-objective (#4584)", "", s.round(3).to_string(), ""]
    (args.out / "REPORT_machine.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
