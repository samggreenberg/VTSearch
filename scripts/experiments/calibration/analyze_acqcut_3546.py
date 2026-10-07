#!/usr/bin/env python3
"""#3546: Autopilot's acquisition cut, eight rules at three presets, against today's cut (#4359 reuses it).

Reads the 24 arms ``launch_acqcut_3546.sh`` writes (``<rule>_<beta>``) and scores
each rule against the control (``ctl``: line - 4 Inclusion steps) at the same
beta, paired on (category, seed), SE clustered on category.  The objective's
readings are the #4583 analyzer's (:mod:`analyze_calsplit_4583`): the filled
curve over votes, and the line after the weak check.  Beside them, what an
acquisition rule is *for*:

* **Goods found** by vote 150 (the harvest);
* the **Hard picks' share Good** (from the ``__picks`` side frame);
* **where the cut sits**: the median pool percentile of ``acq_threshold`` at vote 150;
* **where Hard ends**: the last vote in phase ``hard``.

    python analyze_acqcut_3546.py --base /expscratch/$USER/acqcut-3546 \\
        --baseline /expscratch/$USER/progression-4184-h0.01/text_baseline.csv --out OUT

#4359's Smart bound reads the same measures: ``--base .../smartgate-4359 --rules app,never --control app``.
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

import analyze_calsplit_4583 as A  # noqa: E402

RULES = ("ctl", "line", "off2", "off6", "pin985", "inc0", "tp50", "tp25")
BETAS = A.BETAS
CONTROL = "ctl"
RUN = A.RUN
HORIZON = A.HORIZON


def harvest(frame: pd.DataFrame, cells_dir: Path) -> pd.DataFrame:
    """Per run: Goods by vote 150, the acquisition cut's pool percentile at 150, Hard's end, Hard picks' share Good."""
    votes = frame[frame["phase"].fillna("").astype(str).str.strip() != "check"]
    votes = votes[votes["t"] <= HORIZON]
    last = votes.sort_values("t").groupby(RUN).tail(1).set_index(RUN)
    out = pd.DataFrame(index=last.index)
    out["goods"] = pd.to_numeric(last["n_good"], errors="coerce")
    out["acq_pct"] = pd.to_numeric(last.get("acq_pool_percentile"), errors="coerce")
    out["line_pct"] = pd.to_numeric(last.get("report_pool_percentile"), errors="coerce")
    hard = votes[votes["phase"].astype(str) == "hard"]
    out["hard_end"] = hard.groupby(RUN)["t"].max().reindex(out.index)
    parts = []
    import _cells_io  # noqa: PLC0415

    for p in _cells_io.side_frame_files(cells_dir, "__picks"):
        try:
            k = pd.read_csv(p, usecols=["category", "seed", "phase", "picked_label"])
        except (ValueError, OSError, pd.errors.EmptyDataError):
            continue
        parts.append(k[k["phase"] == "hard"])
    if parts:
        picks = pd.concat(parts, ignore_index=True)
        g = picks.groupby(RUN)["picked_label"]
        out["hard_picks"] = g.size().reindex(out.index)
        out["hard_good_share"] = g.mean().reindex(out.index)
    return out


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    import _cells_io  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=3, help="the grid's seeds: the anchor is cut to them")
    ap.add_argument("--rules", default=",".join(RULES), help="arm stems, comma-separated (#4359 reads app,never)")
    ap.add_argument("--control", default=CONTROL, help="the stem every rule is paired against")
    args = ap.parse_args(list(argv) if argv is not None else None)
    rules = [r for r in args.rules.split(",") if r]
    control = args.control
    args.out.mkdir(parents=True, exist_ok=True)

    base = _cells_io.legacy_datasets(pd.read_csv(args.baseline))
    base = base[base["seed"] < args.seeds].groupby(RUN).mean(numeric_only=True)
    tables: dict[str, dict] = {}
    harv: dict[str, pd.DataFrame] = {}
    prov: dict[str, dict] = {}
    problems: list[str] = []
    for rule in rules:
        for btag in BETAS:
            arm = f"{rule}_{btag}"
            d = args.base / arm / "results"
            if not (d / "cells").exists():
                problems.append(f"{arm}: no cells")
                continue
            frame, p = _cells_io.load_arm(d, keep_check=True)
            anchor = base[f"text_fbeta_{btag}"]
            lost = len(p.get("zero_byte") or []) + len(p.get("unreadable") or [])
            complete = p.get("n_files") == len(anchor) and lost == 0
            prov[arm] = {"load": _cells_io.describe_load(p), "complete": bool(complete)}
            if not complete:
                problems.append(f"{arm}: {p.get('n_files')} of {len(anchor)} cells, {lost} lost")
            if frame.empty:
                continue
            tables[arm] = A.arm_tables(frame, anchor, BETAS[btag], complete)
            harv[arm] = harvest(frame, d / "cells")
            print(f"{arm}: {prov[arm]['load']}", flush=True)

    rows = []
    for btag, beta in BETAS.items():
        ctl = f"{control}_{btag}"
        if ctl not in tables:
            continue
        for rule in rules:
            arm = f"{rule}_{btag}"
            if rule == control or arm not in tables:
                continue
            a, c = tables[arm], tables[ctl]
            reads: dict[str, pd.Series] = {}
            for w, (lo, hi) in A.WINDOWS.items():
                reads[f"objective, {w}"] = A.window_mean(a["F"], lo, hi) - A.window_mean(c["F"], lo, hi)
            reads["objective, after the check"] = a["after"] - c["after"]
            reads["AP, vote 150"] = A.window_mean(a["ap"], 150, 150) - A.window_mean(c["ap"], 150, 150)
            reads["returned set size, after the check"] = a["k_after"] - c["k_after"]
            reads["precision, after the check"] = a["p_after"] - c["p_after"]
            for col in ("goods", "hard_good_share", "hard_picks", "hard_end"):
                if col in harv[arm] and col in harv[ctl]:
                    reads[col] = harv[arm][col] - harv[ctl][col]
            for read, dd in reads.items():
                mean, se, sd, n = A.clustered(dd)
                rows.append({"beta": beta, "rule": rule, "read": read, "n": n, "delta": mean, "se": se, "sd": sd})
    paired = pd.DataFrame(rows)
    if not paired.empty:
        paired["resolved"] = paired["delta"].abs() > 2 * paired["se"]

    lv = []
    for arm, t in tables.items():
        h = harv[arm]
        lv.append(
            {
                "arm": arm,
                "runs": int(t["F"].shape[0]),
                "objective, votes 1-150": float(A.window_mean(t["F"], 1, 150).mean()),
                "objective, after the check": float(t["after"].mean()),
                "AP, vote 150": float(t["ap"].ffill(axis=1)[HORIZON].mean()),
                "goods by 150": float(h["goods"].mean()),
                "hard picks Good": float(h["hard_good_share"].mean()) if "hard_good_share" in h else np.nan,
                "hard picks": float(h["hard_picks"].mean()) if "hard_picks" in h else np.nan,
                "hard ends (median vote)": float(h["hard_end"].median()),
                "acq percentile @150": float(h["acq_pct"].median()),
                "line percentile @150": float(h["line_pct"].median()),
                "returned median, after the check": float(t["k_after"].median()),
            }
        )
    levels = pd.DataFrame(lv)
    curves = pd.DataFrame({arm: t["F"].mean(axis=0) for arm, t in tables.items()})
    curves.index.name = "t"
    paired.to_csv(args.out / "paired.csv", index=False, float_format="%.5g")
    levels.to_csv(args.out / "levels.csv", index=False, float_format="%.4g")
    curves.to_csv(args.out / "curves.csv", float_format="%.4g")
    (args.out / "provenance.json").write_text(json.dumps({"arms": prov, "problems": problems}, indent=2))

    pd.set_option("display.width", 250)
    lines = [f"# {args.base.name} - machine summary", ""]
    if problems:
        lines += ["## Problems", "", *[f"- {x}" for x in problems], ""]
    lines += ["## Levels", "", levels.round(3).to_string(index=False), ""]
    if not paired.empty:
        for read in ("objective, votes 1-150", "objective, after the check", "goods", "AP, vote 150"):
            q = paired[paired.read == read].copy()
            q["v"] = q.apply(lambda r: f"{r.delta:+.3f} ± {r.se:.3f}" + ("*" if r.resolved else ""), axis=1)
            lines += [
                f"## {read} (rule − ctl)",
                "",
                q.pivot_table(index="rule", columns="beta", values="v", aggfunc="first").to_string(),
                "",
            ]
    (args.out / "REPORT_machine.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
