#!/usr/bin/env python
"""Price a Find that gives the region centroid once the typed query has missed, on the Region path (#4800).

Since #4758 a typed-query detector's Find gives the typed query's own set until the label quota. On the Region path
#4758 read the Goods' centroid, built from a Good's own box, as the better set on the votes where the typed query
has **missed**: the Bads quota met before the Goods quota (``n_good < GOOD_QUOTA`` and ``n_bad >= BAD_QUOTA``).
``missed_query_4800.sh`` runs Region's sessions at beta 1 with every candidate set as tagged rows at every vote under
the quota (``CALIB_CENTROID_LINE_VARIANTS``): the typed query's set at each balance (``query@<beta>``) and the
centroid at the app's rule over the withheld half and over the unvoted media (``<rule>~unvoted@<beta>``).

Neither set picks and Autopilot's opening reads no balance, so every design shares every session and differs only in
what Find returns on the votes under the quota with a Good. A design is a rule over the votes in hand: on the votes it
names, Find gives the centroid; elsewhere the typed query's set, as today. Its difference to today at a vote is the
centroid's F-beta minus the query's there, and 0 on every other vote, so a session mean over votes 1-150 is exact
from a run cut at H votes once no vote past H could be named (``checks.json`` says the latest one each run had). A
vote under the quota with a Good and fewer than 3 Goods holds at most 2 Goods and 15 Bads, so every design but
``centroid`` and ``oracle`` is exact from H = 17.

Designs (``DESIGNS``): ``missed`` is #4800's rule; ``missed_1good`` takes the centroid only with 1 Good;
``bads3`` / ``bads2`` / ``bads1`` fall back earlier (3, 2 or 1 Bads before the Goods quota); ``centroid`` is the centroid at every
vote with a Good (the design before #4758, inexact past H when a run still waits for its 4th Bad there); ``oracle``
takes the better set at every such vote (the headroom, not a rule).

Writes under ``--out``: ``cells.csv`` (one row per run x corpus x beta), ``summary.csv`` (each design's paired mean
difference over votes 1-10, 1-50 and 1-150, SEs clustered on the category), ``votes.csv`` (every vote under the quota
with a Good) and ``votes_under_quota.csv`` (those votes by the Goods and Bads in hand, ``goods`` 3 standing for 3 or
more), ``curves.csv`` (per vote: the share of runs whose query has missed, both sets' F-beta on those runs, and
each design's mean difference), ``checks.json`` and ``summary.md``.

Usage::

    python missed_query_4800.py --run <run dir> --out <dir> [--horizon 20] \\
        [--same-session <run dir at another beta> ...] [--reference <production run dir at the run's beta>]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import _cells_io  # noqa: E402
from analyze_centroid_line_4732 import _carry, _mean_se  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from vtscore.detectors.centroid_head import CENTROID_LINE_RULE  # noqa: E402
from vtscore.detectors.label_quota import BAD_QUOTA, GOOD_QUOTA, TIER_CENTROID, label_quota  # noqa: E402
from vtscore.eval.voting_iterations import QUERY_LINE_VARIANT, centroid_line_tag  # noqa: E402

BETAS = (0.25, 1.0, 4.0)
SPANS = {"1-10": 10, "1-50": 50, "1-150": 150}
FULL = 150
CORPORA = {"withheld": CENTROID_LINE_RULE, "labelled": f"{CENTROID_LINE_RULE}~unvoted"}
COLS = ["t", "beta", "fbeta", "precision", "recall", "n_flagged", "detector_tier", "gmm_variant", "schedule", "phase"]
PICK_COLS = ["seed", "dataset", "category", "embedder", "t", "phase", "picked_id", "picked_label", "n_good", "n_bad"]


def _missed(g: np.ndarray, b: np.ndarray, bads: int = BAD_QUOTA) -> np.ndarray:
    return (g < GOOD_QUOTA) & (b >= bads)


#: Each design's votes (among those under the quota with a Good) on which Find gives the centroid.
DESIGNS = {
    "missed": lambda g, b: _missed(g, b),
    "missed_1good": lambda g, b: _missed(g, b) & (g == 1),
    "bads3": lambda g, b: _missed(g, b, 3),
    "bads2": lambda g, b: _missed(g, b, 2),
    "bads1": lambda g, b: _missed(g, b, 1),
    "centroid": lambda g, b: np.ones_like(g, dtype=bool),
}
ORACLE = "oracle"


def _stem(path: str) -> tuple[str, str]:
    ext = ".csv.gz" if path.endswith(".csv.gz") else ".csv"
    return path[: -len(ext)], ext


def _picks(path: str) -> pd.DataFrame | None:
    stem, ext = _stem(path)
    try:
        p = pd.read_csv(f"{stem}__picks{ext}", usecols=lambda c: c in PICK_COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    return None if p.empty else p.sort_values("t", kind="stable").drop_duplicates("t", keep="last")


def read_cell(args: tuple[str, int, float]) -> dict | None:
    """One cell -> the votes in hand and every tagged series over votes 1..H; ``None`` when unreadable."""
    path, horizon, run_beta = args
    picks = _picks(path)
    if picks is None:
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=COLS)
    except (OSError, ValueError):
        return None
    if len(df):
        df = df[~_cells_io.check_rows(df)]
    counts = picks.set_index("t")[["n_good", "n_bad"]].reindex(range(1, horizon + 1)).ffill().fillna(0).astype(int)
    g, b = counts["n_good"].to_numpy(), counts["n_bad"].to_numpy()
    # Under the quota with a Good: where a fallback could give the centroid.
    open_ = np.array([label_quota(int(x), int(y)).tier == TIER_CENTROID for x, y in zip(g, b)])
    # Under the quota, before the first Good too: where Find gives the typed query's set.
    under = np.array([label_quota(int(x), int(y)).tier != "trained" for x, y in zip(g, b)])
    tag = df["gmm_variant"].fillna("").astype(str) if len(df) else pd.Series(dtype=str)

    def series(name: str, beta: float) -> tuple[np.ndarray, np.ndarray]:
        sel = (tag == centroid_line_tag(name, beta)).to_numpy() if len(df) else np.zeros(0, dtype=bool)
        rows = df[sel]
        rows = rows[rows["t"] <= horizon].groupby("t", as_index=False).last()
        have = np.zeros(horizon, dtype=bool)
        have[rows["t"].to_numpy(dtype=int) - 1] = True
        v = _carry(rows["t"].to_numpy(dtype=float), rows["fbeta"].astype(float).fillna(0.0).to_numpy(), horizon)
        return v, have

    first = picks.iloc[0]
    out = {
        "dataset": first["dataset"],
        "category": first["category"],
        "embedder": first.get("embedder", ""),
        "seed": int(first["seed"]),
        "_g": g,
        "_b": b,
        "_open": open_,
        "n_votes": int(len(picks)),
        "missing_rows": 0,
    }
    for beta in BETAS:
        q, qh = series(QUERY_LINE_VARIANT, beta)
        out["missing_rows"] += int((under & ~qh).sum())
        out[f"_q_{beta:g}"] = q
        for corpus, rule in CORPORA.items():
            c, ch = series(rule, beta)
            out["missing_rows"] += int((open_ & ~ch).sum())
            out[f"_c_{corpus}_{beta:g}"] = c
    # The untagged rows are the session's Find: the typed query's set at the run's balance until the quota.
    if len(df):
        base = _cells_io._base_rows(df)
        base = base[(base["detector_tier"].astype(str) == "query") & (base["t"] <= horizon)]
        base = base.groupby("t", as_index=False).last()
        tq = out[f"_q_{run_beta:g}"]
        bt = base["t"].to_numpy(dtype=int)
        out["query_row_mismatch"] = int((~np.isclose(base["fbeta"].astype(float).fillna(0.0), tq[bt - 1])).sum())
        out["query_rows"] = int(len(base))
    else:
        out["query_row_mismatch"], out["query_rows"] = 0, 0
    return out


def designs_for(r: dict, beta: float, corpus: str) -> dict[str, np.ndarray]:
    """Each design's per-vote difference to today (the typed query throughout) over votes 1..H."""
    q, c = r[f"_q_{beta:g}"], r[f"_c_{corpus}_{beta:g}"]
    g, b, open_ = r["_g"], r["_b"], r["_open"]
    out = {name: np.where(open_ & rule(g, b), c - q, 0.0) for name, rule in DESIGNS.items()}
    out[ORACLE] = np.where(open_, np.maximum(c - q, 0.0), 0.0)
    return out


def span_means(d: np.ndarray) -> dict[str, float]:
    """Session means over each span; a vote past H is 0 (no design names one, ``max_named_t`` checks it)."""
    full = np.concatenate([d, np.zeros(max(0, FULL - len(d)))])
    return {span: float(full[:n].sum() / n) for span, n in SPANS.items()}


def same_session(run: Path, others: list[Path], horizon: int) -> list[dict]:
    """Whether each other run's cells (another balance) cast the same votes through the quota as this run's."""
    mine = {Path(f).name: f for f in map(str, _cells_io.main_frame_files(run / "results" / "cells"))}
    out = []
    for other in others:
        same = differ = 0
        first_diff: list[str] = []
        for f in map(str, _cells_io.main_frame_files(other / "results" / "cells")):
            name = Path(f).name
            if name not in mine:
                continue
            a, b = _picks(mine[name]), _picks(f)
            if a is None or b is None:
                continue
            # Through the vote that meets the quota: every vote a fallback could change, and the one after.
            last = _through_quota(a, horizon)
            ka = a[a["t"] <= last][["t", "picked_id"]].to_numpy().tolist()
            kb = b[b["t"] <= last][["t", "picked_id"]].to_numpy().tolist()
            if ka == kb:
                same += 1
            else:
                differ += 1
                first_diff.append(name)
        out.append({"run": str(other), "same": same, "differ": differ, "differ_cells": first_diff[:10]})
    return out


def _through_quota(picks: pd.DataFrame, horizon: int) -> int:
    """The vote that meets the quota (or H): the last vote any design here can change, and the pick it answers."""
    under = np.array([label_quota(int(x), int(y)).tier != "trained" for x, y in zip(picks["n_good"], picks["n_bad"])])
    last = int(picks["t"].to_numpy()[under].max()) + 1 if under.any() else 1
    return min(last, horizon)


def reference(run: Path, ref: Path, horizon: int) -> dict:
    """Against a production run at the same balance: the same session through the quota, and where the two part.

    The designs differ only on votes under the quota, so the run is today's session if every vote through the one
    that meets the quota is the reference's, and its untagged rows there are too. Past it the runs may part: a run
    cut at H votes has no budget for a check the reference prompts (``phase`` there says which).
    """
    mine = {Path(f).name: f for f in map(str, _cells_io.main_frame_files(run / "results" / "cells"))}
    same = differ = rows_same = rows_differ = parted = 0
    bad: list[str] = []
    part_phases: dict[str, int] = {}
    cols = ["t", "fbeta", "precision", "recall", "n_flagged", "detector_tier"]
    for f in map(str, _cells_io.main_frame_files(ref / "results" / "cells")):
        name = Path(f).name
        if name not in mine:
            continue
        a, b = _picks(mine[name]), _picks(f)
        if a is None or b is None:
            continue
        last = _through_quota(b, horizon)
        if (
            a[a["t"] <= last][["t", "picked_id"]].to_numpy().tolist()
            == b[b["t"] <= last][["t", "picked_id"]].to_numpy().tolist()
        ):
            same += 1
        else:
            differ += 1
            bad.append(name)
        m = a[a["t"] <= horizon].merge(b[b["t"] <= horizon], on="t", suffixes=("", "_ref"))
        split = m[m["picked_id"] != m["picked_id_ref"]]
        if len(split):
            parted += 1
            ph = str(b.set_index("t")["phase"].get(int(split["t"].min()), "")) if "phase" in b else ""
            part_phases[ph] = part_phases.get(ph, 0) + 1
        ra = _cells_io._base_rows(pd.read_csv(mine[name], usecols=lambda c: c in COLS, low_memory=False))
        rb = _cells_io._base_rows(pd.read_csv(f, usecols=lambda c: c in COLS, low_memory=False))
        ra = ra[(ra["t"] <= last) & ~_cells_io.check_rows(ra)][cols].reset_index(drop=True)
        rb = rb[(rb["t"] <= last) & ~_cells_io.check_rows(rb)][cols].reset_index(drop=True)
        if ra.shape == rb.shape and ra.astype(str).equals(rb.astype(str)):
            rows_same += 1
        else:
            rows_differ += 1
    return {
        "reference": str(ref),
        "picks_same_through_quota": same,
        "picks_differ_through_quota": differ,
        "differ_cells": bad[:10],
        "untagged_rows_same_through_quota": rows_same,
        "untagged_rows_differ_through_quota": rows_differ,
        "parted_after_quota_by_vote_H": parted,
        "reference_phase_where_they_part": part_phases,
    }


def breakdown(votes: pd.DataFrame) -> list[str]:
    """The votes under the quota with a Good, by the Goods in hand and whether the query has missed."""
    if votes.empty:
        return []
    v = votes.assign(diff=votes["centroid_fbeta"] - votes["query_fbeta"])
    v["missed"] = _missed(v["n_good"].to_numpy(), v["n_bad"].to_numpy())
    out = [
        "## Votes under the quota with a Good: the centroid minus the typed query",
        "",
        "| corpus | beta | Goods in hand | query missed | votes | typed query F | centroid F | difference | centroid better on |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (corpus, beta, ng, missed), h in v.groupby(["corpus", "beta", "n_good", "missed"]):
        if ng >= GOOD_QUOTA:
            ng = f"{GOOD_QUOTA}+"
        out.append(
            f"| {corpus} | {beta:g} | {ng} | {'yes' if missed else 'no'} | {len(h)} | {h['query_fbeta'].mean():.3f} "
            f"| {h['centroid_fbeta'].mean():.3f} | {h['diff'].mean():+.3f} | {(h['diff'] > 0).mean():.0%} |"
        )
    out += ["", "On the missed votes by the Bads in hand (all corpora and balances):", ""]
    out += [
        "| corpus | beta | Bads in hand | votes | typed query F | centroid F | difference |",
        "|---|---|---|---|---|---|---|",
    ]
    m = v[v["missed"]]
    for (corpus, beta, nb), h in m.groupby(
        ["corpus", "beta", pd.cut(m["n_bad"], [3, 5, 8, 11, 15], right=True)], observed=True
    ):
        if len(h):
            out.append(
                f"| {corpus} | {beta:g} | {nb} | {len(h)} | {h['query_fbeta'].mean():.3f} "
                f"| {h['centroid_fbeta'].mean():.3f} | {h['diff'].mean():+.3f} |"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run dir holding results/cells (the beta-1 run)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--run-beta", type=float, default=1.0, help="the run's own balance (its untagged rows')")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--same-session", type=Path, action="append", default=[], help="a run dir at another balance")
    ap.add_argument("--reference", type=Path, help="a production run at the run's balance (same commit's app)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in _cells_io.main_frame_files(args.run / "results" / "cells")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        got = list(pool.map(read_cell, [(f, args.horizon, args.run_beta) for f in files], chunksize=4))
    runs = [g for g in got if g is not None]
    n_bad_cells = len(got) - len(runs)
    print(f"{len(files)} cells, {n_bad_cells} unreadable", file=sys.stderr)

    # A design is exact past H only if it names no vote there: the latest vote any run's state could be named on.
    max_open_t = max((int(np.nonzero(r["_open"])[0].max()) + 1 for r in runs if r["_open"].any()), default=0)
    still_open = sum(1 for r in runs if r["_open"][-1])
    checks = {
        "cells": len(files),
        "unreadable": n_bad_cells,
        "runs": len(runs),
        "horizon": args.horizon,
        "missing_tagged_rows": int(sum(r["missing_rows"] for r in runs)),
        "query_rows": int(sum(r["query_rows"] for r in runs)),
        "query_row_mismatch_vs_query_at_run_beta": int(sum(r["query_row_mismatch"] for r in runs)),
        "latest_vote_under_quota_with_a_good": max_open_t,
        "runs_still_under_quota_with_a_good_at_H": still_open,
    }
    max_missed_t = max(
        (int(np.nonzero(r["_open"] & _missed(r["_g"], r["_b"], 2))[0].max()) + 1 for r in runs
         if (r["_open"] & _missed(r["_g"], r["_b"], 2)).any()),
        default=0,
    )  # fmt: skip
    checks["latest_vote_with_2_bads_before_the_goods_quota"] = max_missed_t
    if args.same_session:
        checks["same_session"] = same_session(args.run, args.same_session, args.horizon)
    if args.reference:
        checks["reference"] = reference(args.run, args.reference, args.horizon)

    cells, votes, curves = [], [], []
    for corpus in CORPORA:
        for beta in BETAS:
            acc: dict[str, list[np.ndarray]] = {k: [] for k in (*DESIGNS, ORACLE, "missed_now", "q", "c")}
            for r in runs:
                d = designs_for(r, beta, corpus)
                q, c = r[f"_q_{beta:g}"], r[f"_c_{corpus}_{beta:g}"]
                missed_now = r["_open"] & _missed(r["_g"], r["_b"])
                rec = {k: r[k] for k in ("dataset", "category", "embedder", "seed")}
                rec.update(
                    corpus=corpus,
                    beta=beta,
                    votes_open=int(r["_open"].sum()),
                    votes_missed=int(missed_now.sum()),
                    query_fbeta_missed=float(q[missed_now].mean()) if missed_now.any() else np.nan,
                    centroid_fbeta_missed=float(c[missed_now].mean()) if missed_now.any() else np.nan,
                )
                for name, dv in d.items():
                    for span, m in span_means(dv).items():
                        rec[f"d_{name}_{span}"] = m
                    acc[name].append(dv)
                acc["missed_now"].append(missed_now.astype(float))
                acc["q"].append(np.where(missed_now, q, np.nan))
                acc["c"].append(np.where(missed_now, c, np.nan))
                cells.append(rec)
                for t in np.nonzero(r["_open"])[0]:
                    votes.append(
                        {
                            "corpus": corpus,
                            "beta": beta,
                            "category": r["category"],
                            "seed": r["seed"],
                            "t": int(t) + 1,
                            "n_good": int(r["_g"][t]),
                            "n_bad": int(r["_b"][t]),
                            "query_fbeta": float(q[t]),
                            "centroid_fbeta": float(c[t]),
                        }
                    )
            if not runs:
                continue
            stack = {k: np.vstack(v) for k, v in acc.items()}
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a vote no run has missed on: NaN
                qm, cm = np.nanmean(stack["q"], axis=0), np.nanmean(stack["c"], axis=0)
            for t in range(args.horizon):
                row = {
                    "corpus": corpus,
                    "beta": beta,
                    "t": t + 1,
                    "share_missed": float(stack["missed_now"][:, t].mean()),
                    "query_fbeta_missed": float(qm[t]),
                    "centroid_fbeta_missed": float(cm[t]),
                }
                row.update({f"d_{name}": float(stack[name][:, t].mean()) for name in (*DESIGNS, ORACLE)})
                curves.append(row)

    frame = pd.DataFrame(cells)
    frame.to_csv(args.out / "cells.csv", index=False)
    vframe = pd.DataFrame(votes)
    vframe.to_csv(args.out / "votes.csv", index=False)
    if len(vframe):
        v = vframe.assign(
            diff=vframe["centroid_fbeta"] - vframe["query_fbeta"],
            missed=_missed(vframe["n_good"].to_numpy(), vframe["n_bad"].to_numpy()),
            goods=np.minimum(vframe["n_good"], GOOD_QUOTA),
        )
        agg = v.groupby(["corpus", "beta", "goods", "missed", "n_bad"], as_index=False).agg(
            votes=("diff", "size"),
            query_fbeta=("query_fbeta", "mean"),
            centroid_fbeta=("centroid_fbeta", "mean"),
            centroid_minus_query=("diff", "mean"),
            centroid_better_share=("diff", lambda d: float((d > 0).mean())),
        )
        agg.to_csv(args.out / "votes_under_quota.csv", index=False)
    pd.DataFrame(curves).to_csv(args.out / "curves.csv", index=False)

    summ = []
    for (corpus, beta), h in frame.groupby(["corpus", "beta"], sort=False):
        for name in (*DESIGNS, ORACLE):
            rec = {
                "corpus": corpus,
                "beta": beta,
                "design": name,
                "runs": len(h),
                "runs_with_missed_votes": int((h["votes_missed"] > 0).sum()),
                "votes_missed": int(h["votes_missed"].sum()),
            }
            for span in SPANS:
                m, se, n = _mean_se(h[f"d_{name}_{span}"], h["category"])
                rec[f"d_{span}"], rec[f"d_{span}_se"], rec["clusters"] = m, se, n
            summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    one = frame.drop_duplicates(["category", "seed"])
    lines = [
        f"# #4800: the region centroid once the typed query has missed (H = {args.horizon} votes)",
        "",
        "Each design minus today (the typed query's set at every vote under the quota), paired per run, session mean "
        "over each span; SE clusters on the category. `missed` gives the centroid once the Bads quota is met before "
        "the Goods quota; `oracle` takes the better set at every vote under the quota with a Good.",
        "",
        "| corpus | beta | design | runs | runs with missed votes | 1-10 | 1-50 | 1-150 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summ:
        spans = " | ".join(f"{r[f'd_{s}']:+.4f} ± {r[f'd_{s}_se']:.4f}" for s in SPANS)
        lines.append(
            f"| {r['corpus']} | {r['beta']:g} | {r['design']} | {r['runs']} | {r['runs_with_missed_votes']} | {spans} |"
        )
    lines += [
        "",
        f"- {len(one)} runs from {len(files)} cells ({n_bad_cells} unreadable); "
        f"{checks['missing_tagged_rows']} tagged rows missing on votes that should carry one.",
        f"- The latest vote under the quota with a Good: {max_open_t}; runs still there at vote {args.horizon}: "
        f"{still_open} (only `centroid` reads those votes). The latest vote with 2+ Bads before the Goods quota: "
        f"{max_missed_t}.",
        f"- query@{args.run_beta:g} against the untagged rows: {checks['query_row_mismatch_vs_query_at_run_beta']} "
        f"mismatches in {checks['query_rows']} rows.",
    ]
    for s in checks.get("same_session", []):
        lines.append(
            f"- Same session at another balance ({Path(s['run']).name}): {s['same']} cells same, {s['differ']} differ."
        )
    if "reference" in checks:
        ref = checks["reference"]
        lines.append(
            f"- Against {Path(ref['reference']).name}: the votes through the one that meets the quota are the same in "
            f"{ref['picks_same_through_quota']} runs, differ in {ref['picks_differ_through_quota']}; the untagged rows "
            f"there same in {ref['untagged_rows_same_through_quota']}, differ in "
            f"{ref['untagged_rows_differ_through_quota']}. Past the quota {ref['parted_after_quota_by_vote_H']} runs "
            f"part by vote {args.horizon}, where the reference is in phase {ref['reference_phase_where_they_part']}."
        )
    lines += ["", *breakdown(vframe)]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
