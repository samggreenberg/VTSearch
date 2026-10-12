#!/usr/bin/env python
"""Price what Find gives under the label quota in a typed-query session (#4758): the Goods' centroid or the typed query.

From the first Good until the label quota, a Find gives the Goods' centroid (#4643), while the screen shows the
typed query's own set at its line (#4605). Before the first Good, a Find gives nothing. The other design gives the
typed query's own set there instead: under the quota (``query_quota``), and also before the first Good
(``query_open``). Neither the centroid nor a Find picks, and the opening reads no balance, so the three designs share
every session and differ only on those votes, and only in what Find returns. One run prices all three, exactly:

- **today** is the run's centroid rows on the quota's votes: the tagged ``centroid_line:<rule>@<beta>`` rows
  (#4732), one series per corpus the line is drawn on. ``withheld`` is a Find on another collection, the run's
  untagged rule; ``labelled`` (``<rule>~unvoted``, #4779) is a Find on the collection being labelled.
- **the typed query's set** is ``text_baseline.csv``'s ``text_line_*`` for the same cell and seed: the typed
  query's display line at each balance (#4603), drawn over the whole haystack and scored on the withheld half,
  which is what the session shows through the opening. It does not move with the votes.

Each vote's tier comes from the pick log's ``n_good``/``n_bad`` after that vote (the app's
:func:`~vtscore.detectors.label_quota.label_quota`), so the votes before the first Good are counted, and so is a
run that found no Good at all (every run in every average, #4631). The difference curves are 0 from the quota on,
since every design gives the trained head there, so session means over votes 1-150 are exact from a run cut at H
votes when its quota fell inside H (``tier_past_h`` says when it did not; such a run carries its last tier state to
150, an approximation the summary counts).

Writes under ``--out``: ``cells.csv`` (one row per run x corpus x beta: tier sizes, the typed query's F-beta, the
centroid's mean F-beta on its own votes, and each design's difference to today over votes 1-10, 1-50 and 1-150),
``summary.csv`` (paired means with SEs clustered on the category), ``curves.csv`` (per vote: each design's mean
difference to today, the share of runs in each tier, and mean F-beta of what each design's Find returns on those
votes), and ``summary.md``.

Usage::

    python analyze_find_typed_query_4758.py --run <run dir> --text-baseline <text_baseline.csv> --out <dir> \\
        [--corpus withheld=count_gap_floor_skip25_b1 --corpus labelled=count_gap_floor_skip25_b1~unvoted] \\
        [--horizon 40]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cells_io  # noqa: E402
from analyze_centroid_line_4732 import TAG, _carry, _fbeta, _mean_se  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from vtscore.detectors.label_quota import TIER_CENTROID, TIER_NONE, TIER_TRAINED, label_quota  # noqa: E402

BETAS = {0.25: "b025", 1.0: "b1", 4.0: "b4"}
SPANS = {"1-10": 10, "1-50": 50, "1-150": 150}
FULL = 150
DESIGNS = ("query_quota", "query_open")
COLS = [
    "t", "beta", "fbeta", "precision", "recall", "n_flagged", "detector_tier", "gmm_variant", "pool_variant", "schedule",
    "phase",
]  # fmt: skip
#: A corpus rule naming the run's own untagged rows: the centroid at the run's commit, at the run's beta only.
APP_RULE = "app"
DEFAULT_CORPORA = ("withheld=count_gap_floor_skip25_b1", "labelled=count_gap_floor_skip25_b1~unvoted")


def tiers_from_picks(picks: pd.DataFrame, horizon: int) -> np.ndarray:
    """The tier Find gives after each of votes 1..*horizon*, from the pick log's running counts.

    A run cut before *horizon* (it cannot be: the opening never ends a run) or a vote missing from the log
    carries the last state.
    """
    p = picks.sort_values("t").drop_duplicates("t", keep="last")
    t = p["t"].to_numpy(dtype=int)
    tier = np.array([label_quota(int(g), int(b)).tier for g, b in zip(p["n_good"], p["n_bad"])], dtype=object)
    idx = np.searchsorted(t, np.arange(1, horizon + 1), side="right") - 1
    out = np.full(horizon, TIER_NONE, dtype=object)
    ok = idx >= 0
    out[ok] = tier[idx[ok]]
    return out


def _rule_rows(df: pd.DataFrame, rule: str, beta: float) -> pd.DataFrame:
    """The centroid's rows under *rule* at *beta*: the tagged variant, or with :data:`APP_RULE` the untagged rows."""
    if df.empty:
        return df
    if rule == APP_RULE:
        base = _cells_io._base_rows(df)
        base = base[(base["detector_tier"].astype(str) == TIER_CENTROID) & np.isclose(base["beta"].astype(float), beta)]
        return base
    return df[df["gmm_variant"].fillna("").astype(str) == f"{TAG}{rule}@{beta:g}"]


def read_cell(args: tuple[str, int, dict[str, str], tuple[float, ...]]) -> list[dict] | None:
    """One cell -> one row per (corpus, beta), with its per-vote curves; ``None`` when unreadable."""
    path, horizon, corpora, betas = args
    picks_path = Path(path.replace(".csv", "__picks.csv"))
    try:
        picks = pd.read_csv(picks_path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if picks.empty:
        return None
    first = picks.iloc[0]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=COLS)
    except (OSError, ValueError):
        return None
    if len(df):
        df = df[~_cells_io.check_rows(df)]
    tiers = tiers_from_picks(picks, horizon)
    counts = picks.sort_values("t").drop_duplicates("t", keep="last").set_index("t")[["n_good", "n_bad"]]
    counts = counts.reindex(range(1, horizon + 1)).ffill().fillna(0).astype(int)
    on_quota = tiers == TIER_CENTROID
    before = tiers == TIER_NONE
    # The quota's state at the last vote read stands for the votes after it, up to 150.
    past_h = tiers[-1] != TIER_TRAINED
    out = []
    for corpus, rule in corpora.items():
        for beta in betas:
            rows = _rule_rows(df, rule, beta)
            rows = rows[rows["t"] <= horizon].groupby("t", as_index=False).last() if len(rows) else rows
            if len(rows):
                cen = _carry(
                    rows["t"].to_numpy(dtype=float), rows["fbeta"].astype(float).fillna(0.0).to_numpy(), horizon
                )
            else:
                cen = np.zeros(horizon)
            if on_quota.any() and not len(rows):
                # Find gave the centroid but the run wrote no such rows: the rule was not priced here.
                return None
            out.append(
                {
                    "dataset": first["dataset"],
                    "category": first["category"],
                    "embedder": first.get("embedder", "siglip"),
                    "seed": int(first["seed"]),
                    "corpus": corpus,
                    "rule": rule,
                    "beta": beta,
                    "n_before": int(before.sum()),
                    "n_quota": int(on_quota.sum()),
                    "first_good": int(np.argmax(~before)) + 1 if (~before).any() else np.nan,
                    "quota_met": int(np.argmax(tiers == TIER_TRAINED)) + 1 if (tiers == TIER_TRAINED).any() else np.nan,
                    "no_good": bool(before.all()),
                    "tier_past_h": bool(past_h),
                    "centroid_fbeta_on_quota": float(cen[on_quota].mean()) if on_quota.any() else np.nan,
                    "_cen": cen,
                    "_on_quota": on_quota,
                    "_before": before,
                    "_n_good": counts["n_good"].to_numpy(),
                    "_n_bad": counts["n_bad"].to_numpy(),
                }
            )
    return out


def add_text(cells: list[dict], tb: pd.DataFrame) -> list[dict]:
    """The typed query's own set at the app's line, per cell, seed and beta; drop runs with no such row."""
    ref = tb.set_index(["dataset", "embedder", "category", "seed"])
    kept = []
    for r in cells:
        key = (r["dataset"], r["embedder"], r["category"], r["seed"])
        if key not in ref.index:
            continue
        row = ref.loc[key]
        tag = BETAS[r["beta"]]
        p, rec = float(row[f"text_line_precision_{tag}"]), float(row[f"text_line_recall_{tag}"])
        if not np.isfinite(rec):
            continue
        # A line that keeps nothing has no precision: an empty set scores 0 (#4631).
        r["text_fbeta"] = _fbeta(p, rec, r["beta"]) if np.isfinite(p) else 0.0
        kept.append(r)
    return kept


def differences(r: dict, horizon: int) -> dict[str, np.ndarray]:
    """Each design's per-vote difference to today over votes 1..150 (the state at H carried past it)."""
    text, cen = r["text_fbeta"], r["_cen"]
    d_quota = np.where(r["_on_quota"], text - cen, 0.0)
    d_open = d_quota + np.where(r["_before"], text, 0.0)
    pad = FULL - horizon
    if pad > 0:
        d_quota = np.concatenate([d_quota, np.full(pad, d_quota[-1])])
        d_open = np.concatenate([d_open, np.full(pad, d_open[-1])])
    return {"query_quota": d_quota, "query_open": d_open}


def breakdown(votes: pd.DataFrame, n_runs: int) -> list[str]:
    """Where the centroid wins: the votes under the quota by Goods in hand and by whether the query has missed.

    "Missed" is the Bads quota met before the Goods quota (``n_good < GOOD_QUOTA`` and ``n_bad >= BAD_QUOTA``):
    the typed query's top turned up the Bads first. The last column is what giving the centroid on those votes
    (and the typed query elsewhere) would add over votes 1-50 against the typed query throughout.
    """
    from vtscore.detectors.label_quota import BAD_QUOTA, GOOD_QUOTA  # noqa: PLC0415

    if votes.empty:
        return []
    v = votes.assign(diff=votes["query_fbeta"] - votes["centroid_fbeta"])
    v["missed"] = (v["n_good"] < GOOD_QUOTA) & (v["n_bad"] >= BAD_QUOTA)
    out = [
        "## Votes under the quota: typed query minus centroid",
        "",
        "| corpus | beta | Goods in hand | missed | votes | typed query F | centroid F | difference | centroid better on |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (corpus, beta, ng, missed), h in v.groupby(["corpus", "beta", "n_good", "missed"]):
        out.append(
            f"| {corpus} | {beta:g} | {ng} | {'yes' if missed else 'no'} | {len(h)} | {h['query_fbeta'].mean():.3f} "
            f"| {h['centroid_fbeta'].mean():.3f} | {h['diff'].mean():+.3f} | {(h['diff'] < 0).mean():.0%} |"
        )
    out += ["", "Centroid once the query has missed, typed query otherwise, minus the typed query throughout:", ""]
    for (corpus, beta), h in v.groupby(["corpus", "beta"]):
        gain = -h.loc[h["missed"], "diff"]
        g10 = -h.loc[h["missed"] & (h["t"] <= 10), "diff"]
        out.append(
            f"- {corpus}, beta {beta:g}: {gain.sum() / n_runs / 50:+.4f} over votes 1-50, "
            f"{g10.sum() / n_runs / 10:+.4f} over 1-10 ({int(h['missed'].sum())} votes)"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run dir holding results/cells")
    ap.add_argument("--text-baseline", type=Path, required=True)
    ap.add_argument("--corpus", action="append", default=[], help="label=<centroid line rule as tagged>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--betas", default="0.25,1,4", help="balances to read (an `app` corpus has the run's only)")
    args = ap.parse_args()
    betas = tuple(float(b) for b in args.betas.split(","))
    corpora = dict(spec.split("=", 1) for spec in (args.corpus or DEFAULT_CORPORA))
    args.out.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in _cells_io.main_frame_files(args.run / "results" / "cells")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        got = list(pool.map(read_cell, [(f, args.horizon, corpora, betas) for f in files], chunksize=8))
    n_bad_cells = sum(1 for g in got if g is None)
    raw = [r for g in got if g for r in g]
    tb = pd.read_csv(args.text_baseline)
    cells = add_text(raw, tb)
    n_runs = len({(r["category"], r["seed"]) for r in cells})
    print(f"{len(files)} cells, {n_bad_cells} unreadable, {n_runs} runs with a typed-query baseline", file=sys.stderr)

    curve_acc: dict[tuple, dict[str, list]] = {}
    for r in cells:
        d = differences(r, args.horizon)
        for design, v in d.items():
            for span, n in SPANS.items():
                r[f"d_{design}_{span}"] = float(v[:n].sum() / n)
        key = (r["corpus"], r["beta"])
        acc = curve_acc.setdefault(
            key, {k: [] for k in ("query_quota", "query_open", "on_quota", "before", "cen", "text")}
        )
        acc["query_quota"].append(d["query_quota"][: args.horizon])
        acc["query_open"].append(d["query_open"][: args.horizon])
        acc["on_quota"].append(r["_on_quota"].astype(float))
        acc["before"].append(r["_before"].astype(float))
        acc["cen"].append(np.where(r["_on_quota"], r["_cen"], np.nan))
        acc["text"].append(np.where(r["_on_quota"] | r["_before"], r["text_fbeta"], np.nan))
    frame = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in cells])
    frame.to_csv(args.out / "cells.csv", index=False)
    votes = pd.DataFrame(
        [
            {
                "corpus": r["corpus"],
                "beta": r["beta"],
                "category": r["category"],
                "seed": r["seed"],
                "t": t + 1,
                "n_good": int(r["_n_good"][t]),
                "n_bad": int(r["_n_bad"][t]),
                "centroid_fbeta": float(r["_cen"][t]),
                "query_fbeta": r["text_fbeta"],
            }
            for r in cells
            for t in np.nonzero(r["_on_quota"])[0]
        ]
    )
    votes.to_csv(args.out / "votes.csv", index=False)

    rows = []
    for (corpus, beta), acc in curve_acc.items():
        stack = {k: np.vstack(v) for k, v in acc.items()}
        with np.errstate(invalid="ignore"):
            cen_mean = np.nanmean(stack["cen"], axis=0)
            text_mean = np.nanmean(stack["text"], axis=0)
        for t in range(args.horizon):
            rows.append(
                {
                    "corpus": corpus,
                    "beta": beta,
                    "t": t + 1,
                    "d_query_quota": float(stack["query_quota"][:, t].mean()),
                    "d_query_open": float(stack["query_open"][:, t].mean()),
                    "share_before_first_good": float(stack["before"][:, t].mean()),
                    "share_under_quota": float(stack["on_quota"][:, t].mean()),
                    "centroid_fbeta_under_quota": float(cen_mean[t]),
                    "query_fbeta_before_quota": float(text_mean[t]),
                }
            )
    pd.DataFrame(rows).to_csv(args.out / "curves.csv", index=False)

    summ = []
    for (corpus, beta), h in frame.groupby(["corpus", "beta"]):
        cl = h["category"]
        rec = {
            "corpus": corpus,
            "beta": beta,
            "runs": len(h),
            "no_good": int(h["no_good"].sum()),
            "tier_past_h": int(h["tier_past_h"].sum()),
            "votes_before_first_good_mean": h["n_before"].mean(),
            "votes_under_quota_mean": h["n_quota"].mean(),
            "runs_with_quota_votes": int((h["n_quota"] > 0).sum()),
            "text_fbeta": h["text_fbeta"].mean(),
            "centroid_fbeta_on_quota": h["centroid_fbeta_on_quota"].mean(),
            "text_fbeta_same_runs": h.loc[h["n_quota"] > 0, "text_fbeta"].mean(),
        }
        q = h[h["n_quota"] > 0]
        m, se, _ = _mean_se(q["text_fbeta"] - q["centroid_fbeta_on_quota"], q["category"])
        rec["query_minus_centroid_on_quota"], rec["query_minus_centroid_on_quota_se"] = m, se
        for design in DESIGNS:
            for span in SPANS:
                m, se, n = _mean_se(h[f"d_{design}_{span}"], cl)
                rec[f"d_{design}_{span}"], rec[f"d_{design}_{span}_se"] = m, se
        rec["clusters"] = n
        summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(args.out / "summary.csv", index=False)

    lines = [
        f"# #4758: Find under the quota in a typed-query opening (H = {args.horizon} votes)",
        "",
        "Differences are the typed query's design minus today's (the Goods' centroid), paired per run, mean over the "
        "votes of each span; SE clusters on the category. `query_quota` gives the typed query's set from the first "
        "Good to the quota; `query_open` also before the first Good, where today's Find gives nothing.",
        "",
        "| corpus | beta | runs | votes before 1st Good | votes under quota | typed query F | centroid F on its votes "
        "| query - centroid there | query_quota 1-10 | 1-50 | 1-150 | query_open 1-10 | 1-50 | 1-150 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summ:
        cells_txt = " | ".join(f"{r[f'd_{d}_{s}']:+.4f} ± {r[f'd_{d}_{s}_se']:.4f}" for d in DESIGNS for s in SPANS)
        lines.append(
            f"| {r['corpus']} | {r['beta']:g} | {r['runs']} | {r['votes_before_first_good_mean']:.2f} "
            f"| {r['votes_under_quota_mean']:.2f} | {r['text_fbeta_same_runs']:.3f} | {r['centroid_fbeta_on_quota']:.3f} "
            f"| {r['query_minus_centroid_on_quota']:+.3f} ± {r['query_minus_centroid_on_quota_se']:.3f} | {cells_txt} |"
        )
    one = frame.drop_duplicates(["category", "seed"])
    lines += [
        "",
        f"- {len(one)} runs; {int(one['no_good'].sum())} found no Good by vote {args.horizon}; "
        f"{int(one['tier_past_h'].sum())} still under the quota at vote {args.horizon} (their 1-150 means carry the "
        f"state at {args.horizon}); {n_bad_cells} cells unreadable or without the rule's rows.",
        "- 'typed query F' and 'centroid F on its votes' are over the runs with at least one vote under the quota.",
    ]
    lines += ["", *breakdown(votes, n_runs)]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
