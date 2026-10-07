"""#4604: price showing a detector during Autopilot's opening, offline, from a State of the App run.

The harness trains a detector at every step from the first Good+Bad, but the app shows one only from the Hard
phase (``app_trained``, #4605). Before that the user sees the typed query's set at its guarded line. Every rule
here picks a hand-off click h per run: the app shows the typed query's set before h and the detector's line from
h (the detector the harness trained at that click, carried over clicks with no row). Today's h is the first
``app_trained`` step. A rule only moves h EARLIER than that unless named ``exact``.

Scored on each run's withheld half: F-beta of the set above the line at the run's own beta (``thr_fbeta``), the
typed query's from the text baseline (``_text_app_line``). Trained runs only, as ``by_click.py``.

    python handoff_price_4604.py --steps steps_binary_b1.csv.gz --picks picks_binary_b1.csv.gz \\
        --cells <analysis-binary-4605>/cells.csv --baseline <text_baseline.csv> --beta 1 --tag binary_b1 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

T = 150
GRID = np.arange(T + 1)
RUN = ["category", "seed"]


def fbeta(p: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """F-beta, with an empty set (precision NaN, recall 0) scoring 0 as ``analyze._fbeta_pr``."""
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    f = np.where(np.isfinite(p) & np.isfinite(r), f, np.nan)
    return np.where(~np.isfinite(p) & (r == 0), 0.0, f)


def counts_fbeta(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (1 + b2) * tp / den, 0.0)


def class_half(cls: str) -> int:
    return int(hashlib.md5(cls.encode()).hexdigest(), 16) % 2


def load(a: argparse.Namespace):
    steps = pd.read_csv(a.steps)
    picks = pd.read_csv(a.picks)
    if a.cells:
        cells = pd.read_csv(a.cells)
        cells = cells[~cells["never_trained"].astype(bool)][["category", "class", "band", "seed", "shown_from"]]
    else:
        # A run with no State of the App analysis (#4583's arms): a trained run is one with steps, and its class
        # and band come off the category; shown_from is recomputed below either way.
        cells = steps[RUN].drop_duplicates().copy()
        cells["class"] = cells["category"].str.split("@").str[0]
        cells["band"] = cells["category"].str.split("@").str[1]
        on = steps.loc[steps["app_trained"] == 1].groupby(RUN)["t"].min().rename("shown_from")
        cells = cells.merge(on, left_on=RUN, right_index=True, how="left").fillna({"shown_from": np.inf})
    base = pd.read_csv(a.baseline)
    base = base[base["embedder"] == a.text_embedder]
    runs = cells.sort_values(RUN).reset_index(drop=True)
    idx = {(c, int(s)): i for i, (c, s) in enumerate(zip(runs["category"], runs["seed"]))}
    n = len(runs)

    # The typed query's set: the guarded line, scored on the withheld half (text_baseline.py).
    tq = np.full(n, np.nan)
    cut = np.full(n, np.nan)
    for r in base.to_dict("records"):
        i = idx.get((r["category"], int(r["seed"])))
        if i is not None:
            tq[i] = fbeta(np.array([r["text_precision"]]), np.array([r["text_recall"]]), a.beta)[0]
            cut[i] = r["text_gmm_cut"]

    # The detector the harness trained at every step, shown or not; carried forward over clicks with no row.
    det = np.full((n, T + 1), np.nan)
    thr = np.full((n, T + 1), np.nan)
    shown = np.full(n, np.inf)
    more = np.full(n, np.inf)  # the first step past the opening's Good and Bad phases (3 Goods + 4 Bads)
    kept = np.full((n, T + 1), np.nan)
    steps = steps[steps["t"] <= T]
    steps = steps.assign(f=fbeta(steps["precision"].to_numpy(), steps["recall"].to_numpy(), a.beta))
    for (c, s), g in steps.groupby(RUN):
        i = idx.get((c, int(s)))
        if i is None:
            continue
        g = g.drop_duplicates("t", keep="last").sort_values("t")
        tt = g["t"].to_numpy(dtype=int)
        det[i, tt] = g["f"].to_numpy()
        thr[i, tt] = g["threshold"].to_numpy()
        kept[i, tt] = g["floor_count"].to_numpy()
        on = g.loc[g["app_trained"] == 1, "t"]
        shown[i] = float(on.min()) if len(on) else np.inf
        past = g.loc[~g["phase"].astype(str).isin(["good", "bad"]), "t"]
        more[i] = float(past.min()) if len(past) else np.inf
    det = pd.DataFrame(det).ffill(axis=1).to_numpy()
    thr = pd.DataFrame(thr).ffill(axis=1).to_numpy()  # the line in force at each click
    kept = pd.DataFrame(kept).ffill(axis=1).to_numpy()
    sf = runs["shown_from"].to_numpy(dtype=float)
    mism = np.sum(~((sf == shown) | (~np.isfinite(sf) & ~np.isfinite(shown))))
    print(f"runs {n}; shown_from mismatches vs cells.csv: {mism}; no typed query: {np.isnan(tq).sum()}")

    # Votes: label, typed-query side and the detector's honest side of each pick. A pick at click s was scored
    # by the detector trained at step s-1, so it is judged against that step's line (progressive validation).
    lab = np.full((n, T + 1), np.nan)
    tside = np.full((n, T + 1), np.nan)
    dside = np.full((n, T + 1), np.nan)
    picks = picks[picks["t"] <= T]
    for (c, s), g in picks.groupby(RUN):
        i = idx.get((c, int(s)))
        if i is None:
            continue
        g = g.drop_duplicates("t", keep="last")
        tt = g["t"].to_numpy(dtype=int)
        lab[i, tt] = g["picked_label"].to_numpy(dtype=float)
        tside[i, tt] = (g["picked_seed_score"].to_numpy(dtype=float) >= cut[i]).astype(float)
        ds = g["picked_detector_score"].to_numpy(dtype=float)
        th = thr[i, np.maximum(tt - 1, 0)]
        dside[i, tt] = np.where(np.isfinite(ds) & np.isfinite(th), (ds >= th).astype(float), np.nan)
    runs["more_from"] = more
    return runs, tq, det, shown, kept, lab, tside, dside


def curve_for(h: np.ndarray, tq: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Per run and click: the detector's F from the hand-off click *h* on (where one exists), else the typed query's."""
    use = (GRID[None, :] >= h[:, None]) & np.isfinite(det)
    return np.where(use, det, tq[:, None])


def first_true(cond: np.ndarray) -> np.ndarray:
    """Per row, the first click where *cond* holds; inf where it never does."""
    any_ = cond.any(axis=1)
    return np.where(any_, cond.argmax(axis=1).astype(float), np.inf)


def rules(runs, tq, det, shown, kept, lab, tside, dside, beta):
    """``{name: (family, param, per-run per-click F)}``."""
    out = {}
    out["today"] = ("today", np.nan, curve_for(shown, tq, det))
    out["detector_always"] = ("always", np.nan, curve_for(np.zeros(len(tq)), tq, det))
    in_opening = GRID[None, :] < shown[:, None]
    ceil_open = np.where(in_opening & np.isfinite(det), np.maximum(det, tq[:, None]), curve_for(shown, tq, det))
    out["ceiling_opening"] = ("ceiling", np.nan, ceil_open)
    out["ceiling_any"] = ("ceiling", np.nan, np.where(np.isfinite(det), np.maximum(det, tq[:, None]), tq[:, None]))
    for n in range(0, 151, 1):
        out[f"click_{n}"] = ("click", n, curve_for(np.minimum(float(n), shown), tq, det))
    for n in (10, 20, 30, 40, 50, 60, 80, 100):
        out[f"exact_{n}"] = ("exact", n, curve_for(np.full(len(tq), float(n)), tq, det))
    goods = np.nan_to_num(lab).cumsum(axis=1)
    for g in range(1, 21):
        out[f"goods_{g}"] = ("goods", g, curve_for(np.minimum(first_true(goods >= g), shown), tq, det))
    # Votes so far, the picks the detector scored before their vote (honest), both sets judged on the same picks.
    have = np.isfinite(dside)
    y = np.where(have, lab, 0.0)
    d = np.where(have, dside, 0.0)
    s = np.where(have, tside, 0.0)
    tp_d, fp_d, fn_d = (y * d).cumsum(1), ((1 - y) * d * have).cumsum(1), (y * (1 - d)).cumsum(1)
    tp_t, fp_t, fn_t = (y * s).cumsum(1), ((1 - y) * s * have).cumsum(1), (y * (1 - s)).cumsum(1)
    npos = y.cumsum(1)
    fd = counts_fbeta(tp_d, fp_d, fn_d, beta)
    ft = counts_fbeta(tp_t, fp_t, fn_t, beta)
    has_det = np.isfinite(det)
    more = runs["more_from"].to_numpy(dtype=float)
    out["after_bad"] = ("after_bad", np.nan, curve_for(np.minimum(more, shown), tq, det))
    for margin in (-0.2, -0.1, 0.0, 0.1, 0.2):
        for mp in (1, 3):
            cond = (fd >= ft + margin) & (npos >= mp) & has_det & (GRID[None, :] >= more[:, None])
            out[f"after_bad_votes_m{margin:+.1f}_p{mp}"] = (
                "after_bad_votes",
                (margin, mp),
                curve_for(np.minimum(first_true(cond), shown), tq, det),
            )
    for margin in (-0.2, -0.1, 0.0, 0.1, 0.2):
        for mp in (1, 3, 5, 8):
            cond = (fd >= ft + margin) & (npos >= mp) & has_det
            out[f"votes_m{margin:+.1f}_p{mp}"] = (
                "votes",
                (margin, mp),
                curve_for(np.minimum(first_true(cond), shown), tq, det),
            )
    return out


def summarize(name, fam, param, F, today, runs) -> dict:
    trained = np.isfinite(F).all(axis=1)
    m = F[trained].mean(axis=0)
    gain = (F[trained, 1:] - today[trained, 1:]).mean(axis=1)
    gain50 = (F[trained, 1:51] - today[trained, 1:51]).mean(axis=1)
    cls = runs["class"].to_numpy()[trained]
    by_cls = pd.Series(gain).groupby(cls).mean()
    row = {
        "rule": name, "family": fam, "param": str(param), "runs": int(trained.sum()),
        "auc150": m[1:].mean(), "auc50": m[1:51].mean(),
        **{f"at_{t}": m[t] for t in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150)},
        "gain150": gain.mean(), "gain150_se": by_cls.std(ddof=1) / np.sqrt(len(by_cls)),
        "gain50": gain50.mean(), "gain50_se": pd.Series(gain50).groupby(cls).mean().std(ddof=1) / np.sqrt(len(by_cls)),
        "worse_runs": float(np.mean(gain < -1e-9)), "better_runs": float(np.mean(gain > 1e-9)),
    }  # fmt: skip
    for half in (0, 1):
        sel = np.array([class_half(c) == half for c in cls])
        row[f"gain150_h{half}"] = gain[sel].mean()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--cells", help="a State of the App analysis's cells.csv; without it, derived from the steps")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--text-embedder", default="siglip")
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, tq, det, shown, kept, lab, tside, dside = load(a)
    rs = rules(runs, tq, det, shown, kept, lab, tside, dside, a.beta)
    today = rs["today"][2]
    rows = [summarize(k, fam, p, F, today, runs) for k, (fam, p, F) in rs.items()]
    summ = pd.DataFrame(rows).assign(tag=a.tag, beta=a.beta)
    summ.to_csv(f"{a.out}/summary_{a.tag}.csv", index=False)
    # Per-click curves of the headline rules, and per-run hand-off clicks for the record.
    keep = ["today", "detector_always", "ceiling_opening", "ceiling_any", "after_bad"] + [
        k for k in rs if k.startswith(("click_", "goods_", "votes_", "exact_", "after_bad_votes"))
    ]
    cur = pd.DataFrame({k: rs[k][2][np.isfinite(rs[k][2]).all(axis=1)].mean(axis=0) for k in keep})
    cur.insert(0, "t", GRID)
    cur.to_csv(f"{a.out}/curves_{a.tag}.csv", index=False)
    # Per-run F at each click for today / always / ceiling, for split-half and per-band reads.
    np.savez_compressed(
        f"{a.out}/runs_{a.tag}.npz",
        category=runs["category"].to_numpy(), cls=runs["class"].to_numpy(), band=runs["band"].to_numpy(),
        seed=runs["seed"].to_numpy(), shown=shown, more=runs["more_from"].to_numpy(dtype=float), tq=tq, det=det, kept=kept, lab=lab, tside=tside, dside=dside,
    )  # fmt: skip
    top = summ.sort_values("gain150", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            summ[summ["family"].isin(["today", "always", "ceiling"])][
                ["rule", "runs", "auc150", "auc50", "at_10", "at_25", "at_50", "at_100", "gain150", "gain150_se"]
            ]
            .round(3)
            .to_string(index=False)
        )
        print(
            summ[summ["family"].isin(["after_bad"])][
                ["rule", "auc150", "auc50", "at_10", "at_25", "at_50", "gain150", "gain150_se", "gain50"]
            ]
            .round(3)
            .to_string(index=False)
        )
        for fam in ("click", "goods", "votes", "after_bad_votes", "exact"):
            t = top[top["family"] == fam].head(5)
            print(
                t[
                    [
                        "rule",
                        "auc150",
                        "auc50",
                        "at_10",
                        "at_25",
                        "at_50",
                        "gain150",
                        "gain150_se",
                        "gain150_h0",
                        "gain150_h1",
                        "worse_runs",
                    ]
                ]
                .round(3)
                .to_string(index=False)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
