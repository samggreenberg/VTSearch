"""#5040: the labels line's chances, closed loop, as curves over the click a user has reached.

Reads ``chances_5040.sh``'s arms (Binary Photo, beta 1/4, 1 and 4, 200 clicks plus the closing check, paired by cell
and seed), which differ only in the chances the labels line is cut on:

* ``ctl``: the app, the labels' Bad component and the counted bound at 2x;
* ``bssl``: the Bad component fitted on the corpus, the held-out Bads counted as labelled points of it;
* ``bssl4``: the same, with the counted bound at 4x;
* ``bmean4``: the Bad component's mean alone fitted on the corpus, the bound at 4x (from a second frozen tree whose
  default path is the first's).

An arm's run is identical to the control's until its labels line first differs. Autopilot reads the line's chances
from the first retrain (its hard picks sample where they put the line), while the screen still shows the typed
query's set, so the arms part at about vote 8. The **pairing check** is that the first pick the two disagree on is
one where the acquisition's threshold moved, or the flow's phase did (a sooner Done), and that the F shown agrees up
to the first click that shows the detector (the arm moves the shown line itself, so after that F may part before
any pick does).

Per run and click it scores what the app shows on the withheld half, at the run's own beta, as
``early_check_4664.py`` does: the typed query's set until the first click with a detector on screen, then the
detector's line, carried over clicks with no row. "After the closing check" is the run's last ``check`` row.
**Done** is the vote count when the flow reported it (the vote before its first ``done`` pick); a user who stops
there stops at that click, or at 200 if it never comes.

    python chances_5040_ab.py --root <state-of-the-app dir> --date 2026-10-17 --seeds 5 \\
        --baseline <text_baseline.csv> --out <dir> [--docs <report dir>/tables] [--tag smoke] [--categories <file>]

``--categories`` reads only the cells a file names, one ``class@band`` a line: #4817 re-reads the contrast on
``o365_better`` over the 107 cells that pair with a built ``coco_better`` cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from early_check_4664 import INK, MUTED, STEP_COLS, _axes_style, _headline, area, fbeta, se_over_classes  # noqa: E402

T = 200
GRID_T = np.arange(T + 1)
CONTROL = "ctl"
ARMS = ("ctl", "bssl", "bssl4", "bmean4")
BETAS = (0.25, 1.0, 4.0)
NAME = {
    "ctl": "the app (labels' Bad component, bound 2x)",
    "bssl": "Bad component fitted on the corpus",
    "bssl4": "Bad component fitted on the corpus, bound 4x",
    "bmean4": "Bad component's mean fitted on the corpus, bound 4x",
}
COLOR = {"ctl": "#8a8985", "bssl": "#2a78d6", "bssl4": "#eb6834", "bmean4": "#1baf7a"}
CLICKS = (10, 25, 35, 50, 75, 100, 125, 150, 175, 200)
AREAS = ((0, 50), (0, 100), (0, 150), (0, 200))


PICK_COLS = ["category", "seed", "embedder", "t", "phase", "check_kind", "picked_id", "picked_label", "acq_threshold"]


def read_cell(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    head = pd.read_csv(path, nrows=0).columns
    m = pd.read_csv(path, usecols=[c for c in STEP_COLS if c in head], low_memory=False)
    m = _headline(m[m["embedder"] == "siglip"])
    stem = path[: -len(".csv.gz")] if path.endswith(".gz") else path[: -len(".csv")]
    ext = ".csv.gz" if path.endswith(".gz") else ".csv"
    try:
        pk = pd.read_csv(f"{stem}__picks{ext}", usecols=lambda c: c in PICK_COLS, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pk = pd.DataFrame(columns=PICK_COLS)
    return m, pk[pk["embedder"] == "siglip"]


def load_arm(d: Path, seeds: int, procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from multiprocessing import Pool

    cells = sorted(str(f) for f in d.glob("task_*.csv*") if "__" not in f.name)
    if not cells:
        return pd.DataFrame(columns=STEP_COLS), pd.DataFrame(columns=PICK_COLS)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, cells, chunksize=8)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    return steps[steps["seed"] < seeds], picks[picks["seed"] < seeds]


def btag(beta: float) -> str:
    return {0.25: "025", 1.0: "1", 4.0: "4"}[float(beta)]


def run_dir(root: Path, date: str, tag: str, arm: str, beta: float) -> Path:
    return root / f"{date}-chances5040{tag}-{arm}-b{btag(beta)}" / "results" / "cells"


def per_run(steps: pd.DataFrame, picks: pd.DataFrame, tq: dict, beta: float) -> dict[str, dict]:
    """Per run: the shown set's F at every click, the after-check F, the pick sequence, Done and the check clicks."""
    out: dict[str, dict] = {}
    for (c, s), g in steps.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        q = tq.get(key, np.nan)
        ordinary = g[g["phase"].astype(str) != "check"].drop_duplicates("t", keep="last").sort_values("t")
        ordinary = ordinary[ordinary["t"] <= T]
        det = np.full(T + 1, np.nan)
        det[ordinary["t"].to_numpy(dtype=int)] = fbeta(ordinary["precision"], ordinary["recall"], beta)
        det = pd.Series(det).ffill().to_numpy()
        on = ordinary.loc[ordinary["app_trained"] == 1, "t"]
        shown = float(on.min()) if len(on) else np.inf
        f = np.where((GRID_T >= shown) & np.isfinite(det), det, q)
        chk = g[g["phase"].astype(str) == "check"].sort_values("t")
        after = float(fbeta(chk["precision"].iloc[-1:], chk["recall"].iloc[-1:], beta)[0]) if len(chk) else f[T]
        ap = np.full(T + 1, np.nan)
        ap[ordinary["t"].to_numpy(dtype=int)] = ordinary["average_precision"].to_numpy(dtype=float)
        out[key] = {"f": f, "after": after, "tq": q, "trained_at": shown, "ap200": pd.Series(ap).ffill().to_numpy()[T]}
    for (c, s), g in picks.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        if key not in out:
            continue
        g = g.reset_index(drop=True)
        ph = g["phase"].astype(str).to_numpy()
        kind = g["check_kind"].fillna("").astype(str).to_numpy()
        lab = g["picked_label"].to_numpy(dtype=float)
        tt = g["t"].to_numpy(dtype=int)
        pid = g["picked_id"].to_numpy(dtype=np.int64)
        acq = g["acq_threshold"].to_numpy(dtype=float) if "acq_threshold" in g else np.full(len(g), np.nan)
        goods = np.zeros(T + 1)
        click = ph != "check"
        np.add.at(goods, np.clip(tt[click], 0, T), lab[click])
        done_t = tt[ph == "done"]
        done_at = float(done_t.min() - 1) if len(done_t) else np.nan
        stop = int(done_at) if np.isfinite(done_at) and done_at <= T else T
        out[key].update(
            seq=(tt, ph, kind, pid, acq),
            goods=np.cumsum(goods),
            check_clicks=int((ph == "prompt").sum()),
            done_at=done_at,
            stop_clicks=float(stop),
            stop_f=float(out[key]["f"][stop]),
        )
    return out


def departure(a: tuple, b: tuple) -> tuple[int, float, str]:
    """Where two pick sequences first disagree: (row, the votes cast before it, why); row -1 when they never do.

    *why* is ``line`` when the acquisition's threshold - which reads the labels line's chances - differs there or
    one pick before, ``phase`` when the flow is in another phase (a sooner or later Done, a check prompted on the
    line's own reading), else ``other``: a departure the chances cannot explain, which the pairing check counts.
    """
    ta, pa, ka, ia, qa = a
    tb, pb, kb, ib, qb = b
    n = min(len(ta), len(tb))
    same = (ta[:n] == tb[:n]) & (pa[:n] == pb[:n]) & (ka[:n] == kb[:n]) & (ia[:n] == ib[:n])
    bad = np.flatnonzero(~same)
    if not len(bad):
        return (-1, float(T), "none") if len(ta) == len(tb) else (n, float(ta[n - 1]) if n else 0.0, "length")
    i = int(bad[0])

    def moved(j: int) -> bool:
        return j >= 0 and not (qa[j] == qb[j] or (np.isnan(qa[j]) and np.isnan(qb[j])))

    why = "line" if moved(i) or moved(i - 1) else "phase" if pa[i] != pb[i] or ka[i] != kb[i] else "other"
    return i, float(ta[i - 1]) if i > 0 else 0.0, why


def analyse_beta(runs: dict[str, dict], beta: float) -> dict:
    """Every arm in *runs* (the control first) against the control, on the runs all of them have."""
    arms = tuple(a for a in ARMS if a in runs)
    others = arms[1:]
    keys = sorted(set.intersection(*(set(runs[a]) for a in arms)))
    keys = [k for k in keys if np.isfinite(runs[CONTROL][k]["tq"]) and all("seq" in runs[a][k] for a in arms)]
    cls = np.array([k.split("|")[0].split("@")[0] for k in keys])
    band = np.array([k.split("|")[0].split("@")[1] for k in keys])
    F = {arm: np.vstack([runs[arm][k]["f"] for k in keys]) for arm in arms}
    G = {arm: np.vstack([runs[arm][k]["goods"] for k in keys]) for arm in arms}
    after = {arm: np.array([runs[arm][k]["after"] for k in keys]) for arm in arms}

    def col(arm: str, name: str) -> np.ndarray:
        return np.array([runs[arm][k].get(name, np.nan) for k in keys], dtype=float)

    # Pairing: an arm may leave the control's picks only where its line does (the acquisition reads the chances).
    dep = {}
    for arm in others:
        rows = [departure(runs[arm][k]["seq"], runs[CONTROL][k]["seq"]) for k in keys]
        at = np.array([r[1] for r in rows])
        moved = np.array([r[0] >= 0 for r in rows])
        why = np.array([r[2] for r in rows])
        # The arm moves the shown line itself, so F may part before any pick does; until the detector is first
        # shown (the typed query's set, the same in both) and until the picks part, it must not.
        upto = np.minimum(at, col(CONTROL, "trained_at") - 1)
        f_ok = np.array([np.allclose(F[arm][i, : int(upto[i]) + 1], F[CONTROL][i, : int(upto[i]) + 1], equal_nan=True)
                         for i in range(len(keys))])  # fmt: skip
        dep[arm] = {"moved": moved, "at": np.where(moved, at, np.nan), "why": why,
                    "f_mismatch": int((~f_ok).sum())}  # fmt: skip

    rows = []
    for arm in arms:
        r: dict = {"beta": beta, "arm": arm, "runs": len(keys)}
        if arm != CONTROL:
            d = dep[arm]
            r.update(departed=float(d["moved"].mean()), departure_median=float(np.nanmedian(d["at"]))
                     if d["moved"].any() else np.nan, departed_on_the_line=int((d["why"] == "line").sum()),
                     departed_on_a_phase=int((d["why"] == "phase").sum()),
                     departed_otherwise=int(np.isin(d["why"], ("other", "length")).sum()),
                     f_prefix_mismatch=d["f_mismatch"])  # fmt: skip
        for lo, hi in AREAS:
            a_ = area(F[arm], lo, hi)
            r[f"meanF_{lo}_{hi}"] = float(np.nanmean(a_))
            g = a_ - area(F[CONTROL], lo, hi)
            r[f"meanF_gain_{lo}_{hi}"], r[f"meanF_se_{lo}_{hi}"] = float(np.nanmean(g)), se_over_classes(g, cls)
            r[f"better_{lo}_{hi}"] = float(np.mean(g > 1e-9))
            r[f"worse_{lo}_{hi}"] = float(np.mean(g < -1e-9))
        for c in CLICKS:
            g = F[arm][:, c] - F[CONTROL][:, c]
            r[f"F_at_{c}"] = float(np.nanmean(F[arm][:, c]))
            r[f"gain_at_{c}"], r[f"se_at_{c}"] = float(np.nanmean(g)), se_over_classes(g, cls)
        g = after[arm] - after[CONTROL]
        r["after"] = float(np.nanmean(after[arm]))
        r["after_gain"], r["after_se"] = float(np.nanmean(g)), se_over_classes(g, cls)
        r["check_clicks"] = float(np.nanmean(col(arm, "check_clicks")))
        done = col(arm, "done_at")
        r["done_reached"] = float(np.mean(np.isfinite(done) & (done <= T)))
        r["done_median"] = float(np.nanmedian(np.where(done <= T, done, np.nan)))
        both = np.isfinite(done) & np.isfinite(col(CONTROL, "done_at")) & (done <= T) & (col(CONTROL, "done_at") <= T)
        dd = done - col(CONTROL, "done_at")
        r["done_shift_mean"] = float(np.nanmean(dd[both])) if both.any() else np.nan
        r["done_shift_se"] = se_over_classes(dd[both], cls[both]) if both.sum() > 2 else np.nan
        r["stop_clicks"] = float(np.nanmean(col(arm, "stop_clicks")))
        r["stop_f"] = float(np.nanmean(col(arm, "stop_f")))
        g = col(arm, "stop_f") - col(CONTROL, "stop_f")
        r["stop_f_gain"], r["stop_f_se"] = float(np.nanmean(g)), se_over_classes(g, cls)
        g = col(arm, "stop_clicks") - col(CONTROL, "stop_clicks")
        r["stop_clicks_gain"], r["stop_clicks_se"] = float(np.nanmean(g)), se_over_classes(g, cls)
        for c in (50, 100, 150, 200):
            r[f"goods_at_{c}"] = float(np.nanmean(G[arm][:, c]))
        r["ap200"] = float(np.nanmean(col(arm, "ap200")))
        r["ap200_gain"] = float(np.nanmean(col(arm, "ap200") - col(CONTROL, "ap200")))
        rows.append(r)
    summary = pd.DataFrame(rows)

    # By band, and by how the session ends (the mean of both arms' F at 200: binning on one arm's own outcome
    # would regress the pair's difference toward the mean of the bin, #4909).
    brow = []
    for arm in others:
        g200 = area(F[arm], 0, T) - area(F[CONTROL], 0, T)
        g50 = area(F[arm], 0, 50) - area(F[CONTROL], 0, 50)
        end = (F[arm][:, T] + F[CONTROL][:, T]) / 2
        outcome = {"below 0.2": end < 0.2, "0.2 to 0.5": (end >= 0.2) & (end < 0.5), "0.5 or more": end >= 0.5}
        for split, mask_of in (
            ("band", {bd: band == bd for bd in sorted(set(band))}),
            ("F at 200, both arms", outcome),
        ):
            for lvl, m in mask_of.items():
                if m.sum() < 2:
                    continue
                brow.append({"beta": beta, "arm": arm, "split": split, "level": lvl, "runs": int(m.sum()),
                             "meanF_gain_0_50": float(g50[m].mean()), "se_0_50": se_over_classes(g50[m], cls[m]),
                             "meanF_gain_0_200": float(g200[m].mean()), "se_0_200": se_over_classes(g200[m], cls[m]),
                             "control_F50": float(np.nanmean(F[CONTROL][m, 50])), "arm_F50": float(np.nanmean(F[arm][m, 50])),
                             "control_F200": float(np.nanmean(F[CONTROL][m, T])), "arm_F200": float(np.nanmean(F[arm][m, T])),
                             "better": float(np.mean(g200[m] > 1e-9)), "worse": float(np.mean(g200[m] < -1e-9))})  # fmt: skip
    splits = pd.DataFrame(brow)

    # Literal examples: per arm, the sessions it helped and hurt most over clicks 0-200.
    erow = []
    for arm in others:
        g = area(F[arm], 0, T) - area(F[CONTROL], 0, T)
        moved = dep[arm]["moved"]
        order = np.argsort(np.where(moved, g, np.nan))
        order = order[np.isfinite(g[order]) & moved[order]]
        for tag, idx in (("helped most", order[::-1][:3]), ("hurt most", order[:3])):
            for i in idx:
                cat, seed = keys[i].split("|")
                erow.append({
                    "beta": beta, "arm": arm, "which": tag, "category": cat, "seed": int(seed),
                    "meanF_gain_0_200": g[i], "departs_at": dep[arm]["at"][i],
                    "F_control_at_50": F[CONTROL][i, 50], "F_arm_at_50": F[arm][i, 50],
                    "F_control_at_100": F[CONTROL][i, 100], "F_arm_at_100": F[arm][i, 100],
                    "F_control_at_200": F[CONTROL][i, T], "F_arm_at_200": F[arm][i, T],
                    "done_control": col(CONTROL, "done_at")[i], "done_arm": col(arm, "done_at")[i],
                    "goods_200_control": G[CONTROL][i, T], "goods_200_arm": G[arm][i, T],
                })  # fmt: skip
    examples = pd.DataFrame(erow)
    curves = pd.DataFrame({"beta": beta, "t": GRID_T, **{arm: np.nanmean(F[arm], axis=0) for arm in arms}})
    gains = pd.DataFrame({"beta": beta, "t": GRID_T})
    for arm in others:
        d = F[arm] - F[CONTROL]
        gains[arm] = np.nanmean(d, axis=0)
        by = pd.DataFrame(d).groupby(cls).mean()
        gains[f"{arm}_se"] = by.std(ddof=1).to_numpy() / np.sqrt(len(by))
    return {"arms": arms, "summary": summary, "splits": splits, "curves": curves, "gains": gains, "after": after,
            "examples": examples}  # fmt: skip


def figure(res: dict[float, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, len(res), figsize=(6.5 * len(res), 8.2), squeeze=False)
    for j, (beta, r) in enumerate(sorted(res.items())):
        ax = axes[0, j]
        for arm in r["arms"]:
            ax.plot(GRID_T, r["curves"][arm], color=COLOR[arm], lw=2 if arm == CONTROL else 1.6, label=NAME[arm])
            ax.plot([T + 4], [np.nanmean(r["after"][arm])], "o", ms=6, mfc="none", mec=COLOR[arm], mew=1.4)
        ax.set_title(f"F-beta of the returned set over clicks, beta {beta:g}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("clicks (circles: after the closing check)", color=MUTED, fontsize=9)
        ax.set_ylabel(f"F-beta (beta {beta:g}) on the withheld half", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1, j]
        for arm in r["arms"][1:]:
            m, se = r["gains"][arm].to_numpy(), r["gains"][f"{arm}_se"].to_numpy()
            ax.fill_between(GRID_T, m - se, m + se, color=COLOR[arm], alpha=0.18, lw=0)
            ax.plot(GRID_T, m, color=COLOR[arm], lw=2, label=NAME[arm])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(
            f"Paired gain over the app, beta {beta:g} (± 1 SE over classes)", color=INK, fontsize=11, loc="left"
        )
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-17")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--betas", default="0.25,1,4")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path, help="the report's tables dir; figures go beside it")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--arms", default=",".join(ARMS), help="the arms to read; the control is always one")
    ap.add_argument("--categories", type=Path, help="read only these cells, one class@band a line")
    a = ap.parse_args()
    if CONTROL not in a.arms.split(","):
        a.arms = f"{CONTROL},{a.arms}"
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == "siglip") & (base["seed"] < a.seeds)]
    keep = set(a.categories.read_text().split("\n")) - {""} if a.categories else None
    res: dict[float, dict] = {}
    counts = {}
    for beta in (float(b) for b in a.betas.split(",")):
        t = btag(beta)
        tq = {f"{c}|{int(s)}": float(fbeta([p], [r], beta)[0]) for c, s, p, r in
              zip(base["category"], base["seed"], base[f"text_line_precision_b{t}"], base[f"text_line_recall_b{t}"])}  # fmt: skip
        runs = {}
        for arm in [x for x in ARMS if x in a.arms.split(",")]:
            steps, picks = load_arm(run_dir(a.root, a.date, a.tag, arm, beta), a.seeds, a.procs)
            if keep is not None:
                steps, picks = steps[steps["category"].isin(keep)], picks[picks["category"].isin(keep)]
            runs[arm] = per_run(steps, picks, tq, beta)
            counts[f"{arm}-b{t}"] = len(runs[arm])
            print(f"beta {beta:g} {arm}: {len(runs[arm])} runs", flush=True)
        if min(len(runs[arm]) for arm in runs) == 0:
            print(f"beta {beta:g}: an arm has no runs yet; skipped")
            continue
        res[beta] = analyse_beta(runs, beta)
    if not res:
        return 1
    outs = [a.out] + ([a.docs] if a.docs else [])
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        for name in ("summary", "splits", "examples", "curves", "gains"):
            pd.concat([r[name] for r in res.values()]).to_csv(o / f"ab_{name}.csv", index=False)
        (o / "ab_counts.json").write_text(json.dumps(counts, indent=1))
    figs = [a.out] + ([a.docs.parent / "figures"] if a.docs else [])
    for o in figs:
        o.mkdir(parents=True, exist_ok=True)
        figure(res, o / "ab_over_clicks.png")
    s = pd.concat([r["summary"] for r in res.values()])
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(s[["beta", "arm", "runs", "departed", "departure_median", "departed_on_the_line", "departed_on_a_phase",
                 "departed_otherwise", "f_prefix_mismatch"]
                + [f"meanF_{lo}_{hi}" for lo, hi in AREAS] + [f"meanF_gain_{lo}_{hi}" for lo, hi in AREAS]
                + [f"meanF_se_{lo}_{hi}" for lo, hi in AREAS] + ["after", "after_gain", "after_se"]].round(4).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm"] + [f"gain_at_{c}" for c in CLICKS]].round(4).to_string(index=False))
        print(s[["beta", "arm", "check_clicks", "done_reached", "done_median", "done_shift_mean", "done_shift_se",
                 "stop_clicks", "stop_clicks_gain", "stop_f", "stop_f_gain", "stop_f_se", "goods_at_50", "goods_at_200",
                 "ap200", "ap200_gain"]].round(4).to_string(index=False))  # fmt: skip
        print(pd.concat([r["splits"] for r in res.values()]).round(4).to_string(index=False))
        print(pd.concat([r["examples"] for r in res.values()]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
