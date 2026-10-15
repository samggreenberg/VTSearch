"""#4909: the weak check's repeat at today's head, as curves over the click a user has reached.

Reads ``weak_repeat_4909.sh``'s arms (Binary Photo, beta 1 and 4, 200 clicks plus the closing check, paired by cell
and seed), which differ only in the weak-separation check:

* ``r25``: the app as shipped, the weak check again 25 votes after a check ends while the labels separate weakly;
* ``r0``: the weak check once;
* ``r50``: again after 50 votes;
* ``nw``: no weak check (the check before Done and the closing check stay).

An arm's run is byte-identical to the control's (``r25``) until the first weak prompt one of them runs and the other
does not, so the **pairing check** is that the first pick the two disagree on is a weak check's pick in one of them.

Per run and click it scores what the app shows on the withheld half, at the run's own beta, as
``early_check_4664.py`` does: the typed query's set until the first click with a detector on screen, then the
detector's line, carried over clicks with no row. Every run ends with the closing check: "after the closing check"
is its last ``check`` row. The pick log's ``check_kind`` splits the clicks spent checking into the weak check's
first run, its repeats, and the check before Done. **Done** is the vote count when the flow reported it (the vote
before its first ``done`` pick); a user who stops there stops at that click, or at 200 if it never comes.

    python weak_repeat_4909.py --root <state-of-the-app dir> --date 2026-10-15 --seeds 5 \\
        --baseline <text_baseline.csv> --out <dir> [--docs <report dir>/tables] [--tag smoke]
"""

from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from early_check_4664 import INK, MUTED, OPENING, STEP_COLS, _axes_style, _headline, area, fbeta, se_over_classes  # noqa: E402

T = 200
GRID_T = np.arange(T + 1)
CONTROL = "r25"
ARMS = ("r25", "r0", "r50", "nw")
OTHERS = ARMS[1:]
BETAS = (1, 4)
NAME = {
    "r25": "weak check again after 25 (as shipped)",
    "r0": "weak check once",
    "r50": "weak check again after 50",
    "nw": "no weak check",
}
COLOR = {"r25": "#8a8985", "r0": "#2a78d6", "r50": "#1baf7a", "nw": "#eb6834"}
CLICKS = (10, 25, 50, 75, 100, 125, 150, 175, 200)
AREAS = ((0, 50), (0, 150), (0, 200))
PICK_COLS = ["category", "seed", "embedder", "t", "phase", "check_kind", "picked_id", "picked_label"]


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


def run_dir(root: Path, date: str, tag: str, arm: str, beta: int) -> Path:
    return root / f"{date}-repeat4909{tag}-{arm}-b{beta}" / "results" / "cells"


def load_arm(d: Path, seeds: int, procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
    if not cells:
        return pd.DataFrame(columns=STEP_COLS), pd.DataFrame(columns=PICK_COLS)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, cells, chunksize=8)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    return steps[steps["seed"] < seeds], picks[picks["seed"] < seeds]


def episodes(kind: np.ndarray, tt: np.ndarray, lab: np.ndarray, which: str) -> list[dict]:
    """The runs of consecutive picks in one *which* check: its start (votes cast before it), end, picks and Goods."""
    out: list[dict] = []
    i, n = 0, len(kind)
    while i < n:
        if kind[i] != which:
            i += 1
            continue
        j = i
        while j + 1 < n and kind[j + 1] == which:
            j += 1
        out.append({"start": float(tt[i - 1]) if i > 0 else 0.0, "end": float(tt[j]), "picks": j - i + 1,
                    "goods": float(lab[i : j + 1].sum())})  # fmt: skip
        i = j + 1
    return out


def per_run(steps: pd.DataFrame, picks: pd.DataFrame, tq: dict, beta: float) -> dict[str, dict]:
    """Per run: the shown set's F at every click, the after-check F, the pick sequence, and the clicks spent checking."""
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
        out[key] = {"f": f, "after": after, "tq": q, "ap200": pd.Series(ap).ffill().to_numpy()[T]}
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
        goods = np.zeros(T + 1)
        click = ph != "check"
        np.add.at(goods, np.clip(tt[click], 0, T), lab[click])
        weak = episodes(kind, tt, lab, "weak")
        done_ck = episodes(kind, tt, lab, "done")
        done_t = tt[ph == "done"]
        # Done shows once the vote before the first pick made in phase ``done`` is cast.
        done_at = float(done_t.min() - 1) if len(done_t) else np.nan
        stop = int(done_at) if np.isfinite(done_at) and done_at <= T else T
        hard = ~np.isin(ph, OPENING + ("check", "prompt"))
        out[key].update(
            seq=(tt, ph, kind, pid),
            goods=np.cumsum(goods),
            weak_n=len(weak),
            weak_picks=int(sum(e["picks"] for e in weak)),
            weak_first_picks=int(weak[0]["picks"]) if weak else 0,
            weak_repeat_picks=int(sum(e["picks"] for e in weak[1:])),
            weak_first_start=weak[0]["start"] if weak else np.nan,
            weak_first_end=weak[0]["end"] if weak else np.nan,
            weak_goods=float(sum(e["goods"] for e in weak)),
            done_check_picks=int(sum(e["picks"] for e in done_ck)),
            done_check_ran=float(len(done_ck) > 0),
            check_clicks=int((ph == "prompt").sum()),
            end_check_picks=int((ph == "check").sum()),
            done_at=done_at,
            stop_clicks=float(stop),
            stop_f=float(out[key]["f"][stop]),
            hard_picks=int(hard.sum()),
            hard_goods=float(lab[hard].sum()),
        )
    return out


def departure(a: tuple, b: tuple) -> tuple[int, bool, float]:
    """Where two pick sequences first disagree: (row, whether a weak check's pick is the reason, the votes before it).

    Row -1 when they never do. The reason is valid when the first disagreeing row is a weak check's pick in one of
    them: one arm started a weak check there and the other did not.
    """
    ta, pa, ka, ia = a
    tb, pb, kb, ib = b
    n = min(len(ta), len(tb))
    same = (ta[:n] == tb[:n]) & (pa[:n] == pb[:n]) & (ka[:n] == kb[:n]) & (ia[:n] == ib[:n])
    bad = np.flatnonzero(~same)
    if not len(bad):
        return (-1, True, float(T)) if len(ta) == len(tb) else (n, False, float(ta[n - 1]) if n else 0.0)
    i = int(bad[0])
    return i, bool(ka[i] == "weak" or kb[i] == "weak"), float(ta[i - 1]) if i > 0 else 0.0


def analyse_beta(runs: dict[str, dict], beta: int) -> dict:
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

    # Pairing: where each arm's picks first leave the control's, and whether a weak check is why.
    dep = {}
    for arm in others:
        rows = [departure(runs[arm][k]["seq"], runs[CONTROL][k]["seq"]) for k in keys]
        at = np.array([r[2] for r in rows])
        moved = np.array([r[0] >= 0 for r in rows])
        valid = np.array([r[1] for r in rows])
        # The shown F must agree up to the departure too (the same picks, the same lines).
        f_ok = np.array([np.allclose(F[arm][i, : int(at[i]) + 1], F[CONTROL][i, : int(at[i]) + 1], equal_nan=True)
                         for i in range(len(keys))])  # fmt: skip
        dep[arm] = {"moved": moved, "at": np.where(moved, at, np.nan), "invalid": int((~valid).sum()),
                    "f_mismatch": int((~f_ok).sum())}  # fmt: skip

    rows = []
    for arm in arms:
        r: dict = {"beta": beta, "arm": arm, "runs": len(keys)}
        if arm != CONTROL:
            d = dep[arm]
            r.update(departed=float(d["moved"].mean()), departure_median=float(np.nanmedian(d["at"]))
                     if d["moved"].any() else np.nan, invalid_departures=d["invalid"], f_prefix_mismatch=d["f_mismatch"])  # fmt: skip
        for lo, hi in AREAS:
            a_ = area(F[arm], lo, hi)
            r[f"meanF_{lo}_{hi}"] = float(np.nanmean(a_))
            g = a_ - area(F[CONTROL], lo, hi)
            r[f"meanF_gain_{lo}_{hi}"], r[f"meanF_se_{lo}_{hi}"] = float(np.nanmean(g)), se_over_classes(g, cls)
        for c in CLICKS:
            g = F[arm][:, c] - F[CONTROL][:, c]
            r[f"F_at_{c}"] = float(np.nanmean(F[arm][:, c]))
            r[f"gain_at_{c}"], r[f"se_at_{c}"] = float(np.nanmean(g)), se_over_classes(g, cls)
        g = after[arm] - after[CONTROL]
        r["after"] = float(np.nanmean(after[arm]))
        r["after_gain"], r["after_se"] = float(np.nanmean(g)), se_over_classes(g, cls)
        if arm != CONTROL and dep[arm]["moved"].any():
            m = dep[arm]["moved"]
            g = area(F[arm], 0, T) - area(F[CONTROL], 0, T)
            r["meanF_gain_0_200_where_moved"] = float(np.nanmean(g[m]))
            r["meanF_se_0_200_where_moved"] = se_over_classes(g[m], cls[m])
        # Clicks spent checking, inside the 200: the weak check's first run, its repeats, and the check before Done.
        for name in ("check_clicks", "weak_picks", "weak_first_picks", "weak_repeat_picks", "done_check_picks",
                     "end_check_picks", "weak_n"):  # fmt: skip
            r[name] = float(np.nanmean(col(arm, name)))
        r["check_share"] = r["check_clicks"] / T
        r["weak_any"] = float(np.mean(col(arm, "weak_n") >= 1))
        r["weak_repeated"] = float(np.mean(col(arm, "weak_n") >= 2))
        r["weak_good_share"] = float(np.nansum(col(arm, "weak_goods")) / max(np.nansum(col(arm, "weak_picks")), 1))
        r["done_check_ran"] = float(np.nanmean(col(arm, "done_check_ran")))
        # The stop: Done by 200, when, and what a user stopping there holds.
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
        r["hard_good_share"] = float(np.nansum(col(arm, "hard_goods")) / max(np.nansum(col(arm, "hard_picks")), 1))
        r["ap200"] = float(np.nanmean(col(arm, "ap200")))
        r["ap200_gain"] = float(np.nanmean(col(arm, "ap200") - col(CONTROL, "ap200")))
        rows.append(r)
    summary = pd.DataFrame(rows)

    # By band; by how many weak checks the control ran (the arms can only differ where it ran two or more, r0 and
    # r50, or one, nw); and, where it ran one, by how the control's session ends (F at click 200): does the weak
    # check help sessions that fail anyway? Gains in mean F over 0-50 and 0-200.
    n_ctl = col(CONTROL, "weak_n")
    by_n = {"0 weak checks": n_ctl == 0, "1": n_ctl == 1, "2": n_ctl == 2, "3 or more": n_ctl >= 3}
    end = F[CONTROL][:, T]
    fired = n_ctl >= 1
    by_end = {"below 0.2": fired & (end < 0.2), "0.2 to 0.5": fired & (end >= 0.2) & (end < 0.5),
              "0.5 or more": fired & (end >= 0.5)}  # fmt: skip
    brow = []
    for arm in others:
        g = area(F[arm], 0, T) - area(F[CONTROL], 0, T)
        g50 = area(F[arm], 0, 50) - area(F[CONTROL], 0, 50)
        ga = after[arm] - after[CONTROL]
        for split, mask_of in (("band", {bd: band == bd for bd in sorted(set(band))}), ("control's weak checks", by_n),
                               ("weak check fired, control's F at 200", by_end)):  # fmt: skip
            for lvl, m in mask_of.items():
                if m.sum() < 2:
                    continue
                brow.append({"beta": beta, "arm": arm, "split": split, "level": lvl, "runs": int(m.sum()),
                             "share": float(m.mean()), "meanF_gain_0_50": float(g50[m].mean()),
                             "se_0_50": se_over_classes(g50[m], cls[m]), "meanF_gain_0_200": float(g[m].mean()),
                             "se": se_over_classes(g[m], cls[m]), "after_gain": float(np.nanmean(ga[m])),
                             "control_F50": float(np.nanmean(F[CONTROL][m, 50])), "arm_F50": float(np.nanmean(F[arm][m, 50])),
                             "control_F200": float(np.nanmean(F[CONTROL][m, T])), "arm_F200": float(np.nanmean(F[arm][m, T])),
                             "check_clicks": float(np.nanmean(col(arm, "check_clicks")[m])),
                             "control_check_clicks": float(np.nanmean(col(CONTROL, "check_clicks")[m])),
                             "better": float(np.mean(g[m] > 1e-9)), "worse": float(np.mean(g[m] < -1e-9))})  # fmt: skip
    splits = pd.DataFrame(brow)

    # Literal examples: per arm, the sessions it helped and hurt most over clicks 0-200, with what the checks did.
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
                    "control_weak_checks": n_ctl[i], "arm_weak_checks": col(arm, "weak_n")[i],
                    "control_check_clicks": col(CONTROL, "check_clicks")[i], "arm_check_clicks": col(arm, "check_clicks")[i],
                    "F_control_at_100": F[CONTROL][i, 100], "F_arm_at_100": F[arm][i, 100],
                    "F_control_at_200": F[CONTROL][i, T], "F_arm_at_200": F[arm][i, T],
                    "after_control": after[CONTROL][i], "after_arm": after[arm][i],
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
    return {"arms": arms, "summary": summary, "splits": splits, "curves": curves, "gains": gains, "F": F, "after": after,
            "examples": examples, "checks": {arm: {n: col(arm, n) for n in ("weak_first_picks", "weak_repeat_picks",
                                                                          "done_check_picks")} for arm in arms}}  # fmt: skip


def figure(res: dict[int, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, len(res), figsize=(6.5 * len(res), 8.2), squeeze=False)
    for j, (beta, r) in enumerate(sorted(res.items())):
        ax = axes[0, j]
        for arm in r["arms"]:
            ax.plot(GRID_T, r["curves"][arm], color=COLOR[arm], lw=2 if arm == CONTROL else 1.6, label=NAME[arm])
            ax.plot([T + 4], [np.nanmean(r["after"][arm])], "o", ms=6, mfc="none", mec=COLOR[arm], mew=1.4)
        ax.set_title(f"F{beta} of the returned set over clicks, beta {beta}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("clicks (circles: after the closing check)", color=MUTED, fontsize=9)
        ax.set_ylabel(f"F{beta} on the withheld half", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1, j]
        for arm in r["arms"][1:]:
            m, se = r["gains"][arm].to_numpy(), r["gains"][f"{arm}_se"].to_numpy()
            ax.fill_between(GRID_T, m - se, m + se, color=COLOR[arm], alpha=0.18, lw=0)
            ax.plot(GRID_T, m, color=COLOR[arm], lw=2, label=NAME[arm])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Paired gain over the app as shipped, beta {beta} (± 1 SE over classes)", color=INK,
                     fontsize=11, loc="left")  # fmt: skip
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_checks(res: dict[int, dict], path: Path) -> None:
    """The clicks each arm spends checking in 200, by check, beside each session's gain in mean F over the control."""
    parts = (("weak_first_picks", "weak check, first run", "#2a78d6"),
             ("weak_repeat_picks", "weak check, repeats", "#9ec5f0"),
             ("done_check_picks", "check before Done", "#c9a227"))  # fmt: skip
    fig, axes = plt.subplots(2, len(res), figsize=(6.5 * len(res), 8.2), squeeze=False)
    for j, (beta, r) in enumerate(sorted(res.items())):
        ax = axes[0, j]
        y = np.arange(len(r["arms"]))[::-1]
        left = np.zeros(len(r["arms"]))
        for name, label, color in parts:
            w = np.array([np.nanmean(r["checks"][arm][name]) for arm in r["arms"]])
            ax.barh(y, w, left=left, color=color, height=0.6, label=label)
            left += w
        for yi, tot in zip(y, left):
            ax.text(tot + 0.8, yi, f"{tot:.0f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(y, [NAME[a] for a in r["arms"]], fontsize=8, color=INK)
        ax.set_title(f"Clicks spent checking in 200, beta {beta}", color=INK, fontsize=11, loc="left")
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1, j]
        bins = np.linspace(-0.08, 0.08, 41)
        for arm in r["arms"][1:]:
            g = area(r["F"][arm], 0, T) - area(r["F"][CONTROL], 0, T)
            moved = np.abs(g) > 1e-9
            ax.hist(np.clip(g[moved], -0.08, 0.08), bins=bins, histtype="step", lw=1.6, color=COLOR[arm],
                    label=f"{NAME[arm]} ({moved.mean():.0%} of sessions moved)")  # fmt: skip
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Per-session gain in mean F, beta {beta}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("paired gain in mean F over clicks 0-200 (clipped to ±0.08)", color=MUTED, fontsize=9)
        ax.set_ylabel("sessions", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--betas", default="1,4")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path, help="the report's tables dir; figures go beside it")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--arms", default=",".join(ARMS), help="the arms to read; the control is always one")
    a = ap.parse_args()
    arms = tuple(x for x in ARMS if x == CONTROL or x in a.arms.split(","))
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == "siglip") & (base["seed"] < a.seeds)]
    res: dict[int, dict] = {}
    counts = {}
    for beta in (int(b) for b in a.betas.split(",")):
        tq = {f"{c}|{int(s)}": float(fbeta([p], [r], beta)[0]) for c, s, p, r in
              zip(base["category"], base["seed"], base[f"text_line_precision_b{beta}"], base[f"text_line_recall_b{beta}"])}  # fmt: skip
        runs = {}
        for arm in arms:
            steps, picks = load_arm(run_dir(a.root, a.date, a.tag, arm, beta), a.seeds, a.procs)
            runs[arm] = per_run(steps, picks, tq, float(beta))
            counts[f"{arm}-b{beta}"] = len(runs[arm])
            print(f"beta {beta} {arm}: {len(runs[arm])} runs", flush=True)
        if min(len(runs[arm]) for arm in arms) == 0:
            print(f"beta {beta}: an arm has no runs yet; skipped")
            continue
        res[beta] = analyse_beta(runs, beta)
    if not res:
        return 1
    outs = [a.out] + ([a.docs] if a.docs else [])
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        for name in ("summary", "splits", "examples", "curves", "gains"):
            pd.concat([r[name] for r in res.values()]).to_csv(o / f"repeat_{name}.csv", index=False)
        (o / "repeat_counts.json").write_text(json.dumps(counts, indent=1))
    figs = [a.out] + ([a.docs.parent / "figures"] if a.docs else [])
    for o in figs:
        o.mkdir(parents=True, exist_ok=True)
        figure(res, o / "repeat_over_clicks.png")
        figure_checks(res, o / "repeat_checks.png")
    s = pd.concat([r["summary"] for r in res.values()])
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(s[["beta", "arm", "runs", "departed", "departure_median", "invalid_departures", "f_prefix_mismatch"]
                + [f"meanF_{lo}_{hi}" for lo, hi in AREAS] + [f"meanF_gain_{lo}_{hi}" for lo, hi in AREAS]
                + [f"meanF_se_{lo}_{hi}" for lo, hi in AREAS] + ["after", "after_gain", "after_se"]].round(4).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm"] + [f"gain_at_{c}" for c in CLICKS] + ["meanF_gain_0_200_where_moved",
                "meanF_se_0_200_where_moved"]].round(4).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm", "check_clicks", "check_share", "weak_picks", "weak_first_picks", "weak_repeat_picks",
                 "done_check_picks", "done_check_ran", "weak_n", "weak_any", "weak_repeated", "weak_good_share",
                 "end_check_picks"]].round(3).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm", "done_reached", "done_median", "done_shift_mean", "done_shift_se", "stop_clicks",
                 "stop_clicks_gain", "stop_f", "stop_f_gain", "stop_f_se", "goods_at_50", "goods_at_200",
                 "hard_good_share", "ap200", "ap200_gain"]].round(4).to_string(index=False))  # fmt: skip
        print(pd.concat([r["splits"] for r in res.values()]).round(4).to_string(index=False))
        print(pd.concat([r["examples"] for r in res.values()]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
