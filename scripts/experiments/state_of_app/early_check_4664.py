"""#4664: a check in every session, early, against Autopilot as shipped, as curves over the click a user has reached.

Reads ``early_check_4664.sh``'s arms (Binary Photo, beta 1 and 4, paired by cell and seed):

* ``a0``: today's app (its own checks: weak separation, and before Done), 200 clicks;
* ``e30`` / ``e50`` / ``e75``: the same, with every session checked at 30 / 50 / 75 votes (``early_check``): a
  prompted check running then counts, otherwise Autopilot prompts one at the first step from N votes past the
  opening, however recently another ended.

The **cooldown variant** (skip where a check ended in the last 25 votes, as the check before Done is skipped) needs
no run of its own: the two arms are byte-identical until the early check, so it is the e-arm's run where the early
check fired 25 or more votes after the control's last check ended, and the control's run where it would skip.

Per run and click it scores what the app shows on the withheld half, at the run's own beta: the typed query's set
until the first click with a detector on screen (``app_trained``), then the detector's line, carried over clicks
with no row (a check round's rows sit at its last vote). Every run ends with the app's check, so each also has an
"after the closing check" value: its last ``check`` row. The pick log's ``check_kind`` says which check each check
pick was in, so the clicks after the early check can be read too: later weak prompts, the check before Done, Done
itself, and Goods.

    python early_check_4664.py --root <state-of-the-app dir> --date 2026-10-14 --seeds 5 \\
        --baseline <text_baseline.csv> --out <dir> [--docs <report dir>] [--tag smoke]
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

T = 200
GRID_T = np.arange(T + 1)
ARMS = {"a0": None, "e30": 30, "e50": 50, "e75": 75}
#: #4482's c50 at today's head: the app stopping at click 50, so its closing check is the check where the user stops.
STOP_ARM, STOP_AT = "c50", 50
EARLY = ("e30", "e50", "e75")
BETAS = (1, 4)
NAME = {
    "a0": "Autopilot as shipped",
    "e30": "+ a check at click 30",
    "e50": "+ a check at click 50",
    "e75": "+ a check at click 75",
}
COLOR = {"a0": "#8a8985", "e30": "#2a78d6", "e50": "#1baf7a", "e75": "#eb6834"}
CLICKS = (10, 25, 40, 50, 60, 75, 90, 100, 125, 150, 175, 200)
#: Windows of clicks a session mean is read over: short sessions first, then the fade.
WINDOWS = ((1, 25), (1, 50), (1, 100), (1, 150), (1, 200), (51, 100), (101, 150), (151, 200))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OPENING = ("good", "bad", "more")
STEP_COLS = [
    "category", "seed", "embedder", "t", "phase", "app_trained", "precision", "recall", "average_precision",
    "gmm_variant", "schedule", "pool_variant",
]  # fmt: skip
PICK_COLS = ["category", "seed", "embedder", "t", "phase", "check_kind", "picked_label"]


def fbeta(p, r, beta: float) -> np.ndarray:
    """F-beta, with an empty set (precision NaN, recall 0) scoring 0 (``analyze._fbeta_pr``)."""
    p, r = np.asarray(p, dtype=float), np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    f = np.where(np.isfinite(p) & np.isfinite(r), f, np.nan)
    return np.where(~np.isfinite(p) & (r == 0), 0.0, f)


def _headline(m: pd.DataFrame) -> pd.DataFrame:
    """The headline row only: every step also carries ~30 diagnostic cut variants (``gmm_variant``)."""
    for col in ("gmm_variant", "schedule"):
        if col in m.columns:
            m = m[m[col].isna() | m[col].astype(str).str.strip().isin(("", "nan", "None"))]
    if "pool_variant" in m.columns:
        m = m[m["pool_variant"].fillna("").astype(str).str.strip().isin(("", "max"))]
    return m


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
    pk = pk[pk["embedder"] == "siglip"]
    if "check_kind" not in pk.columns:
        pk = pk.assign(check_kind="")
    return m, pk


def run_dir(root: Path, date: str, tag: str, arm: str, beta: int) -> Path:
    return root / f"{date}-early4664{tag}-{arm}-b{beta}" / "results" / "cells"


def load_arm(d: Path, seeds: int, procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
    if not cells:
        return pd.DataFrame(columns=STEP_COLS), pd.DataFrame(columns=PICK_COLS)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, cells, chunksize=8)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    return steps[steps["seed"] < seeds], picks[picks["seed"] < seeds]


def _first_t(ts: np.ndarray) -> float:
    return float(ts.min()) if len(ts) else np.nan


def per_run(steps: pd.DataFrame, picks: pd.DataFrame, tq: dict, beta: float) -> dict[str, dict]:
    """Per run: the shown set's F at every click, the after-check F, and what the checks and picks did."""
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
        out[key] = {"f": f, "after": after, "shown": shown, "tq": q, "ap200": pd.Series(ap).ffill().to_numpy()[T]}
    for (c, s), g in picks.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        if key not in out:
            continue
        g = g.reset_index(drop=True)
        ph = g["phase"].astype(str).to_numpy()
        kind = g["check_kind"].fillna("").astype(str).to_numpy()
        lab = g["picked_label"].to_numpy(dtype=float)
        tt = g["t"].to_numpy(dtype=int)
        click = ph != "check"
        goods = np.zeros(T + 1)
        np.add.at(goods, np.clip(tt[click], 0, T), lab[click])
        early = np.flatnonzero(kind == "early")
        if len(early):
            i0 = early[0]
            start = float(tt[i0 - 1]) if i0 > 0 else 0.0
            end = float(tt[early[-1]])
        else:
            start = end = np.nan
        weak = kind == "weak"
        done_ck = kind == "done"
        after_early = tt > end if np.isfinite(end) else np.zeros(len(tt), bool)
        hard = ~np.isin(ph, OPENING + ("check", "prompt"))
        out[key].update(
            goods=np.cumsum(goods),
            # Every prompted pick's vote count: where the control's checks ended, for the cooldown variant.
            prompt_ts=tt[ph == "prompt"],
            early_start=start,
            early_end=end,
            early_picks=int(len(early)),
            early_goods=float(lab[early].sum()),
            # The pre-existing checks' clicks, and those after the early check closed.
            weak_picks=int(weak.sum()),
            weak_before=_first_t(tt[weak]),
            weak_picks_after_early=int((weak & after_early).sum()),
            done_check_picks=int(done_ck.sum()),
            check_clicks=int((ph == "prompt").sum()),
            done_click=_first_t(tt[ph == "done"]),
            hard_picks=int(hard.sum()),
            hard_goods=float(lab[hard].sum()),
            end_check_picks=int((ph == "check").sum()),
        )
    return out


def se_over_classes(x: np.ndarray, cls: np.ndarray) -> float:
    """SE of the mean over classes: the class is the unit (a class's bands and seeds are not independent)."""
    by = pd.Series(x).groupby(cls).mean().dropna()
    return float(by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def area(f: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per run, the trapezoid area under F over clicks lo..hi, divided by the span (the owner's single number)."""
    return np.trapezoid(f[:, lo : hi + 1], dx=1.0, axis=1) / (hi - lo)


def _axes_style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def analyse_beta(runs: dict[str, dict], beta: int) -> dict:
    keys = sorted(set.intersection(*(set(runs[a]) for a in ARMS)))
    keys = [k for k in keys if np.isfinite(runs["a0"][k]["tq"]) and "goods" in runs["a0"][k]]
    cls = np.array([k.split("|")[0].split("@")[0] for k in keys])
    band = np.array([k.split("|")[0].split("@")[1] for k in keys])
    F = {arm: np.vstack([runs[arm][k]["f"] for k in keys]) for arm in ARMS}
    G = {arm: np.vstack([runs[arm][k]["goods"] for k in keys]) for arm in ARMS}
    after = {arm: np.array([runs[arm][k]["after"] for k in keys]) for arm in ARMS}
    col = {
        arm: (lambda name, arm=arm: np.array([runs[arm][k].get(name, np.nan) for k in keys], dtype=float))
        for arm in ARMS
    }

    # The control's runs are byte-identical to an arm's until the early check: a pair that differs before it is a
    # broken pairing (another commit, a node's BLAS), not an effect.
    prefix = {}
    for arm in EARLY:
        st = col[arm]("early_start")
        lim = np.where(np.isfinite(st), st, T).astype(int)
        bad = [
            i
            for i in range(len(keys))
            if not np.allclose(F[arm][i, : lim[i] + 1], F["a0"][i, : lim[i] + 1], equal_nan=True)
        ]
        prefix[arm] = len(bad)

    # The cooldown variant: the e-arm's run where its check started 25+ votes after the control's last prompted check
    # ended (or with none before), the control's run where the variant would have skipped.
    COOLDOWN = 25
    cool_on = {}
    for arm in EARLY:
        st = col[arm]("early_start")
        on = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            if np.isfinite(st[i]):
                before = runs["a0"][k]["prompt_ts"]
                before = before[before <= st[i]]
                on[i] = (not len(before)) or (st[i] - before.max() >= COOLDOWN)
        cool_on[arm] = on
        F[f"{arm}c"] = np.where(on[:, None], F[arm], F["a0"])
        G[f"{arm}c"] = np.where(on[:, None], G[arm], G["a0"])
        after[f"{arm}c"] = np.where(on, after[arm], after["a0"])

    rows = []
    for arm in EARLY + tuple(f"{a}c" for a in EARLY):
        base_arm = arm.rstrip("c")

        def C(name: str, arm: str = arm, base_arm: str = base_arm) -> np.ndarray:
            """A per-run tally of *arm*: the cooldown variant's is the e-arm's where it ran, the control's elsewhere."""
            if arm == base_arm:
                return col[arm](name)
            return np.where(cool_on[base_arm], col[base_arm](name), col["a0"](name))

        st = C("early_start")
        ran = np.isfinite(st) & (cool_on[base_arm] if arm.endswith("c") else True)
        r = {"beta": beta, "arm": arm, "runs": len(keys), "prefix_mismatch": prefix[base_arm], "ran": float(ran.mean())}
        # Not run because a prompted check was running at N (it counts as the session's check).
        r["absorbed"] = float(np.mean(~np.isfinite(col[base_arm]("early_start"))))
        r["start_median"] = float(np.nanmedian(st)) if ran.any() else np.nan
        r["start_p90"] = float(np.nanpercentile(st, 90)) if ran.any() else np.nan
        r["early_picks"] = float(np.nanmean(C("early_picks")[ran])) if ran.any() else 0.0
        r["early_good_share"] = C("early_goods")[ran].sum() / max(C("early_picks")[ran].sum(), 1)
        # Why it did not run: a check had ended within the cooldown (the control checked just before), or none fit.
        a0_weak = col["a0"]("weak_before")
        r["skipped_after_a_check"] = float(np.mean(~ran & np.isfinite(a0_weak)))
        for lo, hi in WINDOWS:
            g = np.nanmean(F[arm][:, lo : hi + 1] - F["a0"][:, lo : hi + 1], axis=1)
            r[f"gain_{lo}_{hi}"], r[f"se_{lo}_{hi}"] = np.nanmean(g), se_over_classes(g, cls)
        for lo, hi in ((0, 50), (0, 150), (0, 200)):
            g = area(F[arm], lo, hi) - area(F["a0"], lo, hi)
            r[f"area_{lo}_{hi}"], r[f"area_se_{lo}_{hi}"] = np.nanmean(g), se_over_classes(g, cls)
        for c in CLICKS:
            g = F[arm][:, c] - F["a0"][:, c]
            r[f"gain_at_{c}"], r[f"se_at_{c}"] = np.nanmean(g), se_over_classes(g, cls)
        g = after[arm] - after["a0"]
        r["after_gain"], r["after_se"] = np.nanmean(g), se_over_classes(g, cls)
        # Where it ran: the gain at the click its check closed, against the control at the same click, and per pick.
        end = C("early_end")
        at_end = np.array(
            [F[arm][i, int(e)] - F["a0"][i, int(e)] if np.isfinite(e) and e <= T else np.nan for i, e in enumerate(end)]
        )
        r["gain_at_close"], r["se_at_close"] = np.nanmean(at_end), se_over_classes(at_end, cls)
        r["gain_at_close_per_pick"] = r["gain_at_close"] / max(r["early_picks"], 1e-9)
        g150 = np.nanmean(F[arm][:, 1:151] - F["a0"][:, 1:151], axis=1)
        r["gain_1_150_where_ran"] = np.nanmean(g150[ran]) if ran.any() else np.nan
        r["gain_1_150_where_ran_se"] = se_over_classes(g150[ran], cls[ran]) if ran.sum() > 2 else np.nan
        # The clicks after it: the app's own checks, Done, Goods.
        for name in ("weak_picks", "done_check_picks", "check_clicks", "end_check_picks"):
            r[name], r[f"a0_{name}"] = np.nanmean(C(name)), np.nanmean(col["a0"](name))
        r["weak_picks_after_early"] = np.nanmean(C("weak_picks_after_early"))
        for name in ("done_click",):
            r[f"{name}_reached"], r[f"a0_{name}_reached"] = (
                np.mean(np.isfinite(C(name))),
                np.mean(np.isfinite(col["a0"](name))),
            )
            r[f"{name}_median"], r[f"a0_{name}_median"] = np.nanmedian(C(name)), np.nanmedian(col["a0"](name))
        for c in (50, 100, 150, 200):
            r[f"goods_at_{c}"], r[f"a0_goods_at_{c}"] = np.nanmean(G[arm][:, c]), np.nanmean(G["a0"][:, c])
        r["hard_good_share"] = C("hard_goods").sum() / max(C("hard_picks").sum(), 1)
        r["a0_hard_good_share"] = col["a0"]("hard_goods").sum() / max(col["a0"]("hard_picks").sum(), 1)
        r["ap200_gain"] = np.nanmean(C("ap200") - col["a0"]("ap200"))
        rows.append(r)
    summary = pd.DataFrame(rows)

    # By band, and by what the session was doing at N votes: a check running (it counts, no early check), a check
    # ended under 25 votes before (the early check fires; the cooldown variant skips), or neither. Gain over 1-150.
    brow = []
    for arm in EARLY:
        g = np.nanmean(F[arm][:, 1:151] - F["a0"][:, 1:151], axis=1)
        fired = np.isfinite(col[arm]("early_start"))
        context = {"absorbed": ~fired, "fired, a check ended < 25 before": fired & ~cool_on[arm],
                   "fired, none in the 25 before": fired & cool_on[arm]}  # fmt: skip
        for split, mask_of in (("band", {bd: band == bd for bd in sorted(set(band))}), ("at N", context)):
            for lvl, m in mask_of.items():
                if m.sum() < 2:
                    continue
                brow.append({"beta": beta, "arm": arm, "split": split, "level": lvl, "runs": int(m.sum()),
                             "share": float(m.mean()), "gain_1_150": g[m].mean(), "se": se_over_classes(g[m], cls[m]),
                             "better": float(np.mean(g[m] > 1e-9)), "worse": float(np.mean(g[m] < -1e-9))})  # fmt: skip
    splits = pd.DataFrame(brow)

    # Literal examples: per early arm, the sessions it helped and hurt most over clicks 1-150, with what they did.
    erow = []
    for arm in EARLY:
        g = np.nanmean(F[arm][:, 1:151] - F["a0"][:, 1:151], axis=1)
        fired = np.isfinite(col[arm]("early_start"))
        order = np.argsort(np.where(fired, g, np.nan))
        order = order[np.isfinite(g[order]) & fired[order]]
        for tag, idx in (("helped most", order[::-1][:3]), ("hurt most", order[:3])):
            for i in idx:
                cat, seed = keys[i].split("|")
                erow.append({
                    "beta": beta, "arm": arm, "which": tag, "category": cat, "seed": int(seed),
                    "gain_1_150": g[i], "early_start": col[arm]("early_start")[i], "early_end": col[arm]("early_end")[i],
                    "early_goods": col[arm]("early_goods")[i], "a0_weak_before": col["a0"]("weak_before")[i],
                    "F_a0_at_100": F["a0"][i, 100], "F_arm_at_100": F[arm][i, 100],
                    "goods_200_a0": G["a0"][i, T], "goods_200_arm": G[arm][i, T],
                })  # fmt: skip
    examples = pd.DataFrame(erow)
    curves = pd.DataFrame({"beta": beta, "t": GRID_T, **{arm: np.nanmean(F[arm], axis=0) for arm in F}})
    gains = pd.DataFrame({"beta": beta, "t": GRID_T})
    for arm in EARLY + tuple(f"{a}c" for a in EARLY):
        d = F[arm] - F["a0"]
        gains[arm] = np.nanmean(d, axis=0)
        by = pd.DataFrame(d).groupby(cls).mean()
        gains[f"{arm}_se"] = by.std(ddof=1).to_numpy() / np.sqrt(len(by))
    stop = stop_table(runs, keys, cls, F, col, cool_on, beta) if STOP_ARM in runs else pd.DataFrame()
    return {
        "summary": summary,
        "splits": splits,
        "curves": curves,
        "gains": gains,
        "F": F,
        "after": after,
        "stop": stop,
        "examples": examples,
        "seed": np.array([int(k.split("|")[1]) for k in keys]),
    }


def stop_table(runs, keys, cls, F, col, cool_on, beta) -> pd.DataFrame:
    """The check at a stop (c50) beside the same check inside the session (e50), on the same sessions.

    c50's closing check against its unchecked line at click 50 is #4482's "the check"; the control's F at click
    50 + k (k = the check's picks) against its F at 50 is "the same number of further Autopilot clicks". e50's
    gain where its check closes, against the control at that click, is what the check is worth when the session
    goes on. Split by what the session was doing at vote 50: a prompted check running or ended in the 25 votes
    before (the control had just checked), or not.
    """
    n = STOP_AT
    c = runs[STOP_ARM]
    ok = np.array([k in c for k in keys])
    unchecked = np.array([c[k]["f"][n] if k in c else np.nan for k in keys])
    after = np.array([c[k]["after"] if k in c else np.nan for k in keys])
    picks = np.array([c[k].get("end_check_picks", np.nan) if k in c else np.nan for k in keys], dtype=float)
    further = np.array(
        [
            F["a0"][i, int(min(n + picks[i], T))] - F["a0"][i, n] if np.isfinite(picks[i]) else np.nan
            for i in range(len(keys))
        ]
    )
    e_end = col["e50"]("early_end")
    at_close = np.array(
        [F["e50"][i, int(e)] - F["a0"][i, int(e)] if np.isfinite(e) else np.nan for i, e in enumerate(e_end)]
    )
    recent = ~cool_on["e50"]  # absorbed, or fired within 25 votes of the control's last check
    rows = []
    for name, m in (
        ("all", ok),
        ("checked in the 25 votes before 50", ok & recent),
        ("not checked recently", ok & ~recent),
    ):
        g = after[m] - unchecked[m]
        rows.append({
            "beta": beta, "sessions": name, "runs": int(m.sum()), "share": float(m.mean()),
            "a0_F_at_50": np.nanmean(F["a0"][m, n]), "c50_unchecked_at_50": np.nanmean(unchecked[m]),
            "after_check": np.nanmean(after[m]), "check_gain": np.nanmean(g), "check_se": se_over_classes(g, cls[m]),
            "check_picks": np.nanmean(picks[m]), "same_clicks_autopilot": np.nanmean(further[m]),
            "same_clicks_se": se_over_classes(further[m], cls[m]),
            "e50_fired": float(np.mean(np.isfinite(e_end[m]))), "e50_gain_at_close": np.nanmean(at_close[m]),
            "e50_close_se": se_over_classes(at_close[m], cls[m]) if np.isfinite(at_close[m]).sum() > 2 else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def figure(res: dict[int, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, len(res), figsize=(6.5 * len(res), 8.2), squeeze=False)
    for j, (beta, r) in enumerate(sorted(res.items())):
        ax = axes[0, j]
        for arm in ARMS:
            ax.plot(GRID_T, r["curves"][arm], color=COLOR[arm], lw=2 if arm == "a0" else 1.6, label=NAME[arm])
            ax.plot([T + 4], [np.nanmean(r["after"][arm])], "o", ms=6, mfc="none", mec=COLOR[arm], mew=1.4)
        ax.set_title(f"F{beta} of the returned set over clicks, beta {beta}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("clicks (circles: after the closing check)", color=MUTED, fontsize=9)
        ax.set_ylabel(f"F{beta} on the withheld half", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1, j]
        for arm in EARLY:
            m, se = r["gains"][arm].to_numpy(), r["gains"][f"{arm}_se"].to_numpy()
            ax.fill_between(GRID_T, m - se, m + se, color=COLOR[arm], alpha=0.18, lw=0)
            ax.plot(GRID_T, m, color=COLOR[arm], lw=2, label=NAME[arm])
            ax.axvline(ARMS[arm], color=COLOR[arm], lw=0.8, ls=":")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(
            f"Paired gain over Autopilot as shipped, beta {beta} (± 1 SE over classes)",
            color=INK,
            fontsize=11,
            loc="left",
        )
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_runs(res: dict[int, dict], path: Path) -> None:
    """Per run, not averaged: e50's paired gain per seed over clicks, and every session's gain over clicks 1-150."""
    fig, axes = plt.subplots(2, len(res), figsize=(6.5 * len(res), 8.2), squeeze=False)
    seed_colors = ("#2a78d6", "#1baf7a", "#eb6834", "#8a5cc7", "#c9a227")
    for j, (beta, r) in enumerate(sorted(res.items())):
        ax = axes[0, j]
        d = r["F"]["e50"] - r["F"]["a0"]
        for s_, c in zip(sorted(set(r["seed"])), seed_colors):
            m = pd.Series(np.nanmean(d[r["seed"] == s_], axis=0)).rolling(5, center=True, min_periods=1).mean()
            ax.plot(GRID_T, m, color=c, lw=1.4, label=f"seed {s_}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(50, color=MUTED, lw=0.8, ls=":")
        ax.set_title(
            f"+ a check at click 50, gain per seed, beta {beta} (5-click mean)", color=INK, fontsize=11, loc="left"
        )
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=5)
        ax = axes[1, j]
        bins = np.linspace(-0.1, 0.1, 41)
        for arm in EARLY:
            g = np.nanmean(r["F"][arm][:, 1:151] - r["F"]["a0"][:, 1:151], axis=1)
            moved = np.abs(g) > 1e-9
            ax.hist(np.clip(g[moved], -0.1, 0.1), bins=bins, histtype="step", lw=1.6, color=COLOR[arm],
                    label=f"{NAME[arm]} ({moved.mean():.0%} of sessions moved)")  # fmt: skip
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Each session's gain over clicks 1-150, beta {beta}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("paired gain in mean F over clicks 1-150 (clipped to ±0.1)", color=MUTED, fontsize=9)
        ax.set_ylabel("sessions", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-14")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--betas", default="1,4")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == "siglip") & (base["seed"] < a.seeds)]
    res: dict[int, dict] = {}
    counts = {}
    for beta in (int(b) for b in a.betas.split(",")):
        tq = {f"{c}|{int(s)}": float(fbeta([p], [r], beta)[0]) for c, s, p, r in
              zip(base["category"], base["seed"], base[f"text_line_precision_b{beta}"], base[f"text_line_recall_b{beta}"])}  # fmt: skip
        runs = {}
        for arm in (*ARMS, STOP_ARM):
            steps, picks = load_arm(run_dir(a.root, a.date, a.tag, arm, beta), a.seeds, a.procs)
            if arm == STOP_ARM and not len(steps):
                continue
            runs[arm] = per_run(steps, picks, tq, float(beta))
            counts[f"{arm}-b{beta}"] = len(runs[arm])
            print(f"beta {beta} {arm}: {len(runs[arm])} runs", flush=True)
        if min(len(runs[arm]) for arm in ARMS) == 0:
            print(f"beta {beta}: an arm has no runs yet; skipped")
            continue
        res[beta] = analyse_beta(runs, beta)
    if not res:
        return 1
    outs = [a.out] + ([a.docs] if a.docs else [])
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        pd.concat([r["summary"] for r in res.values()]).to_csv(o / "early_summary.csv", index=False)
        pd.concat([r["splits"] for r in res.values()]).to_csv(o / "early_splits.csv", index=False)
        pd.concat([r["stop"] for r in res.values()]).to_csv(o / "early_check_at_stop.csv", index=False)
        pd.concat([r["examples"] for r in res.values()]).to_csv(o / "early_examples.csv", index=False)
        pd.concat([r["curves"] for r in res.values()]).to_csv(o / "early_curves.csv", index=False)
        pd.concat([r["gains"] for r in res.values()]).to_csv(o / "early_gains.csv", index=False)
        (o / "early_counts.json").write_text(json.dumps(counts, indent=1))
    figs = [a.out] + ([a.docs.parent / "figures"] if a.docs else [])
    for o in figs:
        o.mkdir(parents=True, exist_ok=True)
        figure(res, o / "early_over_clicks.png")
        figure_runs(res, o / "early_per_run.png")
    s = pd.concat([r["summary"] for r in res.values()])
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(s[["beta", "arm", "runs", "prefix_mismatch", "ran", "absorbed", "start_median", "early_picks", "early_good_share"]
                + [f"gain_{lo}_{hi}" for lo, hi in WINDOWS] + ["after_gain"]].round(4).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm"] + [f"area_{lo}_{hi}" for lo, hi in ((0, 50), (0, 150), (0, 200))]
                + [f"area_se_{lo}_{hi}" for lo, hi in ((0, 50), (0, 150), (0, 200))]
                + ["gain_at_close", "gain_at_close_per_pick", "gain_1_150_where_ran"]].round(4).to_string(index=False))  # fmt: skip
        print(s[["beta", "arm", "weak_picks", "a0_weak_picks", "weak_picks_after_early", "done_check_picks",
                 "a0_done_check_picks", "check_clicks", "a0_check_clicks", "done_click_reached", "a0_done_click_reached",
                 "done_click_median", "a0_done_click_median", "goods_at_200", "a0_goods_at_200"]].round(3).to_string(index=False))  # fmt: skip
        print(pd.concat([r["splits"] for r in res.values()]).round(4).to_string(index=False))
        print(pd.concat([r["stop"] for r in res.values()]).round(4).to_string(index=False))
        print(pd.concat([r["examples"] for r in res.values()]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
