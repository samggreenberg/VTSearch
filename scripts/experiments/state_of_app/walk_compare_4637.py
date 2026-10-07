"""#4637: where Autopilot's More walk should pick once the opening's detector is shown, paired arm against arm.

Two State of the App arms at one commit, identical through the Bad phase:

* **T** - the app: the More walk takes the typed query's top (``CALIB_MORE_WALK`` unset).
* **D** - the walk takes the detector's top, and the detector is shown from its first step
  (``CALIB_MORE_WALK=detector``).

Each arm is first priced by ``handoff_price_4604.py`` (tags ``T_<path>_<b>`` / ``D_<path>_<b>``), which leaves
``runs_<tag>.npz`` (the detector's F at every click, the typed query's F, the hand-off clicks) beside the
extracted steps and picks. Per run and click this reads three returned sets:

* ``T_today``: arm T as the app shows it today, the typed query's set until Hard.
* ``T_after_bad``: arm T with the detector shown from the end of the Bad phase (#4604's ruling, re-priced).
* ``D``: arm D, its walk on the detector's top, which it shows.

The walk question is D against T_after_bad; T_after_bad against T_today re-checks the ruling. Gains are paired
per run, with one SE over classes.

    python walk_compare_4637.py --dir <priced dir> [--docs <report dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T = 150
GRID_T = np.arange(T + 1)
TAGS = [f"{p}_{b}" for p in ("binary", "region") for b in ("b025", "b1", "b4")]
BETA = {"b025": "1/4", "b1": "1", "b4": "4"}
SETS = {
    "T_today": "today: the typed query's set until Hard",
    "T_after_bad": "the walk on the typed query's top, the detector shown from the end of the Bad phase",
    "D": "the walk on the detector's top, which is shown",
}
COLOR = {"T_today": "#8a8985", "T_after_bad": "#2a78d6", "D": "#eb6834"}
CLICKS = (10, 25, 50, 100, 150)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRAINED = ("hard", "new", "done", "exhausted")


def title(tag: str) -> str:
    p, b = tag.split("_")
    return f"{ {'binary': 'Binary Photo', 'region': 'Region Photo'}[p] }, beta {BETA[b]}"


def curve(h: np.ndarray, tq: np.ndarray, det: np.ndarray) -> np.ndarray:
    """The detector's F from hand-off click *h* on (where it has one), else the typed query's (``curve_for``)."""
    use = (GRID_T[None, :] >= h[:, None]) & np.isfinite(det)
    return np.where(use, det, tq[:, None])


def se_over_classes(x: np.ndarray, cls: np.ndarray) -> float:
    by = pd.Series(x).groupby(cls).mean()
    return float(by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def walk_stats(picks: pd.DataFrame) -> pd.DataFrame:
    """Per run: where More and Hard start, the walk's length and hits, and Goods found by a click."""
    rows = []
    for (c, s), g in picks.groupby(["category", "seed"]):
        g = g.sort_values("t")
        ph = g["phase"].astype(str)
        lab = g["picked_label"].to_numpy(dtype=float)
        tt = g["t"].to_numpy(dtype=int)
        more = g.loc[ph == "more"]
        hard = tt[ph.isin(TRAINED).to_numpy()]
        goods = np.cumsum(lab)
        rows.append(
            {
                "key": f"{c}|{int(s)}",
                "more_start": float(more["t"].min()) if len(more) else np.nan,
                "hard_start": float(hard.min()) if len(hard) else np.inf,
                "walk_len": len(more),
                "walk_goods": float(more["picked_label"].sum()) if len(more) else 0.0,
                **{f"goods_{k}": float(goods[tt <= k][-1]) if (tt <= k).any() else 0.0 for k in (25, 50, 150)},
                "pick_ids": tuple(g["picked_id"].to_numpy(dtype=int)),
                "first_more": int(more["t"].min()) if len(more) else int(1e9),
                "goods_walk_end": float(goods[tt <= more["t"].max()][-1]) if len(more) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("key")


def ap_by_click(steps: pd.DataFrame, keys: list[str]) -> np.ndarray:
    ap = np.full((len(keys), T + 1), np.nan)
    pos = {k: i for i, k in enumerate(keys)}
    for (c, s), g in steps.groupby(["category", "seed"]):
        i = pos.get(f"{c}|{int(s)}")
        if i is None:
            continue
        g = g.drop_duplicates("t", keep="last")
        g = g[g["t"] <= T]
        ap[i, g["t"].to_numpy(dtype=int)] = g["average_precision"].to_numpy(dtype=float)
    return pd.DataFrame(ap).ffill(axis=1).to_numpy()


def compare(d: Path, tag: str) -> dict | None:
    zt, zd = (d / f"runs_T_{tag}.npz"), (d / f"runs_D_{tag}.npz")
    if not (zt.exists() and zd.exists()):
        return None
    t, dd = np.load(zt, allow_pickle=True), np.load(zd, allow_pickle=True)
    kt = [f"{c}|{int(s)}" for c, s in zip(t["category"], t["seed"])]
    kd = [f"{c}|{int(s)}" for c, s in zip(dd["category"], dd["seed"])]
    both = sorted(set(kt) & set(kd))
    pt, pd_ = {k: i for i, k in enumerate(kt)}, {k: i for i, k in enumerate(kd)}
    it, idd = np.array([pt[x] for x in both], dtype=int), np.array([pd_[x] for x in both], dtype=int)
    tq = t["tq"][it]
    sets = {
        "T_today": curve(t["shown"][it], tq, t["det"][it]),
        "T_after_bad": curve(np.minimum(t["more"][it], t["shown"][it]), tq, t["det"][it]),
        "D": curve(dd["shown"][idd], dd["tq"][idd], dd["det"][idd]),
    }
    cls = t["cls"][it]
    keep = np.isfinite(tq) & np.all([np.isfinite(v).all(axis=1) for v in sets.values()], axis=0)
    sets = {k: v[keep] for k, v in sets.items()}
    cls, keys = cls[keep], [k for k, m in zip(both, keep) if m]
    wt = walk_stats(pd.read_csv(d / f"picks_T_{tag}.csv.gz")).reindex(keys)
    wd = walk_stats(pd.read_csv(d / f"picks_D_{tag}.csv.gz")).reindex(keys)
    # The arms must be the same session until the walk: the same picks before the first More pick (click t is
    # pick t - 1).
    same = np.array(
        [
            a[: min(fa, fb) - 1] == b[: min(fa, fb) - 1]
            for a, b, fa, fb in zip(wt["pick_ids"], wd["pick_ids"], wt["first_more"], wd["first_more"])
        ]
    )
    ap_t = ap_by_click(pd.read_csv(d / f"steps_T_{tag}.csv.gz"), keys)
    ap_d = ap_by_click(pd.read_csv(d / f"steps_D_{tag}.csv.gz"), keys)
    row: dict = {"tag": tag, "pairs": len(keys), "classes": len(set(cls)), "same_until_walk": float(same.mean())}
    row["only_one_arm"] = len(set(kt) ^ set(kd))
    for name, f in sets.items():
        row[f"{name}_1_50"] = f[:, 1:51].mean()
        row[f"{name}_1_150"] = f[:, 1:].mean()
        for c in CLICKS:
            row[f"{name}_at_{c}"] = f[:, c].mean()
    for a, b, lab in (("D", "T_after_bad", "walk"), ("T_after_bad", "T_today", "handoff"), ("D", "T_today", "total")):
        for lo, hi in ((1, 50), (1, 150)):
            g = (sets[a][:, lo : hi + 1] - sets[b][:, lo : hi + 1]).mean(axis=1)
            row[f"{lab}_{lo}_{hi}"] = g.mean()
            row[f"{lab}_{lo}_{hi}_se"] = se_over_classes(g, cls)
            if (lo, hi) == (1, 50):
                row[f"{lab}_1_50_worse"] = float(np.mean(g < -1e-9))
                row[f"{lab}_1_50_better"] = float(np.mean(g > 1e-9))
    for arm, w, ap in (("T", wt, ap_t), ("D", wd, ap_d)):
        walked = w["walk_len"] > 0
        row[f"{arm}_walk_runs"] = float(walked.mean())
        row[f"{arm}_walk_len"] = w.loc[walked, "walk_len"].median()
        row[f"{arm}_walk_hit"] = w.loc[walked, "walk_goods"].sum() / max(w.loc[walked, "walk_len"].sum(), 1)
        row[f"{arm}_walk_to_20"] = float((w.loc[walked, "goods_walk_end"] >= 20).mean())
        hs = w["hard_start"].replace(np.inf, np.nan)
        row[f"{arm}_hard_start_median"] = hs.median()
        row[f"{arm}_never_hard"] = float(np.isinf(w["hard_start"]).mean())
        for k in (25, 50, 150):
            row[f"{arm}_goods_{k}"] = w[f"goods_{k}"].mean()
        for c in CLICKS:
            row[f"{arm}_ap_at_{c}"] = np.nanmean(ap[:, c])
    diff = {
        lab: sets[a] - sets[b] for a, b, lab in (("D", "T_after_bad", "walk"), ("T_after_bad", "T_today", "handoff"))
    }
    band = {}
    for lab, g in diff.items():
        by = pd.DataFrame(g).groupby(cls).mean()
        band[lab] = (g.mean(axis=0), by.std(ddof=1).to_numpy() / np.sqrt(len(by)))
    return {"row": row, "curves": {k: v.mean(axis=0) for k, v in sets.items()}, "band": band}


def _axes_style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def figure_curves(res: dict[str, dict], out: Path) -> None:
    tags = [t for t in TAGS if t in res]
    nrows = (len(tags) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 3.7 * nrows + 0.6), sharex=True, squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        for k, lab in SETS.items():
            ax.plot(GRID_T, res[tag]["curves"][k], color=COLOR[k], lw=2, label=lab)
        ax.set_title(title(tag), color=INK, fontsize=11, loc="left")
        _axes_style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("F-beta of the returned set (withheld half)", color=MUTED, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.16 / nrows, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_gain(res: dict[str, dict], out: Path) -> None:
    tags = [t for t in TAGS if t in res]
    nrows = (len(tags) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 3.4 * nrows + 0.6), sharex=True, sharey=True, squeeze=False)
    names = {
        "walk": ("the detector's top against the typed query's, both shown from the end of the Bad phase", "#eb6834"),
        "handoff": ("showing the detector from the end of the Bad phase against today, on today's walk", "#2a78d6"),
    }
    for ax, tag in zip(axes.flat, tags):
        for lab, (name, col) in names.items():
            m, se = res[tag]["band"][lab]
            ax.fill_between(GRID_T, m - se, m + se, color=col, alpha=0.18, lw=0)
            ax.plot(GRID_T, m, color=col, lw=2, label=name)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title(tag), color=INK, fontsize=11, loc="left")
        _axes_style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("paired gain in F-beta (± 1 SE over classes)", color=MUTED, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.14 / nrows, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--docs", type=Path, help="also write the figures and tables here")
    a = ap.parse_args()
    res = {tag: r for tag in TAGS if (r := compare(a.dir, tag)) is not None}
    if not res:
        raise SystemExit(f"no paired runs_T_*/runs_D_* under {a.dir}")
    summ = pd.DataFrame([r["row"] for r in res.values()])
    outs = [a.dir] + ([a.docs] if a.docs else [])
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        summ.to_csv(o / "walk_summary.csv", index=False)
        pd.DataFrame(
            {f"{tag}:{k}": v for tag, r in res.items() for k, v in r["curves"].items()} | {"t": GRID_T}
        ).to_csv(o / "walk_curves.csv", index=False)
        figure_curves(res, o / "walk_over_clicks.png")
        figure_gain(res, o / "walk_gain_over_clicks.png")
    cols = ["tag", "pairs", "same_until_walk"] + [
        f"{lab}_1_{hi}{s}" for lab in ("walk", "handoff", "total") for hi in (50, 150) for s in ("", "_se")
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ[cols].round(3).to_string(index=False))
        print(
            summ[["tag"] + [c for c in summ.columns if c.startswith(("T_", "D_")) and "walk" in c]]
            .round(2)
            .to_string(index=False)
        )
        print(summ[["tag"] + [c for c in summ.columns if "goods" in c or "hard" in c]].round(2).to_string(index=False))
        print(summ[["tag"] + [c for c in summ.columns if "_ap_at_" in c]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
