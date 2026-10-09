"""#4731: the COCO guard. Does a Good phase that runs dry cost the State of the App's Binary sessions?

Three arms at one commit, priced first by ``handoff_price_4604.py`` (tags ``C_binary_b1`` / ``Q_binary_b1`` /
``D_binary_b1``), which leaves ``runs_<tag>.npz`` beside the extracted steps and picks. Each arm is read as the
app shows it: the typed query's set until the session first shows a detector (``app_trained``), the detector's
line from then on. For ``Q`` and ``D`` that is the click the Good walk ran dry, where it ran dry.

Per run (class@band x seed), paired against ``C``: the F-beta curve at every click, its mean over 1-150 (the
area the owner reads as one number) and over 1-50, overall, by band, and split by whether the dry run fires in
``C``'s own walk (a run where it never fires is the same session in every arm). One SE over classes.

    python compare_good_dry_4731.py --dir <priced dir> --picks-c <picks_C_binary_b1.csv.gz> [--docs <report dir>]
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
DRY = 16
ARMS = {"C": "the app", "Q": "dry run 16 + the quota tier", "D": "dry run 16 alone"}


def today(z) -> np.ndarray:
    """Per run and click: the detector's F from the first shown click, else the typed query's."""
    det, tq, shown = z["det"], z["tq"], z["shown"]
    use = (GRID_T[None, :] >= shown[:, None]) & np.isfinite(det)
    return np.where(use, det, tq[:, None])


def keys(z) -> list[str]:
    return [f"{c}|{int(s)}" for c, s in zip(z["category"], z["seed"])]


def fires(picks: pd.DataFrame) -> dict[str, float]:
    """Per run: the click a Good walk of DRY misses with a Good in hand ends in this arm's picks, else inf."""
    out = {}
    for (c, s), g in picks.groupby(["category", "seed"]):
        goods, misses, at = 0, 0, np.inf
        for p in g.sort_values("t").itertuples():
            if p.phase != "good":
                break
            if p.picked_label == 1:
                goods, misses = goods + 1, 0
            elif goods >= 1:
                misses += 1
                if misses >= DRY:
                    at = float(p.t)
                    break
        out[f"{c}|{int(s)}"] = at
    return out


def se_over_classes(x: np.ndarray, cls: np.ndarray) -> float:
    by = pd.Series(x).groupby(cls).mean()
    return float(by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--picks-c", required=True, type=Path)
    ap.add_argument("--docs", type=Path)
    a = ap.parse_args()

    z = {
        arm: np.load(a.dir / f"runs_{arm}_binary_b1.npz", allow_pickle=True)
        for arm in ARMS
        if (a.dir / f"runs_{arm}_binary_b1.npz").exists()
    }
    if "C" not in z:
        raise SystemExit("no control arm (runs_C_binary_b1.npz)")
    order = keys(z["C"])
    F = {}
    for arm, zz in z.items():
        k = keys(zz)
        pos = {kk: i for i, kk in enumerate(k)}
        missing = [kk for kk in order if kk not in pos]
        if missing:
            raise SystemExit(f"arm {arm} lacks {len(missing)} of the control's runs, e.g. {missing[:3]}")
        F[arm] = today(zz)[[pos[kk] for kk in order]]
    cls = z["C"]["cls"]
    band = z["C"]["band"]
    fire_at = fires(pd.read_csv(a.picks_c))
    fired = np.array([np.isfinite(fire_at.get(kk, np.inf)) for kk in order])
    print(f"runs {len(order)}; the dry run fires in {fired.sum()} ({fired.mean():.1%}) of the control's walks")

    rows = []
    for arm in F:
        for subset, m in (
            ("all", np.ones(len(order), bool)),
            ("fires", fired),
            ("never fires", ~fired),
            *((f"band {b}", band == b) for b in ("small", "medium", "large")),
        ):
            if not m.any():
                continue
            area = F[arm][m, 1:].mean(axis=1)
            short = F[arm][m, 1:51].mean(axis=1)
            row = {"arm": arm, "subset": subset, "runs": int(m.sum()), "auc150": area.mean(), "auc50": short.mean()}
            if arm != "C":
                d150 = area - F["C"][m, 1:].mean(axis=1)
                d50 = short - F["C"][m, 1:51].mean(axis=1)
                row |= {
                    "gain150": d150.mean(),
                    "gain150_se": se_over_classes(d150, cls[m]),
                    "gain50": d50.mean(),
                    "gain50_se": se_over_classes(d50, cls[m]),
                    "runs_better150": int((d150 > 1e-9).sum()),
                    "runs_worse150": int((d150 < -1e-9).sum()),
                }
            rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(a.dir / "compare_good_dry_4731.csv", index=False)
    curves = pd.DataFrame({"t": GRID_T, **{f"{arm}_all": F[arm].mean(0) for arm in F}})
    for arm in F:
        curves[f"{arm}_fires"] = F[arm][fired].mean(0) if fired.any() else np.nan
    curves.to_csv(a.dir / "curves_good_dry_4731.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(summ.to_string(index=False))

    if a.docs:
        a.docs.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for i, (name, m) in enumerate(
            (("every run", np.ones(len(order), bool)), ("runs where the dry run fires", fired))
        ):
            for arm, color in (("C", "#8a8985"), ("Q", "#eb6834"), ("D", "#2a78d6")):
                if arm in F and m.any():
                    ax[i].plot(GRID_T[1:], F[arm][m, 1:].mean(0), color=color, lw=2, label=f"{arm}: {ARMS[arm]}")
            ax[i].set_title(f"COCO Better Binary, beta 1: {name} ({int(m.sum())})", fontsize=10)
            ax[i].set_xlabel("click")
            ax[i].spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("mean F-beta of the set shown")
        ax[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(a.docs / "fig_coco_guard.png", dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
