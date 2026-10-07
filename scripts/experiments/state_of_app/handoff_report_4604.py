"""#4604: roll the six hand-off pricings (path x preset) into the report's tables and figures.

Reads ``handoff_price_4604.py``'s outputs (``summary_*``, ``curves_*``, ``runs_*``) and the extracted steps
from ``--dir``; prints every table the report quotes and writes ``figures/`` and ``tables/`` under ``--docs``.

    python handoff_report_4604.py --dir <pricing dir> --docs docs/experiments/2026-10-07-opening-handoff-4604 \\
        --baseline binary=<text_baseline.csv> --baseline region=<text_baseline.csv> \\
        --ap binary=<pre-#4605 analysis dir>:<#4605 analysis dir> --ap region=...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAGS = [f"{p}_{b}" for p in ("binary", "region") for b in ("b025", "b1", "b4")]
BETA = {"b025": "1/4", "b1": "1", "b4": "4"}
#: The named rules: today, the two simple hand-offs, the vote rule, and the ceilings.
NAMED = {
    "today": "today: the typed query's set until Hard",
    "detector_always": "the detector from its first click",
    "after_bad": "the detector from the end of the Bad phase",
    "after_bad_votes_m+0.0_p1": "vote rule: from the end of the Bad phase, once the detector's set beats the typed query's on the votes",
    "ceiling_opening": "ceiling: the better set per run and click, in the opening",
    "ceiling_any": "ceiling at every click",
}
VOTE = "after_bad_votes_m+0.0_p1"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def title(tag: str) -> str:
    p, b = tag.split("_")
    return f"{'Binary' if p == 'binary' else 'Region'} Photo, beta {BETA[b]}"


def half(cls: str) -> int:
    return int(hashlib.md5(cls.encode()).hexdigest(), 16) % 2


def headline(summ: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tag, s in summ.items():
        s = s.set_index("rule")
        for rule in NAMED:
            r = s.loc[rule]
            rows.append({"cell": tag, "rule": rule, "mean 1-50": r.auc50, "mean 1-150": r.auc150,
                         **{f"@{t}": r[f"at_{t}"] for t in (10, 25, 50, 100)},
                         "gain 1-50": r.gain50, "se 1-50": r.gain50_se,
                         "gain 1-150": r.gain150, "se 1-150": r.gain150_se,
                         "share of runs worse": r.worse_runs})  # fmt: skip
    return pd.DataFrame(rows)


def split_half(summ: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per family: the param picked on one class half (md5 of the class), scored on the other, both ways."""
    rows = []
    for tag, s in summ.items():
        for fam in ("click", "goods", "votes", "after_bad_votes"):
            f = s[s["family"] == fam]
            a = f.loc[f["gain150_h0"].idxmax()]
            b = f.loc[f["gain150_h1"].idxmax()]
            rows.append({"cell": tag, "family": fam, "picked on half 0": a.rule, "scored on half 1": a.gain150_h1,
                         "picked on half 1": b.rule, "scored on half 0": b.gain150_h0,
                         "held out": (a.gain150_h1 + b.gain150_h0) / 2, "best in-sample": f["gain150"].max()})  # fmt: skip
    return pd.DataFrame(rows)


def uniform(summ: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every rule applied unchanged at all six cells: gain over clicks 1-50."""
    g = pd.concat([s.set_index("rule")["gain50"].rename(tag) for tag, s in summ.items()], axis=1).dropna()
    g["mean"] = g[TAGS].mean(axis=1)
    g["min"] = g[TAGS].min(axis=1)
    return g


def mechanism(d: Path) -> pd.DataFrame:
    """The early detector's returned set on the withheld half (runs with a detector at the click)."""
    rows = []
    for p in ("binary", "region"):
        for b in ("b025", "b1", "b4"):
            st = pd.read_csv(d / f"steps_{p}_{b}.csv.gz")
            st = st.assign(returned=st["recall"] * st["n_test_pos"] + st["fpr"] * st["n_test_neg"])
            for t in (7, 10, 25):
                g = st[st["t"] == t]
                rows.append({"path": p, "beta": BETA[b], "click": t, "runs": len(g), "precision": g["precision"].mean(),
                             "recall": g["recall"].mean(), "returned, median": g["returned"].median()})  # fmt: skip
    return pd.DataFrame(rows)


def typed_query_set(baselines: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for p, f in baselines.items():
        b = pd.read_csv(f)
        b = b[b["embedder"] == "siglip"]
        k = b["text_recall"] * b["n_test_pos"] + b["text_fpr"] * (b["n_test"] - b["n_test_pos"])
        rows.append({"path": p, "rows": len(b), "precision": b["text_precision"].mean(), "recall": b["text_recall"].mean(),
                     "returned, median": k.median(), "AP": b["text_AP"].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def ap_table(ap: dict[str, tuple[Path, Path]]) -> pd.DataFrame:
    """The opening's detector as a ranking: AP on the withheld half against the typed query's, by click.

    Read off a pre-#4605 analysis (its ``curves.csv`` carries the detector's AP from the first detector, and the
    typed query's before it), over the trained runs of the #4605 analysis.
    """
    rows = []
    for p, (old, new) in ap.items():
        c = pd.read_csv(old / "curves.csv", usecols=["category", "seed", "t", "ap"])
        cells = pd.read_csv(new / "cells.csv")
        cells = cells.loc[~cells["never_trained"].astype(bool), ["category", "seed", "text_ap"]]
        c = c.merge(cells, on=["category", "seed"])
        row: dict[str, object] = {"path": p, "runs": len(cells), "typed query AP": cells["text_ap"].mean()}
        for t in (4, 7, 10, 25, 40, 60):
            g = c[c["t"] == t]
            row[f"AP@{t}"] = g["ap"].mean()
            row[f"beats the typed query @{t}"] = (g["ap"] > g["text_ap"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile(runs: dict) -> pd.DataFrame:
    """The issue's first table was the hidden detector's mean over runs that HAVE one at the click; a rule can
    only show the typed query's set to the runs that do not."""
    rows = []
    for tag, r in runs.items():
        det, tq = r["det"], r["tq"]
        for t in (10, 25, 50):
            have = np.isfinite(det[:, t])
            rows.append({"cell": tag, "click": t, "share with a detector": have.mean(),
                         "detector, runs that have one": det[have, t].mean(),
                         "detector, else the typed query": np.where(have, det[:, t], tq).mean()})  # fmt: skip
    return pd.DataFrame(rows)


def after_bad_gains(runs: dict) -> dict[str, np.ndarray]:
    """Per run, the mean gain over clicks 1-50 of the detector from the end of the Bad phase, against today."""
    out = {}
    for tag, r in runs.items():
        det, tq, shown, more = r["det"], r["tq"], r["shown"], r["more"]
        t = np.arange(det.shape[1])
        new = np.where((t[None, :] >= np.minimum(more, shown)[:, None]) & np.isfinite(det), det, tq[:, None])
        old = np.where((t[None, :] >= shown[:, None]) & np.isfinite(det), det, tq[:, None])
        out[tag] = (new[:, 1:51] - old[:, 1:51]).mean(axis=1)
    return out


def by_band(runs: dict, gains: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for tag, r in runs.items():
        g = pd.DataFrame({"band": r["band"], "cls": r["cls"], "gain": gains[tag]})
        for band, x in g.groupby("band"):
            per_cls = x.groupby("cls")["gain"].mean()
            rows.append({"cell": tag, "band": band, "runs": len(x), "gain 1-50": x["gain"].mean(),
                         "se": per_cls.std(ddof=1) / np.sqrt(len(per_cls)), "share worse": (x["gain"] < 0).mean()})  # fmt: skip
    return pd.DataFrame(rows)


def unchanged(runs: dict, gains: dict[str, np.ndarray]) -> pd.DataFrame:
    """The runs no rule changes in clicks 1-50: their Bad phase never ends before Hard."""
    rows = []
    for tag, r in runs.items():
        z = np.abs(gains[tag]) < 1e-12
        first = np.where(np.isfinite(r["det"]).any(1), np.isfinite(r["det"]).argmax(1), np.inf)
        rows.append({"cell": tag, "share unchanged": z.mean(), "of those, Bad phase never ends before Hard": np.mean(r["more"][z] >= r["shown"][z]),
                     "first detector, median (unchanged)": np.median(first[z]), "first detector, median (others)": np.median(first[~z])})  # fmt: skip
    return pd.DataFrame(rows)


def examples(d: Path, runs: dict, gains: dict[str, np.ndarray], baselines: dict[str, Path]) -> list[str]:
    """Two literal runs at the median of their side: a region gain at beta 1/4 and a binary loss at beta 4."""
    out = []
    for tag, want in (("region_b025", "gain"), ("binary_b4", "loss")):
        r, g = runs[tag], gains[tag]
        pool = np.where(g > 0)[0] if want == "gain" else np.where(g < 0)[0]
        i = pool[np.argsort(g[pool])[len(pool) // 2]]
        cat, seed = str(r["category"][i]), int(r["seed"][i])
        st = pd.read_csv(d / f"steps_{tag}.csv.gz")
        st = st[(st["category"] == cat) & (st["seed"] == seed)].set_index("t")
        base = pd.read_csv(baselines[tag.split("_")[0]])
        b = base[(base["embedder"] == "siglip") & (base["category"] == cat) & (base["seed"] == seed)].iloc[0]
        k = b["text_recall"] * b["n_test_pos"] + b["text_fpr"] * (b["n_test"] - b["n_test_pos"])
        out.append(
            f"{title(tag)}, {cat} seed {seed}: typed query F {r['tq'][i]:.2f} (precision {b['text_precision']:.2f}, "
            f"recall {b['text_recall']:.2f}, {k:.0f} returned of {int(b['n_test'])}, {int(b['n_test_pos'])} positive); "
            f"Bad phase ends at click {r['more'][i]:.0f}, Hard at {r['shown'][i]:.0f}; gain over clicks 1-50 {g[i]:+.2f}"
        )
        for c in sorted({int(r["more"][i]), 10, 15, int(r["shown"][i])}):
            if c in st.index:
                x = st.loc[c]
                k = x["recall"] * x["n_test_pos"] + x["fpr"] * x["n_test_neg"]
                out.append(f"  click {c}: detector F {r['det'][i, c]:.2f} (precision {x['precision']:.2f}, recall "
                           f"{x['recall']:.2f}, {k:.0f} returned; {int(x['n_good'])} Goods, {int(x['n_bad'])} Bads)")  # fmt: skip
    return out


def _axes_style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def figure_curves(cur: dict[str, pd.DataFrame], out: Path) -> None:
    lines = [
        ("ceiling_opening", "#1baf7a", "--"),
        ("today", "#8a8985", "-"),
        ("after_bad", "#2a78d6", "-"),
        (VOTE, "#eb6834", "-"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
    for ax, tag in zip(axes.flat, TAGS):
        c = cur[tag]
        for k, col, ls in lines:
            ax.plot(c["t"], c[k], color=col, lw=1.5 if ls == "--" else 2, ls=ls, label=NAMED[k])
        ax.set_xlim(0, 100)
        ax.set_title(title(tag), color=INK, fontsize=11, loc="left")
        _axes_style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("F-beta of the returned set (withheld half)", color=MUTED, fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)


def figure_runs(gains: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True)
    for ax, tag in zip(axes.flat, TAGS):
        g = np.sort(gains[tag])
        x = np.linspace(0, 1, len(g))
        ax.fill_between(x, 0, g, where=g >= 0, color="#2a78d6", lw=0)
        ax.fill_between(x, 0, g, where=g < 0, color="#eb6834", lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(
            f"{title(tag)}: mean {g.mean():+.2f}, {(g < 0).mean():.0%} of runs lose", color=INK, fontsize=10, loc="left"
        )
        _axes_style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("gain over clicks 1-50, per run", color=MUTED, fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("runs, sorted by gain", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def _pairs(specs: list[str]) -> dict[str, str]:
    return dict(s.split("=", 1) for s in specs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, type=Path, help="handoff_price_4604.py's output dir")
    ap.add_argument("--docs", type=Path, help="the report dir: figures/ and tables/ are written under it")
    ap.add_argument("--baseline", action="append", required=True, help="path=text baseline the pricing used")
    ap.add_argument("--ap", action="append", default=[], help="path=<pre-#4605 analysis dir>:<#4605 analysis dir>")
    a = ap.parse_args()
    baselines = {k: Path(v) for k, v in _pairs(a.baseline).items()}
    aps = {k: tuple(Path(x) for x in v.split(":")) for k, v in _pairs(a.ap).items()}
    summ = {t: pd.read_csv(a.dir / f"summary_{t}.csv") for t in TAGS}
    cur = {t: pd.read_csv(a.dir / f"curves_{t}.csv") for t in TAGS}
    runs = {t: np.load(a.dir / f"runs_{t}.npz", allow_pickle=True) for t in TAGS}
    gains = after_bad_gains(runs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 30)
    tables = {
        "headline": headline(summ),
        "split_half": split_half(summ),
        "typed_query_set": typed_query_set(baselines),
        "returned_set": mechanism(a.dir),
        "reconcile": reconcile(runs),
        "by_band": by_band(runs, gains),
        "unchanged": unchanged(runs, gains),
    }
    if aps:
        tables["ap"] = ap_table(aps)  # type: ignore[arg-type]
    for name, t in tables.items():
        print(f"\n== {name}\n{t.round(3).to_string(index=False)}")
    u = uniform(summ)
    print(
        "\n== uniform rules (gain 1-50), top 12 by mean\n",
        u.sort_values("mean", ascending=False).head(12).round(3).to_string(),
    )
    print("\n== examples\n" + "\n".join(examples(a.dir, runs, gains, baselines)))
    if a.docs:
        (a.docs / "figures").mkdir(parents=True, exist_ok=True)
        (a.docs / "tables").mkdir(parents=True, exist_ok=True)
        figure_curves(cur, a.docs / "figures" / "handoff_over_clicks.png")
        figure_runs(gains, a.docs / "figures" / "run_gains.png")
        for name in ("headline", "split_half", "by_band", "returned_set", "ap", "reconcile"):
            if name in tables:
                tables[name].round(4).to_csv(a.docs / "tables" / f"{name}.csv", index=False)
        pd.concat(summ.values()).round(5).to_csv(a.docs / "tables" / "rules.csv", index=False)
        keep = ["t", *NAMED]
        pd.concat([cur[t][keep].assign(cell=t) for t in TAGS]).round(4).to_csv(
            a.docs / "tables" / "curves.csv", index=False
        )
        (a.docs / "tables" / "provenance.json").write_text(
            json.dumps(
                {
                    "pricing_dir": str(a.dir),
                    "baselines": {k: str(v) for k, v in baselines.items()},
                    "ap": {k: [str(x) for x in v] for k, v in aps.items()},
                },
                indent=1,
            )
            + "\n"  # fmt: skip
        )
        print(f"\nfigures and tables -> {a.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
