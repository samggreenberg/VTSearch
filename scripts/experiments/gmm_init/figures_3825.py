#!/usr/bin/env python
"""The #3825 report's figures, from the gate frames.

    python figures_3825.py --analysis <dir>

Four, each answering one question the tables can only answer in numbers.

1. ``tradeoff.png`` - cost against fidelity, one point per arm.  The whole
   decision is a curve, and the two CONTROLS are on it: an arm has to be read
   against what perturbing the incumbent does, not against zero.
2. ``iterations.png`` - the iteration distribution per arm, with the cap drawn.
   The incumbent's mass piled against ``max_iter`` is the issue's opening claim.
3. ``move_ecdf.png`` - the ECDF of the admitted-set move as a fraction of the
   haystack.  A median hides whether an arm is uniformly small or usually zero
   with a bad tail, and those are different risks.
4. ``objective_trace.png`` - the two objectives over EM iterations on real
   folds, which is what makes "the parameter rule crawls a flat ridge" a picture
   rather than an assertion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GATE_INCLUSION = 0
CASE_KEYS = ["cell", "dataset", "embedder", "style", "kind", "case"]

#: Arms drawn as controls rather than candidates - they perturb the incumbent
#: instead of replacing it, and the report's point is where the candidates sit
#: relative to them.
CONTROLS = ("param1e-6", "iter400")


def _paired_moves(cuts: pd.DataFrame) -> pd.DataFrame:
    tcol, acol = f"threshold_i{GATE_INCLUSION}", f"n_admitted_i{GATE_INCLUSION}"
    df = cuts.copy()
    for c in (tcol, acol, "n", "seconds"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    base = df[df.arm == "baseline"][CASE_KEYS + [tcol, acol, "seconds"]].rename(
        columns={tcol: "base_thr", acol: "base_adm", "seconds": "base_seconds"}
    )
    out = df[df.arm != "baseline"].merge(base, on=CASE_KEYS, how="inner", validate="many_to_one")
    out["d_frac"] = (out[acol] - out["base_adm"]).abs() / out["n"]
    out["speedup"] = out["base_seconds"] / out["seconds"].replace(0.0, np.nan)
    return out


def fig_tradeoff(cuts: pd.DataFrame, fits: pd.DataFrame, out: Path) -> None:
    moves = _paired_moves(cuts)
    f = fits.copy()
    for c in ("refit_seconds", "total_seconds"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    base_refit = f[f.arm == "baseline"]["refit_seconds"].sum()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for arm, g in moves.groupby("arm", sort=False):
        speed = base_refit / max(f[f.arm == arm]["refit_seconds"].sum(), 1e-12)
        y = g["d_frac"].quantile(0.9)
        control = arm in CONTROLS
        ax.scatter(
            speed, max(y, 1e-7), s=90, marker="s" if control else "o", color="0.45" if control else "#1f77b4", zorder=3
        )
        ax.annotate(
            arm,
            (speed, max(y, 1e-7)),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
            color="0.35" if control else "black",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("refit speedup over the shipped rule (x, higher is cheaper)")
    ax.set_ylabel("p90 |change in admitted set| / haystack")
    ax.set_title("What each stopping rule buys, and what it moves\n(squares: controls - the incumbent, perturbed)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    plt.close(fig)


def fig_iterations(fits: pd.DataFrame, out: Path) -> None:
    f = fits[fits.provenance == "anchored"].copy()
    f["n_iter"] = pd.to_numeric(f["n_iter"], errors="coerce")
    arms = [a for a in f.arm.unique()]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    data = [f[f.arm == a]["n_iter"].dropna().to_numpy() for a in arms]
    ax.boxplot(data, tick_labels=arms, showfliers=False, whis=(5, 95))
    ax.axhline(200, color="crimson", ls="--", lw=1.2, label="max_iter = 200 (the shipped cap)")
    ax.set_ylabel("EM iterations in the anchored refit")
    ax.set_title("How long each rule runs (box: quartiles, whiskers: 5-95th)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "iterations.png", dpi=150)
    plt.close(fig)


def fig_move_ecdf(cuts: pd.DataFrame, out: Path) -> None:
    moves = _paired_moves(cuts)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for arm, g in moves.groupby("arm", sort=False):
        v = np.sort(g["d_frac"].dropna().to_numpy())
        if v.size == 0:
            continue
        ax.plot(
            np.maximum(v, 1e-7),
            np.arange(1, v.size + 1) / v.size,
            lw=1.6,
            ls="--" if arm in CONTROLS else "-",
            color="0.5" if arm in CONTROLS else None,
            label=arm,
        )
    ax.set_xscale("log")
    ax.set_xlabel("|change in admitted set| / haystack  (1e-7 = no change)")
    ax.set_ylabel("fraction of cases at or below")
    ax.set_title("How much of the haystack each rule moves")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "move_ecdf.png", dpi=150)
    plt.close(fig)


def fig_trace(trace: pd.DataFrame, out: Path) -> None:
    t = trace.copy()
    t["value"] = pd.to_numeric(t["value"], errors="coerce")
    keys = list(t.groupby(["cell", "case", "fold"]).groups)[:4]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        g = t[(t.cell == key[0]) & (t.case == key[1]) & (t.fold == key[2])]
        for objective, gg in g.groupby("objective", sort=False):
            v = gg.sort_values("iteration")["value"].to_numpy(dtype=float)
            if v.size < 2:
                continue
            # Distance from the value the trace converges to: a flat ridge is
            # invisible on the raw scale and obvious on this one.
            ax.semilogy(np.arange(1, v.size + 1), np.maximum(np.abs(v[-1] - v), 1e-16), lw=1.4, label=objective)
        ax.set_xlabel("EM iteration")
        ax.set_title(f"{key[0]}:{key[1]}", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("|objective - its own limit| (nats)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("The flat ridge: what another iteration is still buying", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "objective_trace.png", dpi=150)
    plt.close(fig)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", default=None, help="default: <analysis>/figures")
    args = ap.parse_args(list(argv) if argv is not None else None)

    analysis = Path(args.analysis)
    out = Path(args.out) if args.out else analysis / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_csv(analysis / "gate3825_cuts.csv")
    fits = pd.read_csv(analysis / "gate3825_fits.csv")
    fig_tradeoff(cuts, fits, out)
    fig_iterations(fits, out)
    fig_move_ecdf(cuts, out)
    trace_path = analysis / "gate3825_trace.csv"
    if trace_path.exists() and trace_path.stat().st_size > 0:
        trace = pd.read_csv(trace_path)
        if not trace.empty:
            fig_trace(trace, out)
    print(f"wrote {out}/ ({', '.join(sorted(p.name for p in out.glob('*.png')))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
