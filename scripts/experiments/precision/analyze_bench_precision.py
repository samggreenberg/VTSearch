"""Paired benchmark analysis across precision arms (issue #3143, part 3).

    python analyze_bench_precision.py --arms fp32_l40s,fp16_l40s --root <bench root>

The question is a **null**: does the shipped decision metric move by *less* than
the 0.005 the calibration studies resolve?  A null needs more care than an
effect, so three things are done deliberately:

1. **Pair, then cluster.**  Cells are matched on
   ``(dataset, embedder, category, seed, style, t)`` — identical splits, identical
   exemplar, only the vectors differ.  But steps *within* a cell are strongly
   autocorrelated, so a step-level SE would be anti-conservative by roughly the
   square root of the steps per cell.  Each cell is collapsed to its own mean
   first and the SE is taken **over cells**.  Getting this backwards is how a
   trend claim on these curves gets called significant when it is not (#2825).
2. **Report the resolution, not just the point.**  The verdict distinguishes
   "resolvably smaller than the margin" (|diff| + 2·SE < margin) from "cannot
   resolve at this cell count" — an underpowered null is not evidence of no
   effect, and saying so is more useful than implying the question is settled.
3. **Say how often the decision was literally identical.**  The most legible
   statement available: on what fraction of steps did the two precisions pick
   the same threshold, the same cut, the same items.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import precision_config as pcfg  # noqa: E402

#: The metrics a ship decision actually reads.  Split by whether the 0.005
#: decision margin even *applies*: it is a margin on error RATES in [0, 1].
#: Comparing it to a count of positives is a category error — the first run of
#: this analysis duly reported "n_good EXCEEDS MARGIN" for a difference of 0.87
#: positives, which says nothing about 0.005 and reads as if it did.
RATE_METRICS = [
    "cost",
    "regret",
    "average_precision",
    "fnr",
    "fpr",
    "rule_inefficiency",
    "calibration_shift",
]
#: Reported with the same paired ±SE, but never against the rate margin.
COUNT_METRICS = ["n_good"]
METRICS = [*RATE_METRICS, *COUNT_METRICS]

CELL_KEY = ["dataset", "embedder", "category", "seed", "style"]
PAIR_KEY = [*CELL_KEY, "t"]


def log(msg: str) -> None:
    print(msg, flush=True)


def sig2(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.2g}" if abs(x) >= 0.001 else f"{x:.1e}"


def pm(mean: float, se: float) -> str:
    return f"{sig2(mean)} ± {sig2(se)}"


def load_arm(root: Path, arm: str) -> tuple[pd.DataFrame, dict]:
    cells = root / arm / "results" / "cells"
    files = sorted(f for f in cells.glob("task_*.csv") if not any(k in f.name for k in ("sweep", "cutdiag", "cutincl")))
    frames, bad = [], []
    for f in files:
        if f.stat().st_size == 0:
            bad.append((f.name, "zero-byte"))
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                bad.append((f.name, "header only"))
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            bad.append((f.name, repr(exc)[:60]))
    prov = {"arm": arm, "files": len(files), "unreadable": bad}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--deep-from", type=int, default=100, help="first vote of the deep regime")
    ap.add_argument("--margin", type=float, default=pcfg.DECISION_MARGIN)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    arms = [a for a in args.arms.split(",") if a]
    root = Path(args.root)
    if len(arms) < 2:
        raise SystemExit("need at least two arms to compare")
    ref, *others = arms

    # --- coverage, stated before any comparison ---------------------------
    log("=" * 96)
    log("COVERAGE — everything dropped is counted, not silently excluded")
    log("=" * 96)
    data: dict[str, pd.DataFrame] = {}
    for arm in arms:
        df, prov = load_arm(root, arm)
        data[arm] = df
        n_cells = df.groupby(CELL_KEY).ngroups if not df.empty else 0
        log(
            f"  {arm:16s} {prov['files']:4d} cell files, {len(df):6d} rows, {n_cells:4d} distinct cells, "
            f"{len(prov['unreadable'])} unusable {prov['unreadable'][:3]}"
        )
    if any(df.empty for df in data.values()):
        raise SystemExit("at least one arm has no rows; nothing to compare")

    ref_df = data[ref]
    for arm in others:
        arm_df = data[arm]
        log("")
        log("=" * 96)
        log(f"{arm}  vs  {ref}   (paired; margin = {args.margin})")
        log("=" * 96)

        merged = ref_df.merge(arm_df, on=PAIR_KEY, suffixes=("_ref", "_arm"))
        ref_cells = ref_df.groupby(CELL_KEY).ngroups
        arm_cells = arm_df.groupby(CELL_KEY).ngroups
        paired_cells = merged.groupby(CELL_KEY).ngroups
        log(f"paired steps: {len(merged)}  paired cells: {paired_cells}  (ref had {ref_cells}, arm had {arm_cells})")
        if paired_cells < min(ref_cells, arm_cells):
            log(f"  ** {min(ref_cells, arm_cells) - paired_cells} cell(s) failed to pair and are EXCLUDED **")
        if merged.empty:
            log("  nothing paired — the arms do not share cells; check verify_pairing.py")
            continue

        # --- identical-decision rate, the most legible statement ----------
        for col in ("cost", "average_precision"):
            same = np.isclose(merged[f"{col}_ref"], merged[f"{col}_arm"], rtol=0, atol=0)
            log(f"steps where {col} is bit-identical: {same.mean() * 100:.1f}%  ({int(same.sum())}/{len(merged)})")
        if "threshold_provenance_ref" in merged:
            same_prov = merged["threshold_provenance_ref"] == merged["threshold_provenance_arm"]
            log(f"steps choosing the threshold the same way: {same_prov.mean() * 100:.1f}%")

        # --- paired differences, clustered by cell ------------------------
        deep = merged[merged["t"] >= args.deep_from]
        log("")
        log(f"Deep regime (t >= {args.deep_from}), paired difference per CELL then averaged.")
        log("SE is over cells, not steps: steps within a cell are autocorrelated and a")
        log("step-level SE would be anti-conservative (#2825).")
        log("")
        log(
            f"{'metric':22s} {'ref mean':>10s} {'arm mean':>10s} {'paired diff ± SE':>24s} {'cells':>6s} {'verdict':>28s}"
        )
        rows = []
        for metric in METRICS:
            rc, ac = f"{metric}_ref", f"{metric}_arm"
            if rc not in deep or deep[rc].isna().all():
                log(f"{metric:22s} {'—':>10s} {'—':>10s} {'not emitted':>24s}")
                continue
            sub = deep.dropna(subset=[rc, ac])
            if sub.empty:
                log(f"{metric:22s} {'—':>10s} {'—':>10s} {'no non-null rows':>24s}")
                continue
            per_cell = sub.groupby(CELL_KEY).apply(
                lambda g, rc=rc, ac=ac: pd.Series(
                    {"ref": g[rc].mean(), "arm": g[ac].mean(), "diff": (g[ac] - g[rc]).mean()}
                ),
                include_groups=False,
            )
            n = len(per_cell)
            mean = float(per_cell["diff"].mean())
            se = float(per_cell["diff"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            # A null is only a null if it is resolvable.
            if metric in COUNT_METRICS:
                # A count is not a rate: the margin does not apply to it.  Say
                # whether the difference is resolvable at all instead.
                if not np.isfinite(se):
                    verdict = "count — unresolvable"
                elif abs(mean) > 2 * se:
                    verdict = f"count: {abs(mean) / se:.1f} SE from zero"
                else:
                    verdict = "count: within noise"
            elif not np.isfinite(se):
                verdict = "single cell — unresolvable"
            elif abs(mean) + 2 * se < args.margin:
                verdict = "below margin (resolved)"
            elif abs(mean) - 2 * se > args.margin:
                verdict = "** EXCEEDS MARGIN **"
            else:
                verdict = "cannot resolve at this n"
            log(
                f"{metric:22s} {sig2(per_cell['ref'].mean()):>10s} {sig2(per_cell['arm'].mean()):>10s} "
                f"{pm(mean, se):>24s} {n:6d} {verdict:>28s}"
            )
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "ref": per_cell["ref"].mean(),
                    "value": per_cell["arm"].mean(),
                    "diff": mean,
                    "se": se,
                    "n_cells": n,
                    "verdict": verdict,
                }
            )

        # --- per embedder, because a pooled null can hide one arm ---------
        log("")
        log("Same difference, split by embedder (a pooled average across a crossover")
        log("is precisely the number that hides it):")
        for emb, g in deep.groupby("embedder"):
            parts = []
            for metric in ("cost", "regret", "average_precision"):
                rc, ac = f"{metric}_ref", f"{metric}_arm"
                if rc not in g:
                    continue
                per_cell = g.groupby(CELL_KEY).apply(
                    lambda x, rc=rc, ac=ac: (x[ac] - x[rc]).mean(), include_groups=False
                )
                se = per_cell.std(ddof=1) / np.sqrt(len(per_cell)) if len(per_cell) > 1 else float("nan")
                parts.append(f"{metric} {pm(float(per_cell.mean()), float(se))}")
            log(f"  {emb:14s} ({g.groupby(CELL_KEY).ngroups:3d} cells)  " + "   ".join(parts))

        # If the margin could not be resolved, say what it would take.  "We could
        # not resolve it" is not "it is not there", and a reader deciding whether
        # to buy more cells needs the number, not the verdict alone.
        unresolved = [r for r in rows if r["metric"] in RATE_METRICS and r["verdict"] == "cannot resolve at this n"]
        if unresolved:
            log("")
            log(f"Power: {len(unresolved)} rate metric(s) could not be resolved against the {args.margin} margin.")
            log("Cells needed for 2*SE < margin, at this between-cell spread:")
            for r in unresolved:
                if not np.isfinite(r["se"]) or r["se"] <= 0:
                    continue
                # SE scales as 1/sqrt(n); solve 2*SE*sqrt(n_have/n_need) = margin.
                need = r["n_cells"] * (2 * r["se"] / args.margin) ** 2
                log(f"  {r['metric']:22s} have {r['n_cells']:4d} cells (SE {sig2(r['se'])}) -> need ~{need:.0f}")
            log("A 3e-6 vector change reroutes the vote sequence, so the trajectory")
            log("decorrelates and the per-cell spread is set by that, not by the")
            log("perturbation size. This test bounds a SYSTEMATIC effect; it cannot be")
            log("made tight cheaply.")

        out = root / f"paired_{arm}_vs_{ref}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        log(f"\nwrote {out}")

        if not args.no_figures:
            _figures(merged, ref, arm, root / "figures", args)

    return 0


def _figures(merged: pd.DataFrame, ref: str, arm: str, outdir: Path, args) -> None:
    """Averaged trajectory *and* per-cell traces — the mean alone hides which
    cell owns a divergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    for metric in ("cost", "average_precision"):
        rc, ac = f"{metric}_ref", f"{metric}_arm"
        if rc not in merged:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

        # (a) mean trajectory, both arms
        by_t = merged.groupby("t")[[rc, ac]].mean()
        axes[0].plot(by_t.index, by_t[rc], label=ref, lw=1.4)
        axes[0].plot(by_t.index, by_t[ac], label=arm, lw=1.4, ls="--")
        axes[0].set_xlabel("votes spent (t)")
        axes[0].set_ylabel(metric)
        axes[0].set_title(f"{metric}: mean trajectory")
        axes[0].legend(fontsize=7)

        # (b) every cell's paired difference, one line each
        for _, g in merged.groupby(CELL_KEY):
            g = g.sort_values("t")
            axes[1].plot(g["t"], g[ac] - g[rc], lw=0.6, alpha=0.35, color="#2b6cb0")
        diff_by_t = merged.groupby("t").apply(lambda g, rc=rc, ac=ac: (g[ac] - g[rc]).mean(), include_groups=False)
        axes[1].plot(diff_by_t.index, diff_by_t.values, color="k", lw=1.8, label="mean")
        axes[1].axhline(0, color="#999", lw=1, ls=":")
        for sign in (1, -1):
            axes[1].axhline(
                sign * args.margin, color="#c53030", lw=1, ls="--", label="decision margin" if sign == 1 else None
            )
        axes[1].set_xlabel("votes spent (t)")
        axes[1].set_ylabel(f"Δ{metric} ({arm} − {ref})")
        axes[1].set_title("per-cell paired difference (thin = one cell)")
        axes[1].legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(outdir / f"paired_{metric}_{arm}_vs_{ref}.png", dpi=150)
        plt.close(fig)
    log(f"wrote figures to {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
