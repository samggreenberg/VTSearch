#!/usr/bin/env python3
"""Plot error-vs-K curves from an SOD sweep's ``results.jsonl``.

One figure per ``(dataset, class)``; all configs overlaid on shared axes, each
curve its own color (linestyle = proposal), linear K axis. Two reference overlays
are OFF by default:

  --show-oracle          add each MLP curve's oracle min-over-τ (dashed companion)
  --show-text-baseline   add the text-cosine zero-shot baseline (K=0 horizontal ref)

The underlying ``oracle_cost`` column and the cosine K=0 rows are always present in
``results.jsonl`` regardless of these flags — they only control rendering.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_PROPOSAL_STYLES = {"whole": ":", "sliding": "-", "dino": "--", "hac": "-."}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _config_key(r: dict) -> tuple:
    return (r["embedder"], r["proposal"], r.get("alpha", 0.5), r["head"])


def _mean_by_k(rows: list[dict], field: str) -> dict[int, float]:
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None and v == v:  # skip NaN
            by_k[r["k"]].append(float(v))
    return {k: st.mean(vs) for k, vs in by_k.items() if vs}


def _stats_by_k(rows: list[dict], field: str, band: str) -> dict[int, tuple[float, float, float]]:
    """Per-K ``(mean, lo, hi)`` across seeds. ``band`` = minmax | std | none."""
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None and v == v:  # skip NaN
            by_k[r["k"]].append(float(v))
    out: dict[int, tuple[float, float, float]] = {}
    for k, vs in by_k.items():
        if not vs:
            continue
        m = st.mean(vs)
        if band == "std":
            s = st.pstdev(vs)
            lo, hi = m - s, m + s
        elif band == "minmax":
            lo, hi = min(vs), max(vs)
        else:
            lo, hi = m, m
        out[k] = (m, lo, hi)
    return out


_METRICS = {
    "cost": "weighted FPR+FNR",
    "fpr": "FPR (false-positive rate)",
    "fnr": "FNR (false-negative rate)",
}


def _plot_group(
    rows: list[dict],
    title: str,
    out_path: Path,
    *,
    field: str = "cost",
    band: str = "minmax",
    show_oracle: bool,
    show_text_baseline: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[_config_key(r)].append(r)

    fig, ax = plt.subplots(figsize=(9, 6))
    # One distinct color per curve (tab20 cycles through 20 well-separated hues,
    # so adjacent curves change color frequently).
    cmap = plt.get_cmap("tab20")
    color_i = 0
    for (embedder, proposal, alpha, head), crows in sorted(by_cfg.items()):
        ls = _PROPOSAL_STYLES.get(proposal, "-")

        if head == "cosine":
            # Baseline: only rendered with --show-text-baseline (as a K=0 horiz ref).
            if not show_text_baseline:
                continue
            val_k = _mean_by_k(crows, field)
            if 0 in val_k:
                ax.axhline(
                    val_k[0],
                    color=cmap(color_i % 20),
                    ls=":",
                    lw=1.0,
                    alpha=0.8,
                    label=f"{embedder}/{proposal} text (zero-shot)",
                )
                color_i += 1
            continue

        stats = _stats_by_k(crows, field, band)
        ks = sorted(k for k in stats if k > 0)
        if not ks:
            continue
        color = cmap(color_i % 20)
        color_i += 1
        alpha_tag = f" α{alpha}" if proposal == "hac" else ""
        ax.plot(
            ks,
            [stats[k][0] for k in ks],
            color=color,
            ls=ls,
            lw=1.8,
            marker="o",
            ms=5,
            label=f"{embedder}/{proposal}{alpha_tag}",
        )
        # Seed-variance band (min-max or ±std across --seeds).
        if band != "none":
            ax.fill_between(ks, [stats[k][1] for k in ks], [stats[k][2] for k in ks], color=color, alpha=0.15, lw=0)
        # Oracle companion only makes sense for the combined cost.
        if show_oracle and field == "cost":
            oracle_k = _mean_by_k(crows, "oracle_cost")
            oks = sorted(k for k in oracle_k if k > 0)
            if oks:
                ax.plot(oks, [oracle_k[k] for k in oks], color=color, ls=ls, lw=1.0, alpha=0.35)

    ax.set_xlabel("K (annotation count)")
    ax.set_ylabel(_METRICS.get(field, field))
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_all(
    rows: list[dict],
    out_dir: Path,
    *,
    metrics: list[str] | tuple[str, ...] = ("cost", "fpr", "fnr"),
    band: str = "minmax",
    show_oracle: bool = False,
    show_text_baseline: bool = False,
) -> None:
    """Write one figure per metric per (dataset, class). Reusable by sweep --viz."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["class"])].append(r)
    for (dataset, cls), grows in sorted(groups.items()):
        stem = f"{dataset}_{cls.replace(' ', '_')}"
        for field in metrics:
            out_path = out_dir / f"{stem}_{field}_vs_k.png"
            _plot_group(
                grows,
                f"{dataset}: {cls} — {field} vs K",
                out_path,
                field=field,
                band=band,
                show_oracle=show_oracle,
                show_text_baseline=show_text_baseline,
            )
            print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("docs/experiments/sod-sweep/results.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <results dir>/plots")
    ap.add_argument("--show-oracle", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--show-text-baseline", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--metrics",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=["cost", "fpr", "fnr"],
        help="which metrics to plot (default: cost,fpr,fnr)",
    )
    ap.add_argument(
        "--band-kind",
        choices=("minmax", "std", "none"),
        default="std",
        help="seed-variance band around each curve (default: minmax)",
    )
    args = ap.parse_args()

    rows = _load(args.results)
    if not rows:
        print(f"no rows in {args.results}")
        return 0
    out_dir = args.out_dir or (args.results.parent / "plots")
    render_all(
        rows,
        out_dir,
        metrics=args.metrics,
        band=args.band_kind,
        show_oracle=args.show_oracle,
        show_text_baseline=args.show_text_baseline,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
