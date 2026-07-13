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


def _series_by_seed(rows: list[dict], field: str) -> dict[int, dict[int, float]]:
    """Per-seed ``{k: value}`` series (for the ``all`` band = one distinct line per seed)."""
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for r in rows:
        v = r.get(field)
        if v is not None and v == v:  # skip NaN
            out[int(r["seed"])][int(r["k"])] = float(v)
    return out


def _stats_by_k(rows: list[dict], field: str, band: str) -> dict[int, tuple[float, float, float]]:
    """Per-K ``(mean, lo, hi)`` across seeds. ``band`` = minmax | std | none (``all``
    is handled in the plot: it overlays every seed's own curve instead of a band)."""
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
    "f1": "F1 (at cross-cal threshold)",
    "mean_iou": "mean IoU (top region vs GT)",
    "corloc": "CorLoc@0.5",
}
_DEFAULT_METRICS = ("cost", "fpr", "fnr", "f1", "mean_iou")


def _plot_group(
    rows: list[dict],
    title: str,
    out_path: Path,
    *,
    field: str = "cost",
    band: str = "minmax",
    show_oracle: bool,
    show_text_baseline: bool,
    x_label: str = "K (annotation count)",
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

        alpha_tag = f" α{alpha}" if proposal == "hac" else ""

        if band == "all":
            # One line PER SEED, each its own color + dot markers + a legend entry
            # naming the config and seed, so every individual iteration is distinct.
            for seed, series in sorted(_series_by_seed(crows, field).items()):
                sks = sorted(k for k in series if k > 0)
                if not sks:
                    continue
                ax.plot(
                    sks,
                    [series[k] for k in sks],
                    color=cmap(color_i % 20),
                    ls=ls,
                    lw=1.4,
                    marker="o",
                    ms=4,
                    label=f"{embedder}/{proposal}{alpha_tag} — seed {seed}",
                )
                color_i += 1
            continue

        stats = _stats_by_k(crows, field, band)
        ks = sorted(k for k in stats if k > 0)
        if not ks:
            continue
        color = cmap(color_i % 20)
        color_i += 1
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
        # Seed-variance band (min-max or ±std across seeds).
        if band != "none":
            ax.fill_between(ks, [stats[k][1] for k in ks], [stats[k][2] for k in ks], color=color, alpha=0.15, lw=0)
        # Oracle companion only makes sense for the combined cost.
        if show_oracle and field == "cost":
            oracle_k = _mean_by_k(crows, "oracle_cost")
            oks = sorted(k for k in oracle_k if k > 0)
            if oks:
                ax.plot(oks, [oracle_k[k] for k in oks], color=color, ls=ls, lw=1.0, alpha=0.35)

    ax.set_xlabel(x_label)
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
    metrics: list[str] | tuple[str, ...] = _DEFAULT_METRICS,
    band: str = "minmax",
    show_oracle: bool = False,
    show_text_baseline: bool = False,
    x_label: str = "K (annotation count)",
    x_tag: str = "k",
) -> None:
    """Write one figure per metric per (dataset, class). Reusable by sweep --viz.

    ``x_label``/``x_tag`` let the realistic labeling mode relabel the x-axis to
    "total annotations t" (rows carry ``k == t`` there, so the plotting stays the
    same; only the label and file suffix change).
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["class"])].append(r)
    for (dataset, cls), grows in sorted(groups.items()):
        stem = f"{dataset}_{cls.replace(' ', '_')}"
        for field in metrics:
            out_path = out_dir / f"{stem}_{field}_vs_{x_tag}.png"
            _plot_group(
                grows,
                f"{dataset}: {cls} — {field} vs {x_tag}",
                out_path,
                field=field,
                band=band,
                show_oracle=show_oracle,
                show_text_baseline=show_text_baseline,
                x_label=x_label,
            )
            print(f"wrote {out_path}")


def render_inference_time(timing: list[dict], out_path: Path) -> None:
    """Stacked bar of total time per (embedder × proposal), in seconds.

    ``timing`` is the combined structure from ``sweep._build_total_timing`` — each
    item ``{label, embed_s, compute_s, total_s}``. The bottom segment is the
    embed+propose forward pass (a cache-miss cost, 0 s on a fully-cached run); the
    top segment is the MLP calibrate+fit+score, which runs every sweep across all
    K×seeds — so the chart has data even when embeddings were fully cached. Bars are
    sorted by total time. No-op on an empty list.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not timing:
        return
    timing = sorted(timing, key=lambda t: t["total_s"], reverse=True)
    labels = [t["label"] for t in timing]
    embed = [t["embed_s"] for t in timing]
    compute = [t["compute_s"] for t in timing]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(labels)), 5))
    ax.bar(x, embed, color="#4878c8", label="embed+propose (cache miss)")
    ax.bar(x, compute, bottom=embed, color="#e8843c", label="MLP calibrate+fit+score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("total time (s)")
    ax.set_title("Total time per config (embed + MLP, summed over K×seeds)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
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
        default=list(_DEFAULT_METRICS),
        help="which metrics to plot (default: cost,fpr,fnr,f1,mean_iou)",
    )
    ap.add_argument(
        "--band-kind",
        choices=("minmax", "std", "none", "all"),
        default="std",
        help="per-curve seed spread: minmax/std band, none, or 'all' (overlay every seed's "
        "own curve as a thin line around the mean)",
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
