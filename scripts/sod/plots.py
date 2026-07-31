#!/usr/bin/env python3
"""Plot error-vs-K curves from an SOD sweep's ``results.jsonl``.

One figure per ``(dataset, class)``; all configs overlaid on shared axes, each
curve its own color; solid = the calibrated (actual) value, dashed = the oracle
companion (with --show-oracle). Linear K axis. Two reference overlays are OFF by default:

  --show-oracle          add the oracle companion (dashed) on the cost + F1 plots:
                         oracle_cost = min achievable cost, oracle_f1 = max achievable
                         F1 (both true bounds the calibrated curve can't cross)
  --show-text-baseline   add the text-cosine zero-shot baseline (K=0 horizontal ref)

The underlying ``oracle_cost`` / ``oracle_f1`` columns and the cosine K=0 rows are
always present in ``results.jsonl`` regardless of these flags — they only control
rendering. fpr/fnr have no oracle (their per-metric optimum is degenerate).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _config_key(r: dict) -> tuple:
    # Include the hac sweep axes (leaf seeding/assignment + merge-order PCA) so each
    # variant is its own curve instead of collapsing into one averaged line. Old
    # results without these keys fall back to the production baseline values.
    return (
        r["embedder"],
        r["proposal"],
        r.get("alpha", 0.5),
        r["head"],
        r.get("leaf_seeding", "topk"),
        r.get("leaf_assign", "spatial"),
        r.get("pca_dims"),  # None = full-dim baseline
        r.get("resolution"),  # None = checkpoint default (224)
        r.get("hac_k"),  # HAC leaf count (distinct from the `k`=annotation-count column)
        r.get("leaf_beta"),  # feature-assignment blend; None = reused alpha
        r.get("dinov3_model"),  # DINOv3 checkpoint size; None = app default (vitb16)
    )


def _config_label(key: tuple) -> str:
    """Legend label for a config key. For ``hac`` it appends the leaf/PCA variant
    (``… α0.5 · spread/feature · pca10``) so the ablation renders as distinct,
    self-labelling curves; the input resolution is appended when non-default.
    Non-hac proposals stay ``embedder/proposal`` (+ resolution tag)."""
    embedder, proposal, alpha, _head = key[0], key[1], key[2], key[3]
    seeding, leaf_assign, pca_dims = key[4], key[5], key[6]
    resolution = key[7] if len(key) > 7 else None
    hac_k = key[8] if len(key) > 8 else None
    leaf_beta = key[9] if len(key) > 9 else None
    dinov3_model = key[10] if len(key) > 10 else None
    res_tag = "" if resolution in (None, 0) else f" · r{resolution}"
    # Model tag only when a non-default DINOv3 checkpoint was used (vitb16 is the default).
    model_tag = "" if dinov3_model in (None, "vitb16") else f" · {dinov3_model}"
    if proposal != "hac":
        return f"{embedder}/{proposal}{res_tag}{model_tag}"
    k_tag = "" if hac_k is None else f" k{hac_k}"
    # β shown only when it decouples from α (feature + explicit value).
    beta_tag = f" β{leaf_beta}" if (leaf_assign == "feature" and leaf_beta is not None) else ""
    pca_s = "none" if pca_dims in (None, 0) else str(pca_dims)
    return f"{embedder}/{proposal}{k_tag} α{alpha} · {seeding}/{leaf_assign}{beta_tag} · pca{pca_s}{res_tag}{model_tag}"


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


def _class_series(rows: list[dict], field: str) -> dict[str, dict[int, float]]:
    """Per-class ``{k: mean-over-seeds}`` for one config's rows (the per-class analogue of
    ``_series_by_seed``; used by the summary ``all`` band = one line per class)."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[str(r["class"])].append(r)
    return {cls: _mean_by_k(crows, field) for cls, crows in by_class.items()}


def _summary_stats(rows: list[dict], field: str, band: str) -> dict[int, tuple[float, float, float]]:
    """Per-K ``(macro_mean, lo, hi)`` **across classes** for one config. Reduces each class to its
    mean-over-seeds first (so classes weight equally), then aggregates over classes; ``lo/hi`` is the
    across-class ``std``/``minmax`` spread (how consistent the config is across classes)."""
    per_class = _class_series(rows, field)
    by_k: dict[int, list[float]] = defaultdict(list)
    for series in per_class.values():
        for k, v in series.items():
            by_k[k].append(v)  # one value (that class's seed-mean) per class at k
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
# Metrics that have an oracle companion in the rows, and the field it lives under.
# cost → min achievable cost; f1 → max achievable F1 (both true bounds the calibrated
# curve can't cross). fpr/fnr have no oracle (their per-metric optimum is degenerate —
# only their sum, cost, has a meaningful oracle); mean_iou/corloc aren't threshold-based.
_ORACLE_FIELD = {"cost": "oracle_cost", "f1": "oracle_f1"}


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
    # One distinct color per curve. tab10 = 10 well-separated hues; unlike tab20 it
    # doesn't pair each hue with a light/dark twin, so adjacent curves read as clearly
    # different colours instead of dark-blue-then-light-blue shades.
    cmap = plt.get_cmap("tab10")
    color_i = 0
    oracle_drawn = False
    for key, crows in sorted(by_cfg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        embedder, proposal, head = key[0], key[1], key[3]

        if head == "cosine":
            # Baseline: only rendered with --show-text-baseline (as a K=0 horiz ref).
            if not show_text_baseline:
                continue
            val_k = _mean_by_k(crows, field)
            if 0 in val_k:
                ax.axhline(
                    val_k[0],
                    color=cmap(color_i % 10),
                    ls=":",
                    lw=1.0,
                    alpha=0.8,
                    label=f"{embedder}/{proposal} text (zero-shot)",
                )
                color_i += 1
            continue

        cfg_label = _config_label(key)

        if band == "all":
            # One line PER SEED, each its own color + a legend entry naming the
            # config and seed, so every individual iteration is distinct.
            _ofield = _ORACLE_FIELD.get(field)
            oracle_series = _series_by_seed(crows, _ofield) if (show_oracle and _ofield) else {}
            for seed, series in sorted(_series_by_seed(crows, field).items()):
                sks = sorted(k for k in series if k > 0)
                if not sks:
                    continue
                seed_color = cmap(color_i % 10)
                ax.plot(
                    sks,
                    [series[k] for k in sks],
                    color=seed_color,
                    ls="-",
                    lw=1.4,
                    label=f"{cfg_label} — seed {seed}",
                )
                # This seed's oracle floor (best-τ on the same scores), dashed +
                # faint in the same colour, so each bouncing cost line shows the flat
                # achievable floor beneath it.
                oks = sorted(k for k in oracle_series.get(seed, {}) if k > 0)
                if oks:
                    ax.plot(oks, [oracle_series[seed][k] for k in oks], color=seed_color, ls="--", lw=1.0, alpha=0.5)
                    oracle_drawn = True
                color_i += 1
            continue

        stats = _stats_by_k(crows, field, band)
        ks = sorted(k for k in stats if k > 0)
        if not ks:
            continue
        color = cmap(color_i % 10)
        color_i += 1
        ax.plot(
            ks,
            [stats[k][0] for k in ks],
            color=color,
            ls="-",
            lw=1.8,
            label=cfg_label,
        )
        # Seed-variance band (min-max or ±std across seeds).
        if band != "none":
            ax.fill_between(ks, [stats[k][1] for k in ks], [stats[k][2] for k in ks], color=color, alpha=0.15, lw=0)
        # Oracle companion (this metric at the best-τ / cost-optimal operating point):
        # a faint DASHED line in the curve's colour. The gap to the solid calibrated
        # curve is threshold-placement noise, not detector quality — under extreme
        # imbalance the calibrated value can swing 0→1 while the oracle stays flat.
        # Drawn for cost/fpr/fnr/f1 (not the IoU metrics, which aren't threshold-based).
        _ofield = _ORACLE_FIELD.get(field)
        if show_oracle and _ofield:
            oracle_k = _mean_by_k(crows, _ofield)
            oks = sorted(k for k in oracle_k if k > 0)
            if oks:
                ax.plot(oks, [oracle_k[k] for k in oks], color=color, ls="--", lw=1.2, alpha=0.55)
                oracle_drawn = True

    ax.set_xlabel(x_label)
    ax.set_ylabel(_METRICS.get(field, field))
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    _legend_with_oracle(ax, _ORACLE_FIELD.get(field) if oracle_drawn else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _legend_with_oracle(ax, oracle_label: str | None) -> None:
    """Draw the legend, appending a single grey dashed proxy naming the oracle
    companion (``oracle_label``, e.g. ``"oracle_cost"`` / ``"oracle_f1"``) so
    "dashed = <that oracle>" is self-documenting without one entry per curve.
    Falls back to a plain legend when ``oracle_label`` is None (no oracle drawn)."""
    if not oracle_label:
        ax.legend(fontsize=8, loc="best")
        return
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="grey", ls="--", lw=1.2, alpha=0.8))
    labels.append(oracle_label)
    ax.legend(handles, labels, fontsize=8, loc="best")


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


def _plot_summary_group(
    rows: list[dict],
    title: str,
    out_path: Path,
    *,
    field: str = "cost",
    band: str = "std",
    show_oracle: bool = False,
    x_label: str = "K (annotation count)",
) -> None:
    """One figure for a whole dataset: each config's curve **macro-averaged across classes**.
    Mirrors ``_plot_group`` but aggregates over classes (band = across-class spread); ``band='all'``
    overlays one thin line per class per config. Cosine-head rows are skipped."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[_config_key(r)].append(r)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    color_i = 0
    oracle_drawn = False
    for key, crows in sorted(by_cfg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        head = key[3]
        if head == "cosine":
            continue  # zero-shot baseline isn't a per-t curve; omit from the cross-class summary
        cfg_label = _config_label(key)

        if band == "all":
            # One thin line per class (each its own color) so cross-class spread is visible directly.
            _ofield = _ORACLE_FIELD.get(field)
            oracle_by_cls = _class_series(crows, _ofield) if (show_oracle and _ofield) else {}
            for cls, series in sorted(_class_series(crows, field).items()):
                sks = sorted(k for k in series if k > 0)
                if not sks:
                    continue
                cls_color = cmap(color_i % 10)
                ax.plot(
                    sks,
                    [series[k] for k in sks],
                    color=cls_color,
                    ls="-",
                    lw=1.3,
                    label=f"{cfg_label} — {cls}",
                )
                oks = sorted(k for k in oracle_by_cls.get(cls, {}) if k > 0)
                if oks:
                    ax.plot(oks, [oracle_by_cls[cls][k] for k in oks], color=cls_color, ls="--", lw=1.0, alpha=0.5)
                    oracle_drawn = True
                color_i += 1
            continue

        stats = _summary_stats(crows, field, band)
        ks = sorted(k for k in stats if k > 0)
        if not ks:
            continue
        color = cmap(color_i % 10)
        color_i += 1
        ax.plot(
            ks,
            [stats[k][0] for k in ks],
            color=color,
            ls="-",
            lw=1.8,
            label=cfg_label,
        )
        # Across-class spread band (min-max or ±std over the per-class means).
        if band != "none":
            ax.fill_between(ks, [stats[k][1] for k in ks], [stats[k][2] for k in ks], color=color, alpha=0.15, lw=0)
        # Oracle companion (best-τ), macro-averaged across classes like the main curve.
        _ofield = _ORACLE_FIELD.get(field)
        if show_oracle and _ofield:
            ostats = _summary_stats(crows, _ofield, band)
            oks = sorted(k for k in ostats if k > 0)
            if oks:
                ax.plot(oks, [ostats[k][0] for k in oks], color=color, ls="--", lw=1.2, alpha=0.55)
                oracle_drawn = True

    ax.set_xlabel(x_label)
    ax.set_ylabel(_METRICS.get(field, field))
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    _legend_with_oracle(ax, _ORACLE_FIELD.get(field) if oracle_drawn else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_summary(
    rows: list[dict],
    out_dir: Path,
    *,
    metrics: list[str] | tuple[str, ...] = _DEFAULT_METRICS,
    band: str = "std",
    show_oracle: bool = False,
    x_label: str = "K (annotation count)",
    x_tag: str = "k",
) -> None:
    """Write one cross-class **summary** figure per metric per dataset: each config's curve
    macro-averaged over that dataset's classes (band = across-class spread; ``band='all'`` = one line
    per class). Emitted only for datasets with ≥2 classes — a single-class dataset's per-class figure
    already is its summary, so it's skipped."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["dataset"])].append(r)
    for dataset, grows in sorted(groups.items()):
        n_classes = len({str(r["class"]) for r in grows})
        if n_classes < 2:
            continue
        for field in metrics:
            out_path = out_dir / f"summary_{dataset}_{field}_vs_{x_tag}.png"
            _plot_summary_group(
                grows,
                f"{dataset}: mean over {n_classes} classes — {field} vs {x_tag}",
                out_path,
                field=field,
                band=band,
                show_oracle=show_oracle,
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
    ap.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also emit cross-class summary_<dataset>_<metric>.png (macro-avg over a dataset's "
        "classes); no-op for single-class datasets",
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
    if args.summary:
        render_summary(rows, out_dir, metrics=args.metrics, band=args.band_kind, show_oracle=args.show_oracle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
