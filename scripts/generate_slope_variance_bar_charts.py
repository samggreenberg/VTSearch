#!/usr/bin/env python3
"""Bar charts of per-config SLOPE and VARIANCE from a sweep ``results.csv``.

The SOD sweep's line plots (``scripts/sod/plots.py``) draw one curve per config
combination (embedder x proposal x hac leaf knobs x resolution x ...), metric vs the
annotation count ``t``. This script collapses each such curve to two scalars and
bar-charts them, with the config combinations on the x-axis, so you can compare configs
at a glance instead of eyeballing overlaid curves:

* **slope**    - least-squares ``d(metric)/d(t)`` pooled over all seeds: how fast the
                 config improves per annotation. For a "lower is better" metric (cost /
                 fpr / fnr / oracle_cost) a *negative* slope is improvement; for f1 /
                 oracle_f1 / mean_iou / corloc a *positive* slope is. Bars are coloured
                 green (improving) / red (worsening) accordingly.
* **variance** - by default the across-seed variance of the metric (mean over ``t`` of
                 the per-``t`` variance across seeds): how reproducible / consistent the
                 config is - the "does one seed look nothing like the next" axis.
                 ``--variance-mode residual`` instead reports the variance of the metric
                 around its own slope fit (curve jitter / non-linearity).

By default each bar is averaged over BOTH iterations (seeds) and classes: one figure
per ``dataset`` per metric, where every config's slope/variance is computed per class
(pooled over that class's seeds) and then macro-averaged across the dataset's classes,
with the across-class spread drawn as an error bar. Pass ``--per-class`` for one figure
per ``(dataset, class)`` instead. Each figure has two side-by-side panels (slope left,
variance right), x = config combinations. ``--include`` / ``--exclude`` filter which
combinations are charted (substring match on the label); ``--no-error-bars`` hides the
across-class spread whiskers. Reads the CSV the sweep writes; no model / GPU needed.
Example::

    python scripts/generate_slope_variance_bar_charts.py \\
        docs/experiments/sod-sweep/cmp2-leaf-topk-spatial-vs-spread-feature/results.csv
        
# Matthew Usage:
python scripts/generate_slope_variance_bar_charts.py docs/experiments/sod-sweep/cmp1-siglip-whole-vs-dino-hac-p2/results.csv --exclude dinov3/whole --rename dinov3/hac=DINOv3-HAC siglip2/whole=SigLIP-2 --title "COCO 7 Classes - {metric}"

"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Config columns that define one plotted "combination" (mirrors scripts/sod/plots._config_key).
_CONFIG_COLS = (
    "embedder",
    "proposal",
    "alpha",
    "head",
    "leaf_seeding",
    "leaf_assign",
    "pca_dims",
    "resolution",
    "hac_k",
    "leaf_beta",
    "dinov3_model",
)
# Metrics where a *decrease* over t is improvement (so a negative slope is "good").
_LOWER_BETTER = {"cost", "fpr", "fnr", "oracle_cost"}
_DEFAULT_METRICS = ("cost", "oracle_cost", "f1", "oracle_f1", "mean_iou", "corloc")
_EMPTY = {"", "none", "nan"}


def _as_float(value: str | None) -> float:
    """One CSV cell -> float, or NaN for blank / None / nan."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s.lower() in _EMPTY:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _config_key(row: dict) -> tuple:
    return tuple(str(row.get(c, "")) for c in _CONFIG_COLS)


def _combo_label(row: dict) -> str:
    """Compact human label for one config combination (mirrors plots._config_label)."""
    emb, prop = row.get("embedder", ""), row.get("proposal", "")
    res, model = str(row.get("resolution", "")), str(row.get("dinov3_model", ""))
    res_tag = "" if res.lower() in _EMPTY or res == "0" else f" r{res}"
    model_tag = "" if model.lower() in _EMPTY or model == "vitb16" else f" {model}"
    if prop != "hac":
        return f"{emb}/{prop}{res_tag}{model_tag}"
    k, alpha = row.get("hac_k", ""), row.get("alpha", "")
    seeding, assign = row.get("leaf_seeding", ""), row.get("leaf_assign", "")
    beta, pca = str(row.get("leaf_beta", "")), str(row.get("pca_dims", ""))
    beta_tag = f" b{beta}" if (assign == "feature" and beta.lower() not in _EMPTY) else ""
    pca_s = "none" if pca.lower() in _EMPTY or pca == "0" else pca
    return f"{emb}/hac k{k} a{alpha} {seeding}/{assign}{beta_tag} pca{pca_s}{res_tag}{model_tag}"


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.strip()).strip("_").lower() or "x"


def _apply_rename(label: str, pairs: list[tuple[str, str]]) -> str:
    """Return *label* replaced by the first 'SUBSTR=NAME' whose SUBSTR it contains (else itself)."""
    lo = label.lower()
    for sub, new in pairs:
        if sub.lower() in lo:
            return new
    return label


def _slope_and_variance(rows: list[dict], metric: str, xcol: str, variance_mode: str, min_x: float | None):
    """Return ``(slope, variance)`` for one config's rows, or ``None`` if not computable."""
    xs: list[float] = []
    ys: list[float] = []
    per_x: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        x, y = _as_float(r.get(xcol)), _as_float(r.get(metric))
        if math.isnan(x) or math.isnan(y):
            continue
        if min_x is not None and x < min_x:
            continue
        xs.append(x)
        ys.append(y)
        per_x[x].append(y)
    if len(set(xs)) < 2:  # need at least two distinct x to fit a slope
        return None
    xa, ya = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    coeffs = np.polyfit(xa, ya, 1)
    slope = float(coeffs[0])
    if variance_mode == "residual":
        variance = float(np.var(ya - np.polyval(coeffs, xa)))
    else:  # across_seed: mean over t of the across-seed variance at each t
        variance = float(np.mean([np.var(v) for v in per_x.values()]))
    return slope, variance


def _aggregate_over_classes(rows, metric, xcol, variance_mode, min_x):
    """One config's rows -> its slope + variance MACRO-averaged across classes.

    Computes each class's (slope, across-seed variance) independently (so the two
    classes weight equally and a big-scale class can't dominate), then averages over
    classes. Returns ``(mean_slope, mean_var, std_slope, std_var, n_classes)`` (the two
    stds are the across-class spread, drawn as error bars), or ``None`` if no class was
    computable. With a single class this is just that class's values (std 0)."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[str(r.get("class", ""))].append(r)
    slopes: list[float] = []
    variances: list[float] = []
    for crows in by_class.values():
        sv = _slope_and_variance(crows, metric, xcol, variance_mode, min_x)
        if sv is not None:
            slopes.append(sv[0])
            variances.append(sv[1])
    if not slopes:
        return None
    return (
        float(np.mean(slopes)),
        float(np.mean(variances)),
        float(np.std(slopes)),
        float(np.std(variances)),
        len(slopes),
    )


def _plot(
    dataset,
    scope,
    metric,
    labels,
    slopes,
    variances,
    *,
    xcol,
    variance_mode,
    out_dir,
    slope_err=None,
    var_err=None,
    title=None,
    slope_ylabel=None,
    variance_ylabel=None,
    rotation=0,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Conference-paper styling: large, legible type; clean (despined) axes.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.4,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "figure.titlesize": 24,
        }
    )

    lower_better = metric in _LOWER_BETTER

    def _slope_color(s: float) -> str:
        improving = (s < 0) if lower_better else (s > 0)
        return "#3a9a4f" if improving else "#c0433a"

    n = len(labels)
    fig, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(max(14.0, 1.5 * n + 8.0), 9.0))
    idx = list(range(n))
    err_kw = {"ecolor": "#222222", "capsize": 5, "error_kw": {"elinewidth": 1.6}}

    def _annotate(ax, bars, values):
        # Clamp floating-point noise (e.g. a corloc slope of 1.8e-18) to a clean "0".
        labs = ["0" if abs(v) < 1e-6 else f"{v:.3g}" for v in values]
        ax.bar_label(bars, labels=labs, padding=4, fontsize=15)

    bars_s = ax_s.bar(idx, slopes, color=[_slope_color(s) for s in slopes], yerr=slope_err, **err_kw)
    ax_s.axhline(0, color="black", lw=1.0)
    ax_s.set_ylabel(slope_ylabel or f"slope  d({metric})/d({xcol})")
    # ax_s.set_title("slope  (green = improving, red = worsening)")
    _annotate(ax_s, bars_s, slopes)

    v_label = "across-seed variance" if variance_mode == "across_seed" else "residual (curve-jitter) variance"
    bars_v = ax_v.bar(idx, variances, color="#4878c8", yerr=var_err, **err_kw)
    ax_v.set_ylabel(variance_ylabel or f"{v_label} of {metric}")
    # ax_v.set_title(v_label)
    _annotate(ax_v, bars_v, variances)

    ha = "center" if rotation % 180 == 0 else "right"
    for ax in (ax_s, ax_v):
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=rotation, ha=ha)
        ax.margins(y=0.14)  # headroom for the value labels
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.tick_params(width=1.4, length=6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title or f"{dataset}: {scope}  -  {metric}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"{_slugify(dataset)}_{_slugify(scope)}_{metric}_slope_variance.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_csv", type=Path, help="path to a sweep results.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write PNGs (default: <results_csv dir>/slope_variance)",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=list(_DEFAULT_METRICS),
        help=f"metric columns to chart (default: {' '.join(_DEFAULT_METRICS)}); ones absent from the CSV are skipped",
    )
    ap.add_argument(
        "--x-column",
        default=None,
        help="x-axis column for the slope fit: 't' (annotations) or 'k'; default auto (t if present, else k)",
    )
    ap.add_argument(
        "--variance-mode",
        choices=("across_seed", "residual"),
        default="across_seed",
        help="across_seed (default): reproducibility across seeds; residual: metric jitter around its own slope fit",
    )
    ap.add_argument(
        "--min-x",
        type=float,
        default=None,
        help="only fit/aggregate rows with x >= this (e.g. skip cold-start steps)",
    )
    ap.add_argument(
        "--per-class",
        action="store_true",
        help="one figure per (dataset, class) instead of the default macro-average over each dataset's classes",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=[],
        metavar="SUBSTR",
        help="only chart config combinations whose label contains ANY of these substrings (case-insensitive)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="SUBSTR",
        help="drop config combinations whose label contains ANY of these substrings (applied after --include)",
    )
    ap.add_argument(
        "--error-bars",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw across-class spread (std) as error bars (default off; --error-bars to show). For the "
        "variance panel this is the spread of the across-seed variance across classes, i.e. how much a "
        "config's inconsistency itself differs by class. No-op with --per-class (a single class has no spread).",
    )
    ap.add_argument(
        "--rename",
        nargs="*",
        default=[],
        metavar="SUBSTR=NAME",
        help="rename x-axis config labels: a combo whose label contains SUBSTR is shown as NAME (first match "
        "wins). e.g. --rename spread/feature=spread+feat topk/spatial=baseline",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="override the figure title; supports {dataset} {scope} {metric} {xcol} placeholders "
        "(default: '<dataset>: <scope> - <metric>')",
    )
    ap.add_argument("--slope-ylabel", default="Slope", help="override the slope panel y-axis label")
    ap.add_argument("--variance-ylabel", default="Variance", help="override the variance panel y-axis label")
    ap.add_argument(
        "--label-rotation",
        type=int,
        default=0,
        help="x-tick label rotation in degrees (default 0 = horizontal; use 45 or 90 for long labels)",
    )
    args = ap.parse_args()

    rename_pairs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.rename if "=" in s]

    with args.results_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no rows in {args.results_csv}")

    xcol = args.x_column or ("t" if "t" in fields and any(str(r.get("t", "")).strip() for r in rows) else "k")
    if xcol not in fields:
        raise SystemExit(f"x-column {xcol!r} not in CSV columns: {fields}")
    metrics = [m for m in args.metrics if m in fields]
    if not metrics:
        raise SystemExit(f"none of --metrics {args.metrics} are columns in the CSV: {fields}")

    out_dir = args.out_dir or (args.results_csv.parent / "slope_variance")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default: one figure per dataset, each bar macro-averaged over that dataset's classes
    # (and pooled over iterations/seeds). --per-class: one figure per (dataset, class).
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dataset[str(r.get("dataset", ""))].append(r)

    n_written = 0
    for dataset, drows in sorted(by_dataset.items()):
        if args.per_class:
            by_class: dict[str, list[dict]] = defaultdict(list)
            for r in drows:
                by_class[str(r.get("class", ""))].append(r)
            scopes = sorted(by_class.items())  # (scope_label, rows) per class
        else:
            n_classes = len({str(r.get("class", "")) for r in drows})
            scopes = [(f"macro over {n_classes} classes", drows)]

        for scope_label, srows in scopes:
            by_cfg: dict[tuple, list[dict]] = defaultdict(list)
            for r in srows:
                by_cfg[_config_key(r)].append(r)
            for metric in metrics:
                labels: list[str] = []
                slopes: list[float] = []
                variances: list[float] = []
                slope_err: list[float] = []
                var_err: list[float] = []
                for _key, crows in sorted(by_cfg.items(), key=lambda kv: _combo_label(kv[1][0])):
                    label = _combo_label(crows[0])
                    lo = label.lower()
                    if args.include and not any(inc.lower() in lo for inc in args.include):
                        continue
                    if args.exclude and any(exc.lower() in lo for exc in args.exclude):
                        continue
                    agg = _aggregate_over_classes(crows, metric, xcol, args.variance_mode, args.min_x)
                    if agg is None:
                        continue
                    mean_slope, mean_var, std_slope, std_var, _n = agg
                    labels.append(label)
                    slopes.append(mean_slope)
                    variances.append(mean_var)
                    slope_err.append(std_slope)
                    var_err.append(std_var)
                if not labels:
                    print(f"skip {dataset}/{scope_label} {metric}: no config matched / had >=2 distinct {xcol} points")
                    continue
                # Error bars (across-class spread) only make sense when averaging >1 class.
                use_err = args.error_bars and not args.per_class and any(e > 0 for e in slope_err + var_err)
                disp_labels = [_apply_rename(lbl, rename_pairs) for lbl in labels]
                title = (
                    args.title.format(dataset=dataset, scope=scope_label, metric=metric, xcol=xcol)
                    if args.title
                    else None
                )
                out_path = _plot(
                    dataset,
                    scope_label,
                    metric,
                    disp_labels,
                    slopes,
                    variances,
                    xcol=xcol,
                    variance_mode=args.variance_mode,
                    out_dir=out_dir,
                    slope_err=slope_err if use_err else None,
                    var_err=var_err if use_err else None,
                    title=title,
                    slope_ylabel=args.slope_ylabel,
                    variance_ylabel=args.variance_ylabel,
                    rotation=args.label_rotation,
                )
                n_written += 1
                print(f"wrote {out_path}  ({len(labels)} configs, x={xcol}, {scope_label})")

    print(f"done: {n_written} figure(s) -> {out_dir}")


if __name__ == "__main__":
    main()
