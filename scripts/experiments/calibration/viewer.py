"""Build the **interactive quality-over-clicks viewer** a study's report links to.

The two committed PNGs (:mod:`curves`) answer the questions a report *asks*.
They cannot answer the questions a reader has after reading it — *"is that true
on `visual_genome_m` too?"*, *"is it the scarce categories doing all the work?"*,
*"does it hold on recall, or only on cost?"* — because each of those is a
different slice and a PNG is one slice, chosen in advance by whoever wrote the
analyzer.  This builds a single self-contained HTML page that carries **every**
slice, so a reader can ask their own question instead of asking for a re-run.

What the page lets a reader pick:

* **dataset** — one, or all of them averaged together;
* **category** — one within that dataset, or all averaged;
* **embedder** — any non-empty subset, drawn as **one panel each**.  Embedders
  are never averaged with one another: two embedders are two different
  representations of the haystack, and their mean is a number describing no
  system anyone could run.  Faceting makes that structural rather than a rule
  someone has to remember;
* **arms** — any subset, each carrying its own colour, fixed by the arm's
  position in the study rather than by which arms happen to be on screen;
* **seeds** — averaged, or every seed as its own line;
* **metric** — cost, precision, recall, F1, FPR, FNR, average precision, AUROC:
  whatever the run emitted, from one shared definition
  (:data:`vtscore.eval.calibration_metrics.DETECTION_METRICS`).

Click 0 is the **zero-click text sort**, exactly as in :mod:`curves`: the far
left is what typing the query got for free, so a reader can see at a glance
whether the clicking ever earned its keep and after how many clicks.

Payload
-------

Two arrays, both embedded, because they answer different questions and neither
is derivable from the other at an acceptable size:

``agg``
    Per ``(group, arm, metric, click)``: ``mean``, ``sd`` and ``n``, at **full**
    click resolution, for every metric.  A group is one
    ``(dataset, embedder, category)``.  Storing ``n`` beside the moments is what
    makes "all categories" and "all datasets" *exact* rather than a mean of
    means — the page pools them properly, weighted by the cells that actually
    contributed.
``runs``
    Per ``(group, arm, seed, metric, click)``: the raw series behind the
    per-seed lines, on a **thinned** click grid chosen to fit
    ``runs_budget_mb``.  A 42-seed grid across 24 environments and 8 arms is
    ~13 million numbers at full resolution and no amount of packing makes that a
    committable file; thinning the click axis is the one lossy choice that costs
    a reader nothing they would have seen (per-seed lines are read for spread,
    not for fine structure).  **The chosen grid is written into the payload and
    shown in the page**, and if even the coarsest grid busts the budget the
    per-seed mode is disabled with the reason on screen — never silently
    dropped, and never quietly subsampled down to a tidier-looking set of runs.

Both are quantised to int16, NaN-masked, delta-coded along the click axis and
gzipped; the page inflates them with ``DecompressionStream``.

Usage
-----

::

    python viewer.py --results "$CALIB_RESULTS" --arms prod,top_long \\
        --baseline "$OUT/text_baseline.csv" --out "$OUT/viewer.html"
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import curves  # noqa: E402

#: NaN in the int16 encoding.  Values are masked separately, so this is only the
#: filler that keeps the delta stream smooth across a gap.
FILL = 0

#: Quantisation: values are stored as ``round(value * SCALE)``.  1/4000 is a
#: quarter of a thousandth on a [0,1] metric — an order of magnitude finer than
#: anything a reader can resolve on screen, and coarse enough that the delta
#: stream still compresses.
SCALE = int(os.environ.get("VIEWER_SCALE", "4000"))

#: Byte budget for the per-seed payload, before base64.  The click grid is
#: thinned until it fits.  Raise it for a study small enough to afford full
#: resolution; the page always says which grid it got.
RUNS_BUDGET_MB = float(os.environ.get("VIEWER_RUNS_BUDGET_MB", "1.5"))

TEMPLATE = Path(__file__).with_name("viewer_template.html")

#: The one placeholder the template carries; see the note in its header comment.
TOKEN = "__VIEWER" + "_PAYLOAD__"


def _b64gz(buf: bytes) -> str:
    return base64.b64encode(gzip.compress(buf, 9)).decode("ascii")


def _encode(arr: np.ndarray, scale: int = SCALE) -> dict:
    """Quantise, mask, delta-code along the last axis, gzip, base64.

    The mask is what lets the delta stream stay smooth across a gap: a NaN run
    is filled with its neighbour's value so it contributes zero deltas, and the
    separate 1-bit-per-sample mask (which is all runs, so it compresses to
    nearly nothing) restores the NaNs on the other side.
    """
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    valid = np.isfinite(flat)
    q = np.where(valid, np.clip(np.rint(flat * scale), -32000, 32000), np.nan)
    # Forward-fill the gaps, then zero whatever is still NaN (a leading gap).
    idx = np.where(valid, np.arange(flat.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    q = np.take_along_axis(np.nan_to_num(q, nan=FILL), idx, axis=1)
    qi = q.astype(np.int32)
    deltas = np.diff(qi, axis=1, prepend=0).astype(np.int16)
    return {
        "scale": scale,
        "shape": list(arr.shape),
        "v": _b64gz(deltas.tobytes()),
        "m": _b64gz(np.packbits(valid, axis=None).tobytes()),
    }


def _payload_bytes(enc: dict) -> int:
    return len(enc["v"]) + len(enc["m"])


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def _metric_list(main: pd.DataFrame) -> list[dict]:
    """Every metric the run emitted, in the shared canonical order.

    Read from :data:`vtscore.eval.calibration_metrics.DETECTION_METRICS` so a
    metric's label and its direction come from the same place the harness
    computes it — a viewer that decided for itself which way is "better" is how
    "lower is better" gets attached to recall.
    """
    from vtscore.eval.calibration_metrics import DETECTION_METRICS

    out = []
    for key, (label, lower, domain) in DETECTION_METRICS.items():
        if key in main.columns and main[key].notna().any():
            out.append({"key": key, "label": label, "lower": bool(lower), "lo": domain[0], "hi": domain[1]})
    return out


def _thin(t_full: np.ndarray, keep: int) -> np.ndarray:
    """*keep* clicks out of *t_full*, always including click 0 and the horizon.

    Weighted toward the early clicks, which is where every one of these curves
    does its moving: an evenly spaced grid spends half its points on a plateau.
    """
    if keep >= len(t_full):
        return t_full
    lo, hi = float(t_full[0]), float(t_full[-1])
    # Even spacing in sqrt(t): dense at the start, sparse at the horizon.
    want = np.unique(np.rint(lo + (np.linspace(0.0, 1.0, keep) ** 2) * (hi - lo)).astype(int))
    return np.array([t for t in t_full if t in set(want.tolist())], dtype=int)


class _Shape:
    """The index a payload is built against: groups, arms, seeds, metrics."""

    def __init__(self, main: pd.DataFrame, arms: Sequence[str], denominator: pd.DataFrame | None):
        self.arms = [a for a in arms if (main["arm"] == a).any()]
        self.metrics = _metric_list(main)
        self.datasets = sorted(str(d) for d in main["dataset"].dropna().unique())
        self.embedders = (
            sorted(str(e) for e in main["embedder"].dropna().unique()) if "embedder" in main.columns else [""]
        )
        self.categories = sorted(str(c) for c in main["category"].dropna().unique())
        self.seeds = sorted(int(s) for s in main["seed"].dropna().unique())
        src = denominator if denominator is not None and not denominator.empty else main
        cols = [c for c in ("dataset", "embedder", "category") if c in src.columns]
        pairs = src.loc[:, cols].drop_duplicates()
        self.groups: list[tuple[str, str, str]] = sorted(
            (str(r[0]), str(r[1]) if "embedder" in cols else "", str(r[-1]))
            for r in pairs.itertuples(index=False, name=None)
        )
        self.gi = {g: i for i, g in enumerate(self.groups)}
        self.ai = {a: i for i, a in enumerate(self.arms)}
        self.si = {s: i for i, s in enumerate(self.seeds)}


#: Joins ``(dataset, embedder, category)`` into one groupby key.  A control
#: character rather than a space or a slash: COCO categories are things like
#: "baseball glove" and "hair drier", so any printable separator is a category
#: name away from splitting in the wrong place.
GSEP = "\u001f"


def _gkey(dataset: str, embedder: str, category: str) -> str:
    return GSEP.join((str(dataset), str(embedder or ""), str(category)))


def _group_key(df: pd.DataFrame) -> pd.Series:
    emb = df["embedder"].astype(str) if "embedder" in df.columns else pd.Series("", index=df.index)
    return df["dataset"].astype(str) + GSEP + emb + GSEP + df["category"].astype(str)


def _agg_arrays(  # noqa: C901
    main: pd.DataFrame,
    shape: _Shape,
    t_full: np.ndarray,
    base: dict[str, dict[tuple, float]],
    cells: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, sd, n, cells)`` over ``(group, arm, metric, click)``.

    ``n`` is stored beside the moments on purpose: it is what makes an "all
    categories" or "all datasets" selection *exact* in the page.  Pooling means
    of means would weight a category that trained on 3 cells the same as one
    that trained on 42, which is precisely the survivorship the coverage strip
    exists to expose.
    """
    nG, nA, nM, nT = len(shape.groups), len(shape.arms), len(shape.metrics), len(t_full)
    mean = np.full((nG, nA, nM, nT), np.nan)
    sd = np.full((nG, nA, nM, nT), np.nan)
    n = np.zeros((nG, nA, nM, nT))
    ncells = np.zeros((nG, nA))
    t_pos = {int(t): i for i, t in enumerate(t_full)}

    for (gk, arm), g in main.groupby(["__group", "arm"], dropna=False):
        gi, ai = shape.gi.get(tuple(str(gk).split(GSEP))), shape.ai.get(arm)
        if gi is None or ai is None:
            continue
        rows = g["t"].map(t_pos)
        ok = rows.notna()
        cols = rows[ok].astype(int).to_numpy()
        for mi, spec in enumerate(shape.metrics):
            vals = g.loc[ok, spec["key"]].to_numpy(dtype=float)
            good = np.isfinite(vals)
            if not good.any():
                continue
            # Sum / sumsq / count per click, accumulated with bincount so a
            # cell contributing at only some clicks lands only there.
            cnt = np.bincount(cols[good], minlength=nT).astype(float)
            s1 = np.bincount(cols[good], weights=vals[good], minlength=nT)
            s2 = np.bincount(cols[good], weights=vals[good] ** 2, minlength=nT)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
                var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mu**2, np.nan)
            mean[gi, ai, mi] = mu
            sd[gi, ai, mi] = np.sqrt(np.clip(var, 0.0, None))
            n[gi, ai, mi] = cnt

    for (ds, emb, cat), gi in shape.gi.items():
        for arm, ai in shape.ai.items():
            ncells[gi, ai] = cells.get((arm, _gkey(ds, emb, cat)), 0)

    # Click 0 is the zero-click text sort: every attempted cell has one, whether
    # or not it ever trained a detector.  Written here rather than left to the
    # page so the anchor obeys the same pooling as everything else.
    if 0 in t_pos:
        z = t_pos[0]
        for mi, spec in enumerate(shape.metrics):
            bm = base.get(spec["key"]) or {}
            if not bm:
                continue
            for (ds, emb, cat), gi in shape.gi.items():
                vals = [v for (d, e, c, _s), v in bm.items() if (d, e, c) == (ds, emb, cat)]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                for ai in range(nA):
                    if ncells[gi, ai] <= 0:
                        continue
                    mean[gi, ai, mi, z] = float(arr.mean())
                    sd[gi, ai, mi, z] = float(arr.std())
                    n[gi, ai, mi, z] = float(ncells[gi, ai])
    return mean, sd, n, ncells


def _runs_arrays(
    main: pd.DataFrame,
    shape: _Shape,
    t_grid: np.ndarray,
    base: dict[str, dict[tuple, float]],
) -> tuple[np.ndarray, list[list[int]]]:
    """``values[(run, metric, click)]`` plus the ``(group, arm, seed)`` index.

    Built once at full click resolution; :func:`build_viewer` slices columns out
    of the result for each candidate grid rather than re-running this, which is
    the difference between a few seconds and several minutes on a grid this size.
    """
    t_pos = {int(t): i for i, t in enumerate(t_grid)}
    index: list[list[int]] = []
    nM, nT = len(shape.metrics), len(t_grid)
    seen: dict[tuple[int, int, int], int] = {}
    blocks: list[np.ndarray] = []

    def _slot(gi: int, ai: int, si: int) -> int:
        key = (gi, ai, si)
        if key not in seen:
            seen[key] = len(blocks)
            blocks.append(np.full((nM, nT), np.nan))
            index.append([gi, ai, si])
        return seen[key]

    keys = [spec["key"] for spec in shape.metrics]
    for (gk, arm, seed), g in main.groupby(["__group", "arm", "seed"], dropna=False, sort=False):
        gi, ai, si = shape.gi.get(tuple(str(gk).split(GSEP))), shape.ai.get(arm), shape.si.get(int(seed))
        if gi is None or ai is None or si is None:
            continue
        cols = g["t"].map(t_pos).to_numpy()
        ok = np.isfinite(pd.to_numeric(cols, errors="coerce").astype(float))
        if not ok.any():
            continue
        ci = cols[ok].astype(int)
        blocks[_slot(gi, ai, si)][:, ci] = g.loc[ok, keys].to_numpy(dtype=float).T

    # Every attempted cell gets its click-0 anchor, including one that never
    # trained: a lone point at the far left is exactly what total starvation
    # looks like, and dropping it would render that run as simply absent.
    if 0 in t_pos:
        z = t_pos[0]
        for mi, spec in enumerate(shape.metrics):
            for (ds, emb, cat, seed), val in (base.get(spec["key"]) or {}).items():
                gi = shape.gi.get((ds, emb, cat))
                si = shape.si.get(int(seed))
                if gi is None or si is None or not np.isfinite(val):
                    continue
                for ai in range(len(shape.arms)):
                    blocks[_slot(gi, ai, si)][mi, z] = val
    return (np.stack(blocks) if blocks else np.zeros((0, nM, nT))), index


def _baselines(baseline: pd.DataFrame | None, shape: _Shape) -> dict[str, dict[tuple, float]]:
    """``metric -> {(dataset, embedder, category, seed): value}``.

    Uses :func:`curves.baseline_map` for the column lookup, so the page's click-0
    anchor and the PNG's click-0 anchor read the same column of the same file.
    """
    if baseline is None or baseline.empty:
        return {}
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in baseline.columns]
    out: dict[str, dict[tuple, float]] = {}
    for spec in shape.metrics:
        m = curves.baseline_map(baseline, spec["key"], keys)
        if not m:
            continue
        fixed: dict[tuple, float] = {}
        for k, v in m.items():
            rec = dict(zip(keys, k, strict=False))
            fixed[
                (
                    str(rec.get("dataset", "")),
                    str(rec.get("embedder", "")),
                    str(rec.get("category", "")),
                    int(rec.get("seed", 0)),
                )
            ] = v
        out[spec["key"]] = fixed
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_viewer(  # noqa: C901
    main: pd.DataFrame,
    out_path: Path,
    *,
    arms: Sequence[str],
    denominator: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    title: str = "Quality over clicks",
    subtitle: str = "",
    runs_budget_mb: float = RUNS_BUDGET_MB,
    anchor_label: str = curves.BASELINE_LABEL,
    template: Path = TEMPLATE,
) -> Path:
    """Write the self-contained viewer HTML.  Returns *out_path*."""
    if main.empty:
        raise SystemExit("viewer: no rows to build from")
    main = main.copy()
    if "embedder" not in main.columns:
        main["embedder"] = ""
    main["__group"] = _group_key(main)

    shape = _Shape(main, arms, denominator)
    if not shape.metrics:
        raise SystemExit("viewer: the frame carries none of the known metric columns")

    den = denominator if denominator is not None and not denominator.empty else main
    den = den.copy()
    if "embedder" not in den.columns:
        den["embedder"] = ""
    den["__group"] = _group_key(den)
    cells = {
        (str(arm), str(gk)): int(d.loc[:, ["seed"]].drop_duplicates().shape[0])
        for (arm, gk), d in den.groupby(["arm", "__group"])
    }

    base = _baselines(baseline, shape)
    has_anchor = bool(base)
    t_full = np.arange(0 if has_anchor else 1, int(main["t"].max()) + 1)

    mean, sd, n, ncells = _agg_arrays(main, shape, t_full, base, cells)
    agg = {
        "mean": _encode(mean),
        "sd": _encode(sd),
        "n": _encode(n, scale=1),
        "cells": _encode(ncells.reshape(len(shape.groups), len(shape.arms), 1), scale=1),
    }

    # Per-seed payload: thin the click axis until it fits, and say which grid it
    # landed on.  Never drop runs, never drop metrics - a reader comparing two
    # arms must be looking at the same set of seeds for both, and a metric that
    # silently vanished from one view is worse than a coarser axis in all of
    # them.  Built once at full resolution and sliced, because rebuilding the
    # frame per candidate grid is minutes rather than seconds at this size.
    budget = int(runs_budget_mb * 1024 * 1024)
    runs = None
    runs_note = ""
    full_vals, index = _runs_arrays(main, shape, t_full, base)
    sizes: dict[str, int] = {k: _payload_bytes(v) for k, v in agg.items()}
    if len(index):
        candidates = [len(t_full), 120, 80, 60, 45, 34, 26, 20, 15]
        candidates = [c for i, c in enumerate(candidates) if c <= len(t_full) and c not in candidates[:i]]
        pos = {int(t): i for i, t in enumerate(t_full)}
        for keep in candidates:
            grid = _thin(t_full, keep)
            enc = _encode(full_vals[:, :, [pos[int(t)] for t in grid]])
            fits = _payload_bytes(enc) <= budget
            if fits or keep == candidates[-1]:
                if not fits:
                    runs_note = (
                        f"Per-seed lines are drawn at {len(grid)} of {len(t_full)} clicks — the coarsest grid "
                        f"available, and still over the {runs_budget_mb:g} MB payload budget."
                    )
                elif len(grid) < len(t_full):
                    runs_note = (
                        f"Per-seed lines are drawn at {len(grid)} of {len(t_full)} clicks, thinned to fit the "
                        f"{runs_budget_mb:g} MB payload budget (denser at the start, where these curves move). "
                        f"The averaged view is at full resolution."
                    )
                else:
                    runs_note = f"Per-seed lines are at full click resolution ({len(t_full)} clicks)."
                runs = {"t": [int(x) for x in grid], "index": index, "values": enc}
                sizes["runs"] = _payload_bytes(enc)
                break
    else:
        runs_note = "No per-seed rows were found, so per-seed lines are unavailable."

    payload = {
        "schema": 1,
        "title": title,
        "subtitle": subtitle,
        "anchor": {"has": has_anchor, "label": anchor_label},
        "solid_coverage": curves.SOLID_COVERAGE,
        "datasets": shape.datasets,
        "embedders": shape.embedders,
        "categories": shape.categories,
        "arms": shape.arms,
        "seeds": shape.seeds,
        "metrics": shape.metrics,
        "groups": [list(g) for g in shape.groups],
        "t": [int(x) for x in t_full],
        "agg": agg,
        "runs": runs,
        "runs_note": runs_note,
        "n_cells": int(sum(cells.values())),
        "payload_kb": {k: round(v / 1024) for k, v in sizes.items()},
    }

    html = template.read_text(encoding="utf-8")
    # Exactly one, or the page silently doubles: the substitution is a plain
    # string replace, and the token used to appear in the template's own header
    # comment too - which cost 3 MB and looked like a payload problem.
    if html.count(TOKEN) != 1:
        raise SystemExit(f"{template}: expected exactly 1 {TOKEN}, found {html.count(TOKEN)}")
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(TOKEN, blob), encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(common.RESULTS), help="results root holding one dir per arm")
    ap.add_argument("--arms", required=True, help="comma-separated arm directories, in report order")
    ap.add_argument("--out", required=True, help="path to write the HTML to")
    ap.add_argument("--baseline", default=None, help="text_baseline.py CSV: the click-0 anchor")
    ap.add_argument("--title", default="Quality over clicks")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--runs-budget-mb", type=float, default=RUNS_BUDGET_MB)
    args = ap.parse_args(list(argv) if argv is not None else None)

    arms = [a for a in args.arms.replace(",", " ").split() if a]
    frame = curves._load(Path(args.results), arms)
    if frame.empty:
        print(f"no rows under {args.results} for arms {arms}")
        return 2
    baseline = curves.text_sort_baseline(args.baseline) if args.baseline else None
    out = build_viewer(
        frame,
        Path(args.out),
        arms=arms,
        baseline=baseline,
        title=args.title,
        subtitle=args.subtitle,
        runs_budget_mb=args.runs_budget_mb,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    if not args.baseline:
        print("NOTE: no --baseline, so the page has no click-0 anchor and nothing to compare the far right against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
