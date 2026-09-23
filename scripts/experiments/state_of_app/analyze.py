#!/usr/bin/env python3
"""State of the App (#4159): per-cell quality, and what each clicked image did.

Reads one State of the App run (``launch.sh``) and writes, under ``--out``:

* ``cells.csv`` -- one row per run (arm x cell x seed): the text-only score at
  click 0, cost and F1 at fixed clicks and at the end, and the supervised
  ceiling (``skyline_train_full``). Left to right, that is what typing the
  query bought, what the clicks bought, and what full labels would buy.
* ``influence.csv`` -- one row per click: the change in the held-out test score
  that the click is credited with.
* ``images.csv`` / ``image_detector.csv`` -- the influence rolled up per image,
  and per image x detector, early and late separately.
* ``summary.md`` -- the tables a reader starts from.

**How a click is credited (owner, 2026-09-23).** The harness scores the test
split after every click once a Good and a Bad exist; before that there is no
detector to score. So a scored step's change is split EQUALLY among every click
since the previous scored step, and the first scored step's change is measured
from the text-only score (click 0) and split among the opening clicks. That is
exact when every click is scored and still honest when some are not. A click
after the last scored step is credited nothing.

**What the credit means.** It is the marginal change along THIS trajectory --
the head is refit from scratch each click, so part of any single delta is refit
noise. It becomes a statement about an image only in aggregate, over the runs
and detectors that clicked it; ``images.csv`` carries ``n_obs`` so a reader can
see how much aggregate there is.

    python analyze.py --exp /expscratch/$USER/state-of-the-app/<date> \\
        --baseline <exp>/analysis/text_baseline.csv --out <exp>/analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import _cells_io  # noqa: E402

#: The two production paths, as the harness names them.
ARMS = {("siglip", "whole_image"): "SigLIP binary", ("siglip+dinov3_patch", "max_patch"): "DINOv3 region"}
#: Clicks at which a curve is sampled into ``cells.csv``.
CHECKPOINTS = (10, 25, 50, 100, 150)
#: What "early" and "late" mean for a click, in clicks.
EARLY, LATE = 30, 90
RUN_KEY = ["dataset", "category", "embedder", "style", "seed"]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(exp: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(base, skyline, picks)`` for every cell of the run."""
    cells = exp / "results" / "cells"
    frames, sky, picks = [], [], []
    for f in _cells_io.main_frame_files(cells):
        try:
            df = pd.read_csv(f, low_memory=False)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if df.empty:
            continue
        if "gmm_variant" in df.columns:
            sky.append(df[df["gmm_variant"].astype(str) == "skyline_train_full"])
        frames.append(_cells_io._base_rows(df))
        p = f.with_name(f.stem + "__picks.csv")
        if p.exists() and p.stat().st_size:
            picks.append(pd.read_csv(p))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(frames), cat(sky), cat(picks)


def text_scores(baseline: Path | None) -> dict[tuple, tuple[float, float]]:
    """``{(dataset, category, embedder, seed): (text_cost, text_f1)}``."""
    if baseline is None or not baseline.exists():
        return {}
    tb = pd.read_csv(baseline)
    tb = tb[tb.get("supports_text", 1) == 1]
    return {(r.dataset, r.category, r.embedder, int(r.seed)): (_f(r.text_cost), _f(r.text_f1)) for r in tb.itertuples()}


def _text_for(ts: dict, ds: str, cat: str, emb: str, seed: int):
    # A paired region arm opens on its text half's sort, so its click-0 score is
    # the text embedder's when the pair has no row of its own.
    return ts.get((ds, cat, emb, seed)) or ts.get((ds, cat, emb.split("+")[0], seed))


def attribute(base: pd.DataFrame, picks: pd.DataFrame, ts: dict) -> pd.DataFrame:
    """One row per click: the share of the test-score change it is credited with."""
    series = base.groupby(RUN_KEY + ["t"], as_index=False)[["cost", "f1"]].mean().sort_values(RUN_KEY + ["t"])
    out = []
    pk = picks.sort_values(RUN_KEY + ["t"]) if not picks.empty else picks
    groups = dict(tuple(pk.groupby(RUN_KEY))) if not pk.empty else {}
    for key, s in series.groupby(RUN_KEY):
        clicks = groups.get(key)
        if clicks is None or clicks.empty:
            continue
        ds, cat, emb, style, seed = key
        start = _text_for(ts, ds, cat, emb, int(seed))
        prev_t, prev = 0, (start if start is not None else None)
        ct = clicks["t"].to_numpy()
        for t, cost, f1 in s[["t", "cost", "f1"]].itertuples(index=False):
            grp = clicks[(ct > prev_t) & (ct <= t)]
            if prev is not None and len(grp):
                dc, df1 = (cost - prev[0]) / len(grp), (f1 - prev[1]) / len(grp)
                for r in grp.itertuples():
                    out.append(
                        {
                            "dataset": ds,
                            "category": cat,
                            "embedder": emb,
                            "style": style,
                            "seed": int(seed),
                            "t": int(r.t),
                            "image_id": int(r.picked_id),
                            "label": int(r.picked_label),
                            "phase": getattr(r, "phase", ""),
                            "shared_with": len(grp),
                            "d_cost": dc,
                            "d_f1": df1,
                        }
                    )
            prev_t, prev = int(t), (cost, f1)
    inf = pd.DataFrame(out)
    if not inf.empty:
        inf["arm"] = [ARMS.get((e, s), f"{e}/{s}") for e, s in zip(inf["embedder"], inf["style"], strict=True)]
        inf["class"] = inf["category"].str.split("@").str[0]
        inf["band"] = inf["category"].str.split("@").str[1]
        # Positive = helped: a click that LOWERED cost or RAISED F1.
        inf["help_cost"] = -inf["d_cost"]
    return inf


def cell_table(base: pd.DataFrame, sky: pd.DataFrame, ts: dict, picks: pd.DataFrame) -> pd.DataFrame:
    """One row per run. A run that never found a positive has no scored steps
    (the head cannot train without a Good), and it is the review's most
    important row, so it is listed from its picks with an empty curve rather
    than dropped."""
    found = picks.groupby(RUN_KEY)["picked_label"].sum().to_dict() if not picks.empty else {}
    tail = base.sort_values("t").groupby(RUN_KEY).tail(1)
    base_last = {tuple(k): (p, r) for *k, p, r in tail[RUN_KEY + ["precision", "recall"]].itertuples(index=False)}
    rows = []
    skyd = {}
    if not sky.empty:
        for r in sky.itertuples():
            skyd[(r.dataset, r.category, r.embedder, r.style, int(r.seed))] = _f(r.oracle_cost)
    for key, s in base.groupby(RUN_KEY):
        ds, cat, emb, style, seed = key
        s = s.groupby("t")[["cost", "f1", "average_precision"]].mean()
        text = _text_for(ts, ds, cat, emb, int(seed)) or (np.nan, np.nan)
        row = {
            "arm": ARMS.get((emb, style), f"{emb}/{style}"),
            "dataset": ds,
            "category": cat,
            "class": cat.split("@")[0],
            "band": cat.split("@")[1] if "@" in cat else "",
            "seed": int(seed),
            "text_cost": text[0],
            "text_f1": text[1],
        }
        for c in CHECKPOINTS:
            at = s[s.index <= c]
            # Before a detector exists, what the user sees IS the text sort, so
            # the text score stands in -- never a gap, which would let the mean
            # at a checkpoint silently skip the runs that are still starving.
            row[f"cost_{c}"] = at["cost"].iloc[-1] if len(at) else text[0]
            row[f"f1_{c}"] = at["f1"].iloc[-1] if len(at) else text[1]
        row["final_t"] = int(s.index.max())
        row["final_cost"] = s["cost"].iloc[-1]
        row["final_f1"] = s["f1"].iloc[-1]
        row["final_ap"] = s["average_precision"].iloc[-1]
        last = base_last.get(key)
        row["final_precision"] = last[0] if last else np.nan
        row["final_recall"] = last[1] if last else np.nan
        row["ceiling_cost"] = skyd.get((ds, cat, emb, style, int(seed)), np.nan)
        row["clicks_bought"] = row["text_cost"] - row["final_cost"]
        row["headroom"] = row["final_cost"] - row["ceiling_cost"]
        row["positives_found"] = int(found.get(key, 0))
        rows.append(row)
    scored = {tuple(k) for k in base[RUN_KEY].drop_duplicates().itertuples(index=False)}
    for key, n in found.items():
        if tuple(key) in scored:
            continue
        ds, cat, emb, style, seed = key
        text = _text_for(ts, ds, cat, emb, int(seed)) or (np.nan, np.nan)
        rows.append(
            {
                "arm": ARMS.get((emb, style), f"{emb}/{style}"),
                "dataset": ds,
                "category": cat,
                "class": cat.split("@")[0],
                "band": cat.split("@")[1] if "@" in cat else "",
                "seed": int(seed),
                "text_cost": text[0],
                "text_f1": text[1],
                # No detector was ever trained, so the user is left with the
                # text sort all the way: that IS this run's score at every click.
                **{f"cost_{c}": text[0] for c in CHECKPOINTS},
                **{f"f1_{c}": text[1] for c in CHECKPOINTS},
                "final_cost": text[0],
                "final_f1": text[1],
                "clicks_bought": 0.0,
                "ceiling_cost": skyd.get((ds, cat, emb, style, int(seed)), np.nan),
                "positives_found": int(n),
                "never_trained": True,
            }
        )
    out = pd.DataFrame(rows)
    out["never_trained"] = (
        out.get("never_trained", pd.Series(False, index=out.index)).astype("boolean").fillna(False).astype(bool)
    )
    out["headroom"] = out["final_cost"] - out["ceiling_cost"]
    return out


def curves(cells: pd.DataFrame, base: pd.DataFrame, horizon: int = 150) -> pd.DataFrame:
    """Every run on a common click grid 0..horizon, as the USER would see it.

    Click 0 is the text-only score; until the first scored click the user still
    sees the text sort, so it carries forward; after that the last scored value
    carries forward. A run that never trained stays at its text score.
    """
    by_run = {tuple(k): g.groupby("t")[["cost", "f1"]].mean() for k, g in base.groupby(RUN_KEY)}
    grid = np.arange(0, horizon + 1)
    rows = []
    for r in cells.itertuples():
        key = (r.dataset, r.category, _emb_of(r.arm), _style_of(r.arm), int(r.seed))
        s = by_run.get(key)
        cost = np.full(len(grid), r.text_cost, dtype=float)
        f1 = np.full(len(grid), r.text_f1, dtype=float)
        if s is not None and len(s):
            idx = np.searchsorted(s.index.to_numpy(), grid, side="right") - 1
            have = idx >= 0
            cost[have] = s["cost"].to_numpy()[idx[have]]
            f1[have] = s["f1"].to_numpy()[idx[have]]
            cost[0], f1[0] = r.text_cost, r.text_f1
        rows.append(
            pd.DataFrame({"arm": r.arm, "category": r.category, "seed": r.seed, "t": grid, "cost": cost, "f1": f1})
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


_ARM_OF = {v: k for k, v in ARMS.items()}


def _emb_of(arm: str) -> str:
    return _ARM_OF[arm][0]


def _style_of(arm: str) -> str:
    return _ARM_OF[arm][1]


def roll_up(inf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if inf.empty:
        return inf, inf
    inf = inf.assign(when=np.where(inf["t"] <= EARLY, "early", np.where(inf["t"] > LATE, "late", "mid")))
    agg = dict(
        n_obs=("help_cost", "size"),
        n_runs=("seed", "size"),
        help_cost=("help_cost", "mean"),
        help_cost_sd=("help_cost", "std"),
        d_f1=("d_f1", "mean"),
    )
    img = inf.groupby("image_id").agg(
        n_obs=("help_cost", "size"),
        n_detectors=("category", "nunique"),
        n_classes=("class", "nunique"),
        clicked_as_positive=("label", "sum"),
        help_cost=("help_cost", "mean"),
        help_cost_sd=("help_cost", "std"),
        d_f1=("d_f1", "mean"),
    )
    for w in ("early", "late"):
        sub = inf[inf["when"] == w].groupby("image_id")["help_cost"].agg(["mean", "size"])
        img[f"help_{w}"] = sub["mean"]
        img[f"n_{w}"] = sub["size"]
    img = img.reset_index().sort_values("help_cost", ascending=False)
    det = inf.groupby(["image_id", "arm", "class", "label"]).agg(**{k: v for k, v in agg.items() if k != "n_runs"})
    return img, det.reset_index()


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """A GitHub markdown table, without pandas' optional `tabulate` dependency."""
    d = df.reset_index() if index else df
    head = [str(c) for c in d.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def summary(cells: pd.DataFrame, img: pd.DataFrame, det: pd.DataFrame, out: Path) -> None:
    lines = ["# State of the App -- summary tables", ""]
    if not cells.empty and cells["never_trained"].any():
        nt = cells[cells["never_trained"]][["arm", "category", "positives_found", "text_cost", "ceiling_cost"]]
        lines += ["## Runs that never found a positive (no detector was ever trained)", "", _md(nt, index=False), ""]
    if not cells.empty:
        lines += ["## Per path, over all cells", ""]
        cols = [
            "text_cost",
            "cost_25",
            "cost_50",
            "final_cost",
            "ceiling_cost",
            "text_f1",
            "f1_50",
            "final_f1",
            "final_precision",
            "final_recall",
        ]
        g = cells.groupby("arm")[cols].mean().round(3)
        lines += [_md(g), ""]
        lines += ["## Per path and band", ""]
        lines += [
            _md(
                cells.groupby(["arm", "band"])[["text_cost", "final_cost", "ceiling_cost", "final_f1"]].mean().round(3)
            ),
            "",
        ]
        for arm, a in cells.groupby("arm"):
            by = a.groupby("class")[["text_cost", "final_cost", "ceiling_cost", "final_f1", "headroom"]].mean()
            lines += [
                f"## {arm}: the 10 hardest classes (final cost)",
                "",
                _md(by.sort_values("final_cost", ascending=False).head(10).round(3)),
                "",
            ]
            lines += [f"## {arm}: the 10 easiest classes", "", _md(by.sort_values("final_cost").head(10).round(3)), ""]
            lines += [
                f"## {arm}: most headroom (final minus full-label ceiling)",
                "",
                _md(by.sort_values("headroom", ascending=False).head(10).round(3)),
                "",
            ]
    if not img.empty:
        seen = img[img["n_obs"] >= 3]
        lines += [f"## Images clicked 3+ times ({len(seen)} of {len(img)})", ""]
        lines += ["### Most helpful (mean cost removed per click)", "", _md(seen.head(15).round(4), index=False), ""]
        lines += ["### Most harmful", "", _md(seen.tail(15).iloc[::-1].round(4), index=False), ""]
        both = seen.dropna(subset=["help_early", "help_late"])
        flips = both[np.sign(both["help_early"]) != np.sign(both["help_late"])]
        lines += [f"### Early/late sign flips ({len(flips)} images seen both early and late)", ""]
        lines += [_md(flips.sort_values("n_obs", ascending=False).head(15).round(4), index=False), ""]
    (out / "summary.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None, help="text_baseline.csv (the click-0 score)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base, sky, picks = load(args.exp)
    if base.empty:
        raise SystemExit(f"no cells under {args.exp}/results/cells")
    ts = text_scores(args.baseline)
    cells = cell_table(base, sky, ts, picks)
    inf = attribute(base, picks, ts)
    img, det = roll_up(inf)
    cells.to_csv(args.out / "cells.csv", index=False)
    curves(cells, base).to_csv(args.out / "curves.csv", index=False)
    inf.to_csv(args.out / "influence.csv", index=False)
    img.to_csv(args.out / "images.csv", index=False)
    det.to_csv(args.out / "image_detector.csv", index=False)
    summary(cells, img, det, args.out)
    print(f"{len(cells)} runs, {len(inf)} credited clicks, {len(img)} images -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
