#!/usr/bin/env python3
"""#4936: the calibration fold models' scale against the final model's, per class, on the withheld half.

Reads the test-score snapshots ``fold_scale_4936.sh`` runs save (``CALIB_SAVE_FOLD_SCALES``): at each listed click,
the withheld half under the final model and under every live fold model, the sim set under every model with its
truth and votes, and the vote behind each held-out score. Nothing here re-runs a session.

Per snapshot and fold *k*, on the withheld half (images no model has seen), in the labels line's logit units:

* **the true map**: for positives and negatives separately, the final model's logit against fold *k*'s on the same
  images: mean shift, affine fit (slope, R^2), rank correlation, spread ratio; and for the negatives the line cuts
  among (the top 1% of the withheld half by fold *k*'s score).
* **candidate maps** from fold *k*'s scale to the final model's, judged against it: ``none`` (today), ``q`` (the
  quantile transfer on the unvoted remainder, #4910), ``qt`` (the same, tails kept), ``z`` (the owner's robust z:
  median and MAD of each haystack's logits), ``vs`` (label-aware: per class, the mean final-minus-fold logit over
  the votes fold *k* trained on, both in-sample), ``ho`` (label-aware: per class, over the votes fold *k* held
  out, the final model in-sample), and ``oracle`` (per class, the affine map fitted on the withheld half's own
  positives and top negatives: the truth, not deployable). Each is read three ways: its error on the withheld
  positives themselves (scale alone), the Good component it gives the labels line against the withheld positives'
  own mean (scale and selection), and the F-beta of Find's line on the withheld half from that component, on the
  same head (the run's own line is ``none``).

  fold_scale_4936.py measure --results DIR [--results DIR ...] --out DIR [--procs N]
  fold_scale_4936.py report --out DIR [--docs DIR]

``measure`` writes ``snapshots.csv.gz`` (one row per snapshot); ``report`` the tables and figures from it.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

MAPS = ("none", "q", "qt", "z", "vs", "ho", "oracle", "oracle_pos", "oracle_neg")
WHOLE_MAPS = MAPS[:7]  # the oracle's halves differ from it only in F-beta: on its own class each is the oracle
TOP_SHARE = 0.01  # the negatives the line cuts among: the top 1% of the withheld half by the fold model's score
ORACLE_NEG_SHARE = 0.02  # the oracle's negative map is fitted on the top 2% of the withheld half's negatives
EMBEDDER_STYLE = "siglip/whole_image"  # Binary Photo's snapshot keys
OTHER_BETAS = (0.25, 4.0)  # the other presets, read on the same heads


def beta_tag(b: float) -> str:
    return {0.25: "025", 4.0: "4", 1.0: "1"}[float(b)]


def _logit(p):
    from vtscore.training.thresholds.labels_line import _logit as lg

    return lg(p)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _ols(x, y):
    """``(slope, intercept, r2)`` of y on x; NaN with fewer than 3 points or no spread."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) == 0:
        return math.nan, math.nan, math.nan
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 else math.nan
    return float(b), float(a), float(r * r)


def _spearman(x, y):
    from scipy.stats import spearmanr

    if len(x) < 3:
        return math.nan
    return float(spearmanr(x, y).statistic)


def _mad(x):
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def load_snapshots(npz_path: str) -> dict[str, dict]:
    """``{key: snapshot}`` for every ``t<N>`` snapshot carrying the fold-scale arrays, arrays and JSON meta merged."""
    z = np.load(npz_path, allow_pickle=False)
    snaps: dict[str, dict] = {}
    for name in z.files:
        key, field = name.rsplit("/", 1)
        snaps.setdefault(key, {})[field] = z[name]
    out = {}
    for key, snap in snaps.items():
        if not re.search(r"/t\d+$", key) or "fold_test_scores" not in snap:
            continue
        meta = json.loads(str(snap.pop("meta")))
        out[key] = {**meta, **snap}
    return out


class Fold:
    """Fold *k* of one snapshot: its logits of the withheld half, the sim set, and its votes."""

    def __init__(self, snap: dict, k: int):
        self.k = k
        self.test = _logit(snap["fold_test_scores"][k])
        self.sim = _logit(snap["fold_sim_scores"][k])
        sel = snap["fold_index"] == k
        self.held_scores = np.asarray(snap["fold_scores"][sel], dtype=np.float64)
        self.held_labels = np.asarray(snap["fold_labels"][sel], dtype=np.int8)
        self.held_ids = np.asarray(snap["fold_ids"][sel], dtype=np.int64)


class Snapshot:
    def __init__(self, snap: dict):
        self.raw = snap
        self.t = int(snap["t"])
        self.beta = float(snap["beta"])
        self.y = np.asarray(snap["labels"], dtype=np.int8)
        self.scores = np.asarray(snap["scores"], dtype=np.float64)
        self.final = _logit(self.scores)
        self.sim_ids = np.asarray(snap["sim_ids"], dtype=np.int64)
        self.sim_scores = np.asarray(snap["sim_scores"], dtype=np.float64)
        self.sim_final = _logit(self.sim_scores)
        self.sim_labels = np.asarray(snap["sim_labels"], dtype=np.int8)
        self.sim_vote = np.asarray(snap["sim_vote"], dtype=np.int8)
        self.k = int(snap["n_live_folds"])
        self.folds = [Fold(snap, k) for k in range(self.k)]
        unvoted = self.sim_vote < 0
        self.hay_mask = unvoted if bool(snap["fold_sim_excluding"]) else np.ones(unvoted.size, dtype=bool)
        self.pos = self.y == 1
        self.at = {int(c): i for i, c in enumerate(self.sim_ids)}

    def labels_dict(self) -> dict[int, bool]:
        return {int(c): bool(v == 1) for c, v in zip(self.sim_ids, self.sim_vote) if v >= 0}

    def orderings(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(f.held_scores, f.held_labels.astype(np.float64)) for f in self.folds]


# --- the maps: each returns, for fold f, a function from fold logits (and a class) to final-scale logits ------------


def _quantile_map(s: Snapshot, f: Fold, tails: bool):
    from vtscore.training.thresholds import transfer_fold_orderings

    hay_k = _sig(f.sim[s.hay_mask])
    hay_f = s.sim_scores[s.hay_mask]

    def m(x, _cls):
        p = _sig(x)
        moved = transfer_fold_orderings([(p, np.zeros(p.size))], [hay_k], hay_f, tails=tails)[0][0]
        return _logit(moved)

    return m


def _z_map(s: Snapshot, f: Fold):
    mk, sk = _mad(f.sim[s.hay_mask])
    mf, sf = _mad(s.sim_final[s.hay_mask])
    scale = sf / sk if sk > 0 else 1.0
    return lambda x, _cls: mf + (np.asarray(x) - mk) * scale


def _vote_shift(s: Snapshot, f: Fold, held: bool):
    """Per class, the mean final-minus-fold logit over fold *f*'s held-out votes (*held*) or its training votes."""
    held_set = set(int(v) for v in f.held_ids if v >= 0)
    shift = {}
    for cls in (1, 0):
        idx = [i for i, (c, v) in enumerate(zip(s.sim_ids, s.sim_vote)) if v == cls and ((int(c) in held_set) == held)]
        idx = np.asarray(idx, dtype=np.int64)
        shift[cls] = float(np.mean(s.sim_final[idx] - f.sim[idx])) if idx.size else 0.0
    return lambda x, cls: np.asarray(x) + shift[cls]


def _oracle_map(s: Snapshot, f: Fold):
    """Per class, the affine map the withheld half itself gives (positives; the top negatives)."""
    fit = {}
    b, a, _ = _ols(f.test[s.pos], s.final[s.pos])
    fit[1] = (b, a) if np.isfinite(b) else (1.0, float(np.mean(s.final[s.pos] - f.test[s.pos])))
    neg = ~s.pos
    cut = np.quantile(f.test[neg], 1.0 - ORACLE_NEG_SHARE)
    top = neg & (f.test >= cut)
    b, a, _ = _ols(f.test[top], s.final[top])
    fit[0] = (b, a) if np.isfinite(b) else (1.0, 0.0)
    return lambda x, cls: fit[cls][1] + fit[cls][0] * np.asarray(x)


def _one_class(m, cls: int):
    """*m* on class *cls* only; the other class keeps its fold logits."""
    return lambda x, c: m(x, c) if c == cls else np.asarray(x)


def build_maps(s: Snapshot) -> dict[str, list]:
    maps: dict[str, list] = {"none": [lambda x, _c: np.asarray(x)] * s.k}
    maps["q"] = [_quantile_map(s, f, tails=False) for f in s.folds]
    maps["qt"] = [_quantile_map(s, f, tails=True) for f in s.folds]
    maps["z"] = [_z_map(s, f) for f in s.folds]
    # The vote maps need to know which votes each fold held out: unnamed on the grouped (Region) path.
    if all((f.held_ids >= 0).all() for f in s.folds):
        maps["vs"] = [_vote_shift(s, f, held=False) for f in s.folds]
        maps["ho"] = [_vote_shift(s, f, held=True) for f in s.folds]
    maps["oracle"] = [_oracle_map(s, f) for f in s.folds]
    # The oracle's two halves: the positives' map on the Goods alone, the negatives' on the Bads alone.
    maps["oracle_pos"] = [_one_class(m, 1) for m in maps["oracle"]]
    maps["oracle_neg"] = [_one_class(m, 0) for m in maps["oracle"]]
    return maps


def mapped_orderings(s: Snapshot, fmaps: list) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for f, m in zip(s.folds, fmaps):
        x = _logit(f.held_scores)
        y = f.held_labels
        mx = np.empty_like(x)
        for cls in (1, 0):
            sel = y == cls
            if sel.any():
                mx[sel] = m(x[sel], cls)
        out.append((_sig(mx), y.astype(np.float64)))
    return out


def find_fbeta(s: Snapshot, orderings, labels: dict[int, bool]) -> tuple[float, int, float, object, dict]:
    """``(F-beta, kept, threshold, line, F-beta at OTHER_BETAS)`` of Find's line on the withheld half from *orderings*, as the harness draws it."""
    from vtscore.eval import calibration_metrics as cm
    from vtscore.training.thresholds.labels_line import fit_labels_line

    line = fit_labels_line(orderings or None, s.sim_scores, s.sim_ids.tolist(), labels)
    if line is None:
        return math.nan, -1, math.nan, None, {}
    on_test = line.on_corpus(s.scores)
    thr = float(on_test.threshold(s.beta))
    # The same head's Find line at the other presets (a same-head reading: a session at another preset differs).
    other = {}
    for b in OTHER_BETAS:
        t_b = float(on_test.threshold(b))
        other[b] = float(cm.fbeta_metrics(s.scores, s.y, t_b, b)["fbeta"])
    return (
        float(cm.fbeta_metrics(s.scores, s.y, thr, s.beta)["fbeta"]),
        int(np.count_nonzero(s.scores >= thr)),
        thr,
        line,
        other,
    )


def measure(snap: dict) -> dict:
    """One row: the true map, every candidate's error, Good component and same-head F-beta, for one snapshot."""
    from vtscore.training.thresholds.labels_line import (
        HEAD_COUNT_GOODS,
        _line_on,
        _pooled_class_model,
    )

    from vtscore.detectors.label_quota import query_holds

    s = Snapshot(snap)
    row: dict = {
        "t": s.t,
        "n_folds": s.k,
        "n_test": int(s.y.size),
        "n_pos_test": int(s.pos.sum()),
        "n_good": int((s.sim_vote == 1).sum()),
        "n_bad": int((s.sim_vote == 0).sum()),
        "n_pos_remainder": int(((s.sim_vote < 0) & (s.sim_labels == 1)).sum()),
        "excluding": bool(snap["fold_sim_excluding"]),
        # Under the typed query's hold (#4736) Find shows the query's own set, not this head's line.
        "query_hold": bool(query_holds(int((s.sim_vote >= 0).sum()), 0, patch_space=False)),
        "mu_true": float(s.final[s.pos].mean()) if s.pos.any() else math.nan,
        "sd_true": float(s.final[s.pos].std(ddof=1)) if s.pos.sum() > 1 else math.nan,
    }
    meta = snap.get("model") or {}
    labels = s.labels_dict()
    if s.k == 0 or not s.pos.any():
        row["line_src"] = "no_folds"
        return row
    neg = ~s.pos

    # --- the true map, per fold, then averaged over the folds ---
    acc: dict[str, list[float]] = {}

    def put(name, v):
        acc.setdefault(name, []).append(v)

    for f in s.folds:
        put("dpos", float(np.mean(s.final[s.pos] - f.test[s.pos])))
        put("dpos_med", float(np.median(s.final[s.pos] - f.test[s.pos])))
        b, a, r2 = _ols(f.test[s.pos], s.final[s.pos])
        put("slope_pos", b)
        put("icpt_pos", a)
        put("r2_pos", r2)
        put("rho_pos", _spearman(f.test[s.pos], s.final[s.pos]))
        put("sdratio_pos", float(np.std(s.final[s.pos]) / np.std(f.test[s.pos])) if s.pos.sum() > 1 else math.nan)
        put("dneg", float(np.mean(s.final[neg] - f.test[neg])))
        b, a, r2 = _ols(f.test[neg], s.final[neg])
        put("slope_neg", b)
        put("r2_neg", r2)
        put("rho_neg", _spearman(f.test[neg], s.final[neg]))
        top = neg & (f.test >= np.quantile(f.test, 1.0 - TOP_SHARE))
        put("dnegtop", float(np.mean(s.final[top] - f.test[top])) if top.any() else math.nan)
        put("n_negtop", float(top.sum()))
        rem = (s.sim_vote < 0) & (s.sim_labels == 1)
        put("drem_pos", float(np.mean(s.sim_final[rem] - f.sim[rem])) if rem.any() else math.nan)
        # The held-out and training Goods' own final-minus-fold logits (the final model trained on both).
        held_set = set(int(v) for v in f.held_ids if v >= 0)
        if not (f.held_ids >= 0).all():  # the grouped path names no held-out vote: no split to read
            for name in ("dho_good", "dtr_good", "dho_bad", "dtr_bad"):
                put(name, math.nan)
            held_set = None
        if held_set is not None:
            hg = np.array([s.at[v] for v in held_set if s.sim_vote[s.at[v]] == 1], dtype=np.int64)
            tg = np.array(
                [i for i in np.flatnonzero(s.sim_vote == 1) if int(s.sim_ids[i]) not in held_set], dtype=np.int64
            )
            put("dho_good", float(np.mean(s.sim_final[hg] - f.sim[hg])) if hg.size else math.nan)
            put("dtr_good", float(np.mean(s.sim_final[tg] - f.sim[tg])) if tg.size else math.nan)
            hb = np.array([s.at[v] for v in held_set if s.sim_vote[s.at[v]] == 0], dtype=np.int64)
            tb = np.array(
                [i for i in np.flatnonzero(s.sim_vote == 0) if int(s.sim_ids[i]) not in held_set], dtype=np.int64
            )
            put("dho_bad", float(np.mean(s.sim_final[hb] - f.sim[hb])) if hb.size else math.nan)
            put("dtr_bad", float(np.mean(s.sim_final[tb] - f.sim[tb])) if tb.size else math.nan)
        # Fold k's own held-out Goods against the withheld positives, both on fold k's scale (selection, #4937).
        hp = f.held_labels == 1
        put("sel_fold", float(np.mean(_logit(f.held_scores[hp])) - np.mean(f.test[s.pos])) if hp.any() else math.nan)
    for name, vals in acc.items():
        row[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan

    # --- the Good component today, and its error split into scale and selection (exact, folds pooled) ---
    w = np.array([float((f.held_labels == 1).sum()) for f in s.folds])
    if w.sum() > 0:
        w = w / w.sum()
        row["gap_scale"] = float(sum(wk * np.mean(f.test[s.pos] - s.final[s.pos]) for wk, f in zip(w, s.folds)))
        row["gap_selection"] = float(
            sum(
                wk * (np.mean(_logit(f.held_scores[f.held_labels == 1])) - np.mean(f.test[s.pos]))
                for wk, f in zip(w, s.folds)
                if wk > 0
            )
        )

    # --- every candidate map ---
    maps = build_maps(s)
    for name, fmaps in maps.items():
        bias, sq, sds, bias_nt = [], [], [], []
        for f, m in zip(s.folds, fmaps):
            mp = m(f.test[s.pos], 1)
            bias.append(float(np.mean(mp - s.final[s.pos])))
            sq.append(float(np.mean((mp - s.final[s.pos]) ** 2)))
            sds.append(float(np.std(mp) / np.std(s.final[s.pos])) if s.pos.sum() > 1 else math.nan)
            top = neg & (f.test >= np.quantile(f.test, 1.0 - TOP_SHARE))
            if top.any():
                bias_nt.append(float(np.mean(m(f.test[top], 0) - s.final[top])))
        row[f"{name}_bias_pos"] = float(np.mean(bias))
        row[f"{name}_rmse_pos"] = float(math.sqrt(np.mean(sq)))
        row[f"{name}_sdratio_pos"] = float(np.nanmean(sds)) if np.isfinite(sds).any() else math.nan
        row[f"{name}_bias_negtop"] = float(np.mean(bias_nt)) if bias_nt else math.nan
        orderings = mapped_orderings(s, fmaps)
        model = _pooled_class_model(orderings)
        if model is not None:
            row[f"{name}_mu_pos"] = model.mu_pos
            row[f"{name}_mu_neg"] = model.mu_neg
            row[f"{name}_sigma"] = model.sigma
            row[f"{name}_gap"] = model.mu_pos - row["mu_true"]
        fb, kept, thr, line, other = find_fbeta(s, orderings, labels)
        row[f"{name}_fbeta"] = fb
        row[f"{name}_kept"] = kept
        for b, v in other.items():
            row[f"{name}_fbeta_b{beta_tag(b)}"] = v
        if name == "none":
            row["line_src"] = (
                "none" if line is None else ("in_sample" if model is None or model.mu_pos <= model.mu_neg else "folds")
            )
            row["head_count"] = bool(model is not None and model.n_pos < HEAD_COUNT_GOODS)
            # The replay must be the run's own line: the class model and the Train cut it drew.
            if line is not None and meta:
                row["chk_mu_pos"] = abs(line.model.mu_pos - float(meta.get("mu_pos", math.nan)))
                row["chk_train_thr"] = abs(float(line.threshold(s.beta)) - float(snap["train_threshold"]))
    # The ceiling on fixing the Good component outright: its mean set to the withheld positives' own.
    model = _pooled_class_model(s.orderings())
    if model is not None and model.mu_pos > model.mu_neg:
        from vtscore.eval import calibration_metrics as cm

        fixed = replace(model, mu_pos=row["mu_true"])
        line = _line_on(fixed, s.scores, None, None)
        if line is not None:
            thr = float(line.threshold(s.beta))
            row["truemu_fbeta"] = float(cm.fbeta_metrics(s.scores, s.y, thr, s.beta)["fbeta"])
            row["truemu_kept"] = int(np.count_nonzero(s.scores >= thr))
    # The best cut on the same ranking: what any line could reach.
    tp = np.cumsum(s.y[np.argsort(-s.scores, kind="stable")])
    k = np.arange(1, s.scores.size + 1)
    b2 = s.beta**2
    fb = (1 + b2) * tp / (b2 * s.pos.sum() + k)
    row["best_fbeta"] = float(fb.max())
    row["best_kept"] = int(k[int(np.argmax(fb))])
    return row


def click_of(t: int, listed: list[int]) -> int | None:
    """The listed click a snapshot taken at step *t* reads for: the largest one at or below it."""
    below = [c for c in listed if c <= t]
    return max(below) if below else None


def _cell_rows(args) -> list[dict]:
    """One row per distinct snapshot step of a cell.

    run_cells.py keeps, for each listed click, the first ordinary step at or past it, so a session whose trained
    head first scores the withheld half at step 128 files that one step under every click from 10 to 125. Each
    step is measured once and read for the largest listed click at or below it (``click``).
    """
    npz, ident = args
    rows = []
    try:
        snaps = load_snapshots(npz)
    except Exception as exc:  # a truncated file: report, do not stop the pool
        return [{**ident, "error": f"{type(exc).__name__}: {exc}"}]
    snaps = {k: v for k, v in snaps.items() if k.startswith(EMBEDDER_STYLE)}
    listed = sorted(int(k.rsplit("/t", 1)[1]) for k in snaps)
    by_step: dict[int, dict] = {}
    for snap in snaps.values():
        by_step.setdefault(int(snap["t"]), snap)
    base = _base_rows(npz)
    for t, snap in sorted(by_step.items()):
        row = {**ident, "click": click_of(t, listed), **measure(snap)}
        # The run's own row at this step: its Find F-beta is the replay's ``none``, its shadow's the replay's ``q``.
        if t in base.index:
            row["chk_row_fbeta"] = abs(float(base.at[t, "fbeta"]) - row.get("none_fbeta", math.nan))
            row["chk_row_q"] = abs(float(base.at[t, "shadow_fbeta"]) - row.get("q_fbeta", math.nan))
        rows.append(row)
    return rows


def _base_rows(npz: str) -> pd.DataFrame:
    """The cell's base calibration rows (max pooling, no variant, no check), indexed by step."""
    main = npz.replace("__testscores.npz", ".csv.gz")
    if not os.path.exists(main):
        main = main[:-3]
    df = pd.read_csv(main, usecols=["t", "phase", "pool_variant", "gmm_variant", "fbeta", "shadow_fbeta"])
    df = df[(df["pool_variant"] == "max") & df["gmm_variant"].isna() & (df["phase"] != "check")]
    return df.drop_duplicates("t", keep="last").set_index("t")


def _ident(cells_dir: str, npz: str) -> dict:
    idx = int(re.search(r"task_(\d+)__", os.path.basename(npz)).group(1))
    main = os.path.join(cells_dir, f"task_{idx:04d}.csv.gz")
    if not os.path.exists(main):
        main = main[:-3]
    head = pd.read_csv(main, usecols=["category", "seed"], nrows=1)
    cat = str(head["category"].iloc[0])
    cls, _, band = cat.partition("@")
    return {"idx": idx, "cell": cat, "class": cls, "band": band, "seed": int(head["seed"].iloc[0])}


def cmd_measure(a) -> None:
    jobs = []
    for res in a.results:
        cells = os.path.join(res, "cells")
        for npz in sorted(glob.glob(os.path.join(cells, "task_*__testscores.npz"))):
            jobs.append((npz, {"run": os.path.basename(os.path.dirname(res.rstrip("/"))), **_ident(cells, npz)}))
    print(f"{len(jobs)} cells", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for i, out in enumerate(ex.map(_cell_rows, jobs, chunksize=1)):
            rows.extend(out)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "snapshots.csv.gz"), index=False)
    bad = df.get("error")
    print(f"wrote {len(df)} snapshot rows; errors: {0 if bad is None else int(bad.notna().sum())}")
    for col in ("chk_mu_pos", "chk_train_thr", "chk_row_fbeta", "chk_row_q"):
        if col in df:
            print(f"{col}: max {df[col].max():.2e}, n>1e-6 {int((df[col] > 1e-6).sum())}")


# --- report ------------------------------------------------------------------------------------------------------

WINDOWS = [(10, 10), (25, 25), (50, 50), (75, 75), (100, 100), (125, 125), (150, 150)]


def by_class(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean and SE over classes (a class's bands and seeds averaged first: they are not independent)."""
    if col not in df:
        return math.nan, math.nan, 0
    per = df.groupby("class")[col].mean().dropna()
    if per.empty:
        return math.nan, math.nan, 0
    se = float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else math.nan
    return float(per.mean()), se, int(len(per))


def fmt(m: float, se: float, digits: int = 2) -> str:
    if not np.isfinite(m):
        return "-"
    return f"{m:+.{digits}f} (± {se:.{digits}f})" if np.isfinite(se) else f"{m:+.{digits}f}"


def cmd_report(a) -> None:
    df = pd.read_csv(os.path.join(a.out, "snapshots.csv.gz"))
    df = df[df.get("error").isna()] if "error" in df else df
    folded = df[df["line_src"] == "folds"].copy()
    tabs = Path(a.docs) if a.docs else Path(a.out) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    clicks = sorted(df["click"].dropna().unique())

    def table(cols: list[tuple[str, str]], frame: pd.DataFrame, digits=2) -> pd.DataFrame:
        out = []
        for t in clicks:
            sub = frame[frame["click"] == t]
            rec = {"click": t, "snapshots": len(sub)}
            for label, col in cols:
                m, se, _ = by_class(sub, col)
                rec[label] = fmt(m, se, digits)
                rec[label + "_mean"] = m
                rec[label + "_se"] = se
            out.append(rec)
        return pd.DataFrame(out)

    true_map = table(
        [
            ("positives: final - fold", "dpos"),
            ("positives: slope", "slope_pos"),
            ("positives: R2", "r2_pos"),
            ("positives: rank corr.", "rho_pos"),
            ("positives: spread ratio", "sdratio_pos"),
            ("negatives: final - fold", "dneg"),
            ("negatives: slope", "slope_neg"),
            ("top negatives: final - fold", "dnegtop"),
            ("remainder positives: final - fold", "drem_pos"),
            ("training Goods: final - fold", "dtr_good"),
            ("held-out Goods: final - fold", "dho_good"),
        ],
        folded,
    )
    true_map.to_csv(tabs / "true_map.csv", index=False)
    maps = table(
        [(f"{m}: bias on positives", f"{m}_bias_pos") for m in MAPS]
        + [(f"{m}: bias on top negatives", f"{m}_bias_negtop") for m in MAPS]
        + [(f"{m}: rmse on positives", f"{m}_rmse_pos") for m in MAPS],
        folded,
    )
    maps.to_csv(tabs / "maps_on_positives.csv", index=False)
    folded["sel_plus_scale"] = folded["gap_scale"] + folded["gap_selection"]
    comp = table(
        [(f"{m}: Good mean - truth", f"{m}_gap") for m in MAPS]
        + [("scale", "gap_scale"), ("selection", "gap_selection")],
        folded,
    )
    comp.to_csv(tabs / "good_component.csv", index=False)
    for m in MAPS:
        if f"{m}_fbeta" in folded:
            folded[f"{m}_dF"] = folded[f"{m}_fbeta"] - folded["none_fbeta"]
    folded["truemu_dF"] = folded["truemu_fbeta"] - folded["none_fbeta"]
    fb = table(
        [("today's line F-beta", "none_fbeta"), ("best cut F-beta", "best_fbeta")]
        + [(f"{m}: F-beta gain", f"{m}_dF") for m in MAPS if m != "none"]
        + [("Good mean = truth: F-beta gain", "truemu_dF")],
        folded,
        digits=3,
    )
    fb.to_csv(tabs / "same_head_fbeta.csv", index=False)
    for b in OTHER_BETAS:
        tag = beta_tag(b)
        for m in MAPS:
            if f"{m}_fbeta_b{tag}" in folded:
                folded[f"{m}_dF_b{tag}"] = folded[f"{m}_fbeta_b{tag}"] - folded[f"none_fbeta_b{tag}"]
        table(
            [(f"today's line F-beta at beta {b:g}", f"none_fbeta_b{tag}")]
            + [(f"{m}: gain", f"{m}_dF_b{tag}") for m in MAPS if m != "none"],
            folded,
            digits=3,
        ).to_csv(tabs / f"same_head_fbeta_b{tag}.csv", index=False)
    band_rows = []
    for (band, c), sub in folded.groupby(["band", "click"]):
        rec = {"band": band, "click": c, "snapshots": len(sub)}
        for label, col in (
            ("positives: final - fold", "dpos"),
            ("top negatives", "dnegtop"),
            ("Good mean - truth", "none_gap"),
            ("scale", "gap_scale"),
            ("selection", "gap_selection"),
        ):
            m, se, _ = by_class(sub, col)
            rec[label] = fmt(m, se)
        band_rows.append(rec)
    pd.DataFrame(band_rows).to_csv(tabs / "by_band.csv", index=False)
    # Does a deployable estimate track the true shift? Across snapshots, by click.
    track = []
    for c, sub in folded.groupby("click"):
        rec = {"click": c}
        for a_col, b_col in (
            ("dtr_good", "dpos"),
            ("dho_good", "dpos"),
            ("dtr_bad", "dnegtop"),
            ("dho_bad", "dnegtop"),
        ):
            ok = sub[[a_col, b_col]].dropna()
            rec[f"corr({a_col},{b_col})"] = round(float(ok[a_col].corr(ok[b_col])), 2) if len(ok) > 2 else math.nan
            rec[f"mean {a_col} - {b_col}"] = round(float((ok[a_col] - ok[b_col]).mean()), 3) if len(ok) else math.nan
        track.append(rec)
    pd.DataFrame(track).to_csv(tabs / "vote_estimates.csv", index=False)
    # Per session, not per class: how far single snapshots stray from the means above.
    spread = []
    for c, sub in folded.groupby("click"):
        rec = {"click": c, "snapshots": len(sub)}
        for col in ("dpos", "dnegtop", "gap_scale", "gap_selection", "none_gap"):
            q10, q50, q90 = sub[col].quantile([0.1, 0.5, 0.9])
            rec[col] = f"{q10:+.2f} / {q50:+.2f} / {q90:+.2f}"
        rec["share |dpos| < 0.1"] = round(float((sub["dpos"].abs() < 0.1).mean()), 2)
        spread.append(rec)
    pd.DataFrame(spread).to_csv(tabs / "spread_10_50_90.csv", index=False)
    # The replay against the run's own rows: every Find F-beta mismatch should be a step under the typed query's
    # hold, where the row reports the query's set.
    if "chk_row_fbeta" in df:
        off = df[df["chk_row_fbeta"] > 1e-6]
        print(
            f"replay vs run: {len(off)} Find F-beta mismatches, {int(off['query_hold'].astype(bool).sum())} under "
            f"the hold; max off the hold {df.loc[~df['query_hold'].astype(bool), 'chk_row_fbeta'].max():.1e}; "
            f"quantile shadow max {df['chk_row_q'].max():.1e}"
        )
    src = df.groupby("click")["line_src"].value_counts().unstack(fill_value=0)
    src.to_csv(tabs / "line_source.csv")
    print(src)
    figures(df, folded, Path(a.figs) if a.figs else Path(a.out) / "figures")
    for name, t in (("true map", true_map), ("maps", maps), ("good component", comp), ("same-head F-beta", fb)):
        print(f"\n== {name}")
        print(t[[c for c in t.columns if not c.endswith(("_mean", "_se"))]].to_string(index=False))


MAP_LABEL = {
    "none": "today (no map)",
    "q": "quantile",
    "qt": "quantile, tails kept",
    "z": "robust z",
    "vs": "training votes' shift",
    "ho": "held-out votes' shift",
    "oracle": "oracle (withheld truth)",
    "oracle_pos": "oracle, Goods only",
    "oracle_neg": "oracle, Bads only",
}
MAP_COLOR = {
    "none": "#333333",
    "q": "#1f77b4",
    "qt": "#17becf",
    "z": "#ff7f0e",
    "vs": "#2ca02c",
    "ho": "#d62728",
    "oracle": "#9467bd",
    "oracle_pos": "#c5b0d5",
    "oracle_neg": "#7b4173",
}


def _curve(frame: pd.DataFrame, col: str, clicks: list) -> tuple[np.ndarray, np.ndarray]:
    m, se = [], []
    for c in clicks:
        mm, ss, _ = by_class(frame[frame["click"] == c], col)
        m.append(mm)
        se.append(ss)
    return np.array(m), np.array(se)


def figures(df: pd.DataFrame, folded: pd.DataFrame, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    clicks = sorted(folded["click"].dropna().unique())
    x = np.array(clicks, dtype=float)

    def band(ax, col, label, color, frame=folded, ls="-"):
        m, se = _curve(frame, col, clicks)
        ax.plot(x, m, ls, color=color, label=label, marker="o", ms=3)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)

    # 1. The true map on the withheld half.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    band(ax, "dpos", "positives", "#2ca02c")
    band(ax, "dnegtop", "top 1% negatives", "#d62728")
    band(ax, "dneg", "all negatives", "#7f7f7f")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("final minus fold logit, same images")
    ax.set_xlabel("click")
    ax.set_ylabel("logit")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    band(ax, "slope_pos", "positives", "#2ca02c")
    band(ax, "slope_neg", "negatives", "#7f7f7f")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_title("slope of final on fold")
    ax.set_xlabel("click")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    band(ax, "rho_pos", "positives: rank corr.", "#2ca02c")
    band(ax, "r2_pos", "positives: affine R²", "#2ca02c", ls="--")
    band(ax, "rho_neg", "negatives: rank corr.", "#7f7f7f")
    ax.set_title("how tight the map is")
    ax.set_xlabel("click")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "true_map.png", dpi=130)
    plt.close(fig)

    # 2. Each map's error on the withheld positives (scale alone) and on the top negatives.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for m in WHOLE_MAPS:
        band(axes[0], f"{m}_bias_pos", MAP_LABEL[m], MAP_COLOR[m])
        band(axes[1], f"{m}_bias_negtop", MAP_LABEL[m], MAP_COLOR[m])
    for ax, title in zip(axes, ("withheld positives", "withheld top 1% negatives")):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"mapped fold logit minus final logit: {title}")
        ax.set_xlabel("click")
    axes[0].set_ylabel("logit")
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figdir / "maps_on_withheld.png", dpi=130)
    plt.close(fig)

    # 3. The Good component each map gives the line, against the withheld positives' own mean.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in WHOLE_MAPS:
        band(axes[0], f"{m}_gap", MAP_LABEL[m], MAP_COLOR[m])
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("Good component's mean minus the withheld positives'")
    axes[0].set_xlabel("click")
    axes[0].set_ylabel("logit")
    axes[0].legend(fontsize=8, frameon=False)
    band(axes[1], "none_gap", "today's error", "#333333")
    band(axes[1], "gap_selection", "selection: held-out Goods vs withheld positives, fold scale", "#8c564b")
    band(axes[1], "gap_scale", "scale: fold vs final, withheld positives", "#2ca02c")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("today's error, split")
    axes[1].set_xlabel("click")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "good_component.png", dpi=130)
    plt.close(fig)

    # 4. Find's F-beta on the same heads.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    band(axes[0], "none_fbeta", "today's line", "#333333")
    band(axes[0], "best_fbeta", "best cut on the same ranking", "#bbbbbb")
    for m in MAPS:
        if m != "none":
            band(axes[0], f"{m}_fbeta", MAP_LABEL[m], MAP_COLOR[m], ls="--")
    axes[0].set_title("Find's F-beta on the withheld half, same heads")
    axes[0].set_xlabel("click")
    axes[0].legend(fontsize=7, frameon=False)
    for m in MAPS:
        if m != "none":
            band(axes[1], f"{m}_dF", MAP_LABEL[m], MAP_COLOR[m])
    band(axes[1], "truemu_dF", "Good mean set to the truth", "#8c564b", ls=":")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("gain over today's line, same heads")
    axes[1].set_xlabel("click")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "same_head_fbeta.png", dpi=130)
    plt.close(fig)

    # 5. Per session: the positives' shift and today's Good-component error, one line per session.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, col, title in (
        (axes[0], "dpos", "positives: final minus fold"),
        (axes[1], "none_gap", "Good component minus the withheld positives"),
    ):
        for _, g in folded.groupby(["cell", "seed"]):
            g = g.sort_values("click")
            ax.plot(g["click"], g[col], color="#1f77b4", alpha=0.06, lw=0.7)
        m, _ = _curve(folded, col, clicks)
        ax.plot(x, m, color="k", lw=2, label="mean over classes")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("click")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("logit")
    fig.tight_layout()
    fig.savefig(figdir / "per_session.png", dpi=130)
    plt.close(fig)


def pick_examples(folded: pd.DataFrame, click: int) -> list[tuple[str, pd.Series]]:
    """Four literal snapshots at *click* (folds drawing the line, no head-count cut): typical and the extremes."""
    sub = folded[(folded["click"] == click) & ~folded["head_count"].astype(bool)].dropna(subset=["dpos", "q_gap"])
    if sub.empty:
        return []
    med = sub.iloc[int(np.argmin(np.abs(sub["dpos"] - sub["dpos"].median())))]
    picks = [
        ("typical: median shift on positives", med),
        ("final model highest above the folds on positives", sub.loc[sub["dpos"].idxmax()]),
        ("final model lowest below the folds on positives", sub.loc[sub["dpos"].idxmin()]),
        ("quantile map's Good mean furthest below the truth", sub.loc[sub["q_gap"].idxmin()]),
    ]
    seen: set = set()
    out = []
    for why, row in picks:
        key = (row["cell"], row["seed"], row["t"])
        if key not in seen:
            seen.add(key)
            out.append((why, row))
    return out


def cmd_examples(a) -> None:
    """Scatter the final model's withheld logits on fold 0's for four literal snapshots, with the maps drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(a.out, "snapshots.csv.gz"))
    folded = df[df["line_src"] == "folds"]
    roots = {os.path.basename(os.path.dirname(r.rstrip("/"))): r for r in a.results}
    picks = pick_examples(folded, a.click)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.2))
    lines = []
    for ax, (why, row) in zip(np.atleast_1d(axes), picks):
        npz = os.path.join(roots[row["run"]], "cells", f"task_{int(row['idx']):04d}__testscores.npz")
        snap = next(v for v in load_snapshots(npz).values() if int(v["t"]) == int(row["t"]))
        s = Snapshot(snap)
        f = s.folds[0]
        neg = ~s.pos
        top = neg & (f.test >= np.quantile(f.test, 0.98))
        ax.scatter(f.test[top], s.final[top], s=4, color="#999999", label="top 2% negatives")
        ax.scatter(f.test[s.pos], s.final[s.pos], s=10, color="#2ca02c", label="positives")
        held = _logit(f.held_scores[f.held_labels == 1])
        ax.plot(
            held, np.full(held.size, ax.get_ylim()[0]), "|", color="#8c564b", ms=12, label="fold 0's held-out Goods"
        )
        lo = min(f.test[top].min(), f.test[s.pos].min())
        hi = max(f.test[s.pos].max(), held.max() if held.size else -np.inf)
        grid = np.linspace(lo, hi, 200)
        maps = build_maps(s)
        ax.plot(grid, grid, color="k", lw=0.8, label="identity (today)")
        ax.plot(grid, maps["q"][0](grid, 1), color=MAP_COLOR["q"], lw=1.2, label="quantile map")
        ax.plot(grid, maps["z"][0](grid, 1), color=MAP_COLOR["z"], lw=1.2, label="robust z map")
        ax.set_title(f"{row['cell']}, seed {int(row['seed'])}, vote {int(row['t'])}\n{why}", fontsize=8)
        ax.set_xlabel("fold 0 logit")
        lines.append(
            f"| {row['cell']} | {int(row['seed'])} | {int(row['t'])} | {why} | {row['dpos']:+.2f} | "
            f"{row['dnegtop']:+.2f} | {row['none_gap']:+.2f} | {row['q_gap']:+.2f} | "
            f"{row['none_fbeta']:.2f} / {row['q_fbeta']:.2f} |"
        )
    np.atleast_1d(axes)[0].set_ylabel("final model logit")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    figdir = Path(a.figs) if a.figs else Path(a.out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"examples_vote{a.click}.png", dpi=130)
    print(
        "| session | seed | vote | why | positives: final - fold | top negatives | Good mean - truth, today |"
        " the same, quantile | F-beta today / quantile |"
    )
    print("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--results", action="append", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--docs")
    r.add_argument("--figs")
    e = sub.add_parser("examples")
    e.add_argument("--results", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--click", type=int, default=100)
    e.add_argument("--figs")
    a = p.parse_args(argv)
    {"measure": cmd_measure, "report": cmd_report, "examples": cmd_examples}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
