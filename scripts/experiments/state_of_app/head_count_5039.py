#!/usr/bin/env python3
"""#5039: price counts for the head's count line on #4937's snapshots, at every preset.

While the class model holds fewer than ``HEAD_COUNT_GOODS`` held-out Goods, the labels line at every balance is the
head's count line (#4786, #4798): ``beta ** 0.708`` times what stands out in the head's scores (``head_count_cut``).
#4937 found that count keeps 0.14 of the F-beta-best set at beta 4 on Binary Photo, 0.57 at beta 1 and 1.2 at beta
1/4. ``head_count_cut`` is a pure function of the corpus's scores, so any count is read here offline on the same heads:
Find's line on the withheld half, as #4937 read it.

FHIBE faces sit on the head's count line all session (they never hold 6 held-out Goods), so a change to it is priced
on them too, from the withheld-half snapshots of ``faces_snap`` runs (``CALIB_SAVE_TEST_SCORES_AT``).

  head_count_5039.py extract --snapshots CSV --results DIR [--results DIR ...] --out DIR [--procs N]
  head_count_5039.py extract-faces --runs DIR [--runs DIR ...] --out DIR [--procs N]
  head_count_5039.py price --x DIR --out DIR [--procs N] [--class-cut]
  head_count_5039.py grid --x DIR --out DIR [--procs N]
  head_count_5039.py report --priced DIR --grid DIR --docs DIR --figs DIR
  head_count_5039.py examples --x DIR --priced DIR --arm NAME --docs DIR --figs DIR

``extract`` keeps, for every snapshot on the head's count line (``mall/snapshots.csv.gz``'s ``head_count``, outside
the typed query's hold), the withheld half's scores best first and their truth, and the folds' held-out Goods' and
Bads' scores (the class model's evidence), one ``<run>.npz`` per run, and checks
that today's count reproduces #4937's ``today_kept`` on every one. ``extract-faces`` does the same for FHIBE face
runs (every snapshot whose class model holds fewer than ``HEAD_COUNT_GOODS`` held-out Goods).

``price`` draws, on every snapshot, in each prevalence scenario (``SCENARIOS``: the withheld half as it is, its
positives thinned to a quarter, its negatives to 0.44, and its negatives alone) and at each preset, the line of each
candidate (``CANDIDATES``) through the library itself: ``fit_labels_line`` on the held-out votes, then
``head_count_cut`` on the scenario's corpus with the class model's held-out Goods, under the candidate's
``head_count_recall`` knobs. ``grid`` reads, in the same corpora, the count at any multiplier of what stands out
(``MULTIPLIERS``) in place of ``beta ** 0.708``, with the preset's own gap floor; at ``beta ** 0.708`` it is the
library's count on every snapshot (``chk_today``). ``--class-cut`` adds, on the withheld half as it is, the class model's own cut (the gate
closed): #4937's hand-over to the class model.
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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_scale_4936 import PATHS, _ident, load_snapshots  # noqa: E402

KEY = ["run", "idx", "path", "t"]
#: The arms' exponent: 1.3x what stands out at beta 4 (log 1.3 / log 4).
E13 = 0.18925
#: name -> (HEAD_COUNT_RECALL_EXPONENT, HEAD_COUNT_RECALL_Z) as ``head_count_recall`` takes them.
CANDIDATES: dict[str, tuple] = {
    "today": ("off", "off"),
    "x10": (math.log(10.0) / math.log(4.0), "off"),  # the issue's steeper exponent: 10x what stands out at beta 4
    "e13": (E13, "off"),  # the count at 1.3x, no reach
    "z3": ("off", 3.0),  # the reach alone, today's count
    "e13z3": (E13, 3.0),  # the closed loop's arms
    "e13z275": (E13, 2.75),
    "e13z25": (E13, 2.5),
}
#: Candidates whose reach is not gated: drawn with a held-out Good below every score, so the gate always opens.
UNGATED: dict[str, tuple] = {"z3_open": ("off", 3.0), "e13z3_open": (E13, 3.0)}
#: The corpora each snapshot's line is drawn on: the withheld half (~0.44% on Binary), its positives thinned to a
#: quarter (~0.1%), its negatives to 0.44 (~1%): the State of the App's three prevalences (#4959); and its negatives
#: alone, the corpus with no positives #4452 keeps few of.
SCENARIOS = ("native", "pos1/4", "neg0.44", "empty")
BETAS = (0.25, 1.0, 4.0)


def _cell_arrays(args) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """``(key, scores best first, truth in that order, held-out Goods' and Bads' scores)`` per wanted snapshot."""
    from vtscore.eval.live_threshold_rules import head_count_recall
    from vtscore.training.thresholds.class_model import _finite_unit
    from vtscore.training.thresholds.labels_line import head_count_cut

    npz, run, wanted = args
    snaps = load_snapshots(npz)
    ident = _ident(os.path.dirname(npz), npz)
    out = []
    for prefix, path in PATHS.items():
        for key, snap in snaps.items():
            if not key.startswith(prefix + "/"):
                continue
            t = int(snap["t"])
            if (path, t) not in wanted:
                continue
            s = _finite_unit(np.asarray(snap["scores"], dtype=np.float64))
            y = np.asarray(snap["labels"], dtype=np.int8)
            order = np.argsort(-s, kind="stable")
            s_desc, y_desc = s[order], y[order]
            with head_count_recall("off", "off"):  # #4937's line: the head's count before #5039
                thr = head_count_cut(s_desc, float(snap["beta"]))
            fl = np.asarray(snap["fold_labels"], dtype=np.int8)
            fs = np.asarray(snap["fold_scores"], dtype=np.float64)
            meta = {"run": run, **ident, "path": path, "t": t, "k_today": int((s >= thr).sum())}
            out.append((meta, s_desc, y_desc, fs[fl == 1], fs[fl == 0]))
    # A step can be filed under several clicks; keep one copy of each.
    seen, uniq = set(), []
    for item in out:
        k = (item[0]["path"], item[0]["t"])
        if k not in seen:
            seen.add(k)
            uniq.append(item)
    return uniq


def cmd_extract(a) -> None:
    mall = pd.read_csv(a.snapshots, low_memory=False)
    mall = mall[(mall["head_count"] == True) & (mall["query_hold"] != True)]  # noqa: E712
    os.makedirs(a.out, exist_ok=True)
    for res in a.results:
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        mine = mall[mall["run"] == run]
        wanted_by_idx: dict[int, set] = {}
        for idx, path, t in mine[["idx", "path", "t"]].itertuples(index=False):
            wanted_by_idx.setdefault(int(idx), set()).add((str(path), int(t)))
        jobs = []
        for npz in sorted(glob.glob(os.path.join(res, "cells", "task_*__testscores.npz"))):
            idx = int(os.path.basename(npz).split("_")[1])
            if idx in wanted_by_idx:
                jobs.append((npz, run, wanted_by_idx[idx]))
        print(f"{run}: {len(jobs)} cells, {len(mine)} snapshots", flush=True)
        metas, ss, ys, gs, bs = [], [], [], [], []
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            for items in ex.map(_cell_arrays, jobs, chunksize=4):
                for meta, s, y, g, b in items:
                    metas.append(meta)
                    ss.append(s)
                    ys.append(y)
                    gs.append(g)
                    bs.append(b)

        def offsets(parts):
            return np.concatenate([[0], np.cumsum([p.size for p in parts])]).astype(np.int64)

        off = offsets(ss)
        np.savez(
            os.path.join(a.out, f"{run}.npz"),
            s=np.concatenate(ss),
            y=np.concatenate(ys),
            off=off,
            goods=np.concatenate(gs),
            goods_off=offsets(gs),
            bads=np.concatenate(bs),
            bads_off=offsets(bs),
        )
        meta = pd.DataFrame(metas)
        meta["row"] = np.arange(len(meta))
        chk = meta.merge(mine[KEY + ["today_kept"]], on=KEY, how="left")
        bad = int((chk["k_today"] != chk["today_kept"]).sum())
        missing = len(mine) - len(meta)
        print(
            f"{run}: wrote {len(meta)} snapshots ({off[-1]:,} scores); today_kept mismatches {bad}; missing {missing}"
        )
        meta.to_csv(os.path.join(a.out, f"{run}.meta.csv"), index=False)


def _face_cell(args) -> list[tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """The head-count snapshots of one FHIBE cell, as ``_cell_arrays`` gives them."""
    from vtscore.eval.live_threshold_rules import head_count_recall
    from vtscore.training.thresholds.class_model import _finite_unit
    from vtscore.training.thresholds.labels_line import HEAD_COUNT_GOODS, head_count_cut

    npz, run, beta = args
    z = np.load(npz, allow_pickle=False)
    idx = int(re.search(r"task_(\d+)__", os.path.basename(npz)).group(1))
    out, seen = [], set()
    for key in sorted({n.rsplit("/", 1)[0] for n in z.files if re.search(r"/t\d+/", n)}):
        meta = json.loads(str(z[key + "/meta"]))
        t = int(meta["t"])
        model = meta.get("model")
        if t in seen or not model or int(model["n_pos"]) >= HEAD_COUNT_GOODS:
            continue
        seen.add(t)
        s = _finite_unit(np.asarray(z[key + "/scores"], dtype=np.float64))
        y = np.asarray(z[key + "/labels"], dtype=np.int8)
        order = np.argsort(-s, kind="stable")
        fl = np.asarray(z[key + "/fold_labels"], dtype=np.int8)
        fs = np.asarray(z[key + "/fold_scores"], dtype=np.float64)
        with head_count_recall("off", "off"):  # the head's count before #5039, as the arms' control draws it
            thr = head_count_cut(s[order], beta)
        out.append(
            (
                {"run": run, "idx": idx, "cell": f"face{idx}", "class": f"face{idx}", "band": "", "seed": 0,
                 "path": "face", "t": t, "k_today": int((s >= thr).sum())},
                s[order], y[order], fs[fl == 1], fs[fl == 0],
            )
        )  # fmt: skip
    return out


def _save(out_dir: str, run: str, items: list) -> pd.DataFrame:
    metas = [m for m, *_ in items]

    def cat(j):
        return np.concatenate([it[j] for it in items])

    def offsets(j):
        return np.concatenate([[0], np.cumsum([it[j].size for it in items])]).astype(np.int64)

    np.savez(
        os.path.join(out_dir, f"{run}.npz"),
        s=cat(1), y=cat(2), off=offsets(1), goods=cat(3), goods_off=offsets(3), bads=cat(4), bads_off=offsets(4),
    )  # fmt: skip
    meta = pd.DataFrame(metas)
    meta["row"] = np.arange(len(meta))
    meta.to_csv(os.path.join(out_dir, f"{run}.meta.csv"), index=False)
    return meta


def cmd_extract_faces(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    for d in a.runs:
        run = "faces-" + os.path.basename(d.rstrip("/")).split("-roll-")[-1]
        beta = float(json.load(open(os.path.join(d, "results", "grid_shape.json")))["beta"])
        files = sorted(glob.glob(os.path.join(d, "results", "cells", "task_*__testscores.npz")))
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            items = [it for part in ex.map(_face_cell, [(f, run, beta) for f in files], chunksize=8) for it in part]
        items = [it for it in items if it[2].sum() > 0]
        meta = _save(a.out, run, items)
        print(f"{run}: {len(files)} cells, {len(meta)} head-count snapshots (beta {beta:g})", flush=True)


def load_x(x_dir: str) -> list[tuple[str, str, float, pd.DataFrame, list]]:
    """Every extracted run: (run, path, its own beta, meta, [(scores best first, truth, Goods, Bads)])."""
    out = []
    for npz in sorted(glob.glob(os.path.join(x_dir, "*.npz"))):
        run = os.path.basename(npz)[: -len(".npz")]
        meta = pd.read_csv(npz[: -len(".npz")] + ".meta.csv")
        z = np.load(npz)
        sa, ya, ga, ba = z["s"], z["y"], z["goods"], z["bads"]  # each read once: an NpzFile reads a member per access
        off, go, bo = z["off"], z["goods_off"], z["bads_off"]
        arrs = [
            (sa[off[i] : off[i + 1]], ya[off[i] : off[i + 1]], ga[go[i] : go[i + 1]], ba[bo[i] : bo[i + 1]])
            for i in range(len(meta))
        ]
        m = re.search(r"-b(025|1|4)$", run)
        beta = {"025": 0.25, "1": 1.0, "4": 4.0}[m.group(1)] if m else math.nan
        path = str(meta["path"].iloc[0]) if len(meta) else ""
        out.append((run, path, beta, meta, arrs))
    return out


def scenario_corpora(s: np.ndarray, y: np.ndarray, key: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """:data:`SCENARIOS`' corpora from one withheld half (best first), seeded by the snapshot."""
    rng = np.random.default_rng(key)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = rng.random(pos.size) < 0.25
    if not keep.any():
        keep[rng.integers(pos.size)] = True
    thin = np.sort(np.concatenate([pos[keep], neg]))
    sparse = np.sort(np.concatenate([pos, neg[rng.random(neg.size) < 0.44]]))
    return {
        "native": (s, y),
        "pos1/4": (s[thin], y[thin]),
        "neg0.44": (s[sparse], y[sparse]),
        "empty": (s[neg], y[neg]),
    }


def _fbeta_kept(s_desc: np.ndarray, y_desc: np.ndarray, thr: float, beta: float) -> tuple[int, float]:
    kept = int(np.count_nonzero(s_desc >= thr))
    p = float(y_desc.sum())
    if p == 0:
        return kept, math.nan
    b2 = beta * beta
    return kept, (1 + b2) * float(y_desc[:kept].sum()) / (b2 * p + kept)


def _price_snapshot(args) -> list[dict]:
    from vtscore.detectors.centroid_head import centroid_cut
    from vtscore.eval.live_threshold_rules import head_count_bads, head_count_recall
    from vtscore.training.thresholds.labels_line import _logit, fit_labels_line, head_count_cut

    ident, s, y, goods, bads, class_cut = args
    orderings = [(np.r_[goods, bads], np.r_[np.ones(goods.size), np.zeros(bads.size)])]
    line = fit_labels_line(orderings, s)
    held = None if line is None else line.model.pos_logits
    rows = []
    for scen, (sv, yv) in scenario_corpora(s, y, ident["key"]).items():
        on = None if (line is None or not class_cut or scen != "native") else line.on_corpus(sv)
        for beta in BETAS:
            r = {**ident, "scenario": scen, "beta": beta, "n": int(sv.size), "n_pos": int(yv.sum())}
            x = _logit(sv)
            r["standouts"] = int(np.count_nonzero(x >= centroid_cut(x, beta=1.0))) if sv.size else 0
            r["deepest_good_rank"] = (
                int(np.count_nonzero(x >= float(held.min()))) if held is not None and held.size else -1
            )
            if yv.sum() > 0:
                b2 = beta * beta
                tp = np.cumsum(yv)
                f = (1 + b2) * tp / (b2 * tp[-1] + np.arange(1, yv.size + 1))
                r["best_f"], r["best_k"] = float(f.max()), int(np.argmax(f)) + 1
            for name, (e, z) in CANDIDATES.items():
                with head_count_recall(e, z):
                    r[f"k_{name}"], r[f"F_{name}"] = _fbeta_kept(sv, yv, head_count_cut(sv, beta, held), beta)
            for name, (e, z) in UNGATED.items():
                with head_count_recall(e, z):
                    r[f"k_{name}"], r[f"F_{name}"] = _fbeta_kept(
                        sv, yv, head_count_cut(sv, beta, np.array([-1e9])), beta
                    )
            # The bulk above 3 spreads against a Gaussian's (#4603's median and MAD of the logits).
            if sv.size:
                med = float(np.median(x))
                sd = 1.4826 * float(np.median(np.abs(x - med)))
                r["above_z3"] = int(np.count_nonzero(x >= med + 3.0 * sd))
                r["gauss_z3"] = float(sv.size * 0.5 * math.erfc(3.0 / math.sqrt(2.0)))
            if on is not None:
                with head_count_bads(0):
                    r["k_class"], r["F_class"] = _fbeta_kept(sv, yv, on.threshold(beta), beta)
            rows.append(r)
    return rows


def cmd_price(a) -> None:
    jobs = _jobs(a.x, (a.class_cut,))
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)][: a.limit]
    print(f"{len(jobs)} snapshots", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for i, out in enumerate(ex.map(_price_snapshot, jobs, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "priced.csv.gz"), index=False)
    print(f"wrote {len(df)} rows")


#: The multipliers ``grid`` reads, log-spaced.
MULTIPLIERS = tuple(float(c) for c in np.exp(np.linspace(np.log(0.05), np.log(30.0), 40)))


def _count_at(x_desc: np.ndarray, beta: float, c: float) -> int:
    """The head's count with multiplier *c* in place of ``beta ** 0.708`` (``count_gap_floor`` below beta 1, its
    ``skip25`` drop from beta 1 up, as ``CENTROID_LINE_RULE`` draws it), as the number of logits it keeps."""
    from vtscore.detectors.centroid_head import _top_gap_count
    from vtscore.training.thresholds.gmm import _count_estimate

    n = x_desc.size
    gap = _top_gap_count(x_desc, drop="skip25" if beta >= 1.0 else "widest")
    counted = _count_estimate(x_desc)
    if counted is None:
        return -1
    k = min(max(1, round(c * max(counted, float(gap)))), n)
    line = min(float(np.partition(x_desc, n - k)[n - k]), float(x_desc[gap - 1]))
    return int(np.count_nonzero(x_desc >= line))


def _grid_snapshot(args) -> list[dict]:
    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT
    from vtscore.eval.live_threshold_rules import head_count_recall
    from vtscore.training.thresholds.labels_line import _logit, head_count_cut

    ident, s, y, _goods, _bads = args
    rows = []
    for scen, (sv, yv) in scenario_corpora(s, y, ident["key"]).items():
        x = _logit(sv)
        for beta in BETAS:
            with head_count_recall("off", "off"):  # today's: the line before #5039
                today_cut = head_count_cut(sv, beta)
            today = int(np.count_nonzero(sv >= today_cut))
            r = {**ident, "scenario": scen, "beta": beta, "k_today": today, "n_pos": int(yv.sum())}
            ks = [_count_at(x, beta, c) for c in MULTIPLIERS]
            # The cut sits between the last kept and the first dropped, as head_count_cut places it.
            kept = [int(np.count_nonzero(sv >= ((sv[k - 1] + sv[k]) / 2 if 0 < k < sv.size else sv[-1]))) if k > 0 else 0
                    for k in ks]  # fmt: skip
            r["chk_today"] = abs(
                today
                - (lambda k: int(np.count_nonzero(sv >= ((sv[k - 1] + sv[k]) / 2 if 0 < k < sv.size else sv[-1]))))(
                    _count_at(x, beta, beta**TEXT_SORT_COUNT_EXPONENT)
                )
            )
            if yv.sum() > 0:
                b2 = beta * beta
                f = (1 + b2) * np.cumsum(yv) / (b2 * yv.sum() + np.arange(1, yv.size + 1))
                r["F_today"] = float(f[today - 1]) if today > 0 else 0.0
                r.update({f"F{j}": float(f[k - 1]) if k > 0 else 0.0 for j, k in enumerate(kept)})
            r.update({f"k{j}": k for j, k in enumerate(kept)})
            rows.append(r)
    return rows


def _jobs(x_dir: str, extra=()) -> list:
    jobs = []
    for run, path, run_beta, meta, arrs in load_x(x_dir):
        for i, (s, y, g, b) in enumerate(arrs):
            m = meta.iloc[i]
            ident = {"run": run, "path": path, "run_beta": run_beta, "idx": int(m["idx"]), "t": int(m["t"]),
                     "cell": str(m["cell"]), "class": str(m["class"]), "band": str(m.get("band", "")),
                     "seed": int(m["seed"]), "n_goods": int(g.size), "n_bads": int(b.size)}  # fmt: skip
            ident["key"] = int.from_bytes(f"{run}|{ident['idx']}|{ident['t']}".encode()[-8:].ljust(8, b"\0"), "little")
            jobs.append((ident, s, y, g, b, *extra))
    return jobs


def cmd_grid(a) -> None:
    jobs = _jobs(a.x)
    print(f"{len(jobs)} snapshots", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for out in ex.map(_grid_snapshot, jobs, chunksize=4):
            rows.extend(out)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "grid.csv.gz"), index=False)
    print(f"wrote {len(df)} rows; chk_today max {df['chk_today'].max()}, nonzero {int((df['chk_today'] > 0).sum())}")


BENCH = {"binary": "Binary Photo", "region": "Region Photo", "face": "FHIBE faces"}
SCEN_NAME = {"native": "as it is", "pos1/4": "positives / 4", "neg0.44": "negatives x 0.44", "empty": "no positives"}
#: Click windows per bench (the step a snapshot was taken at).
WINDOWS = {
    "binary": ((30, 50), (51, 100), (101, 150)),
    "region": ((1, 25), (26, 50), (51, 150)),
    "face": ((1, 40), (41, 80), (81, 150)),
}
ARMS = ("e13z3", "e13z275")


def by_class(frame: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean and SE over classes (a class's snapshots averaged first: its bands, seeds and clicks are not independent)."""
    g = frame.groupby("class")[col].mean().dropna()
    if len(g) == 0:
        return math.nan, math.nan, 0
    return float(g.mean()), float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else math.nan, len(g)


def _pm(m: float, se: float, d: int = 3) -> str:
    if not np.isfinite(m):
        return "-"
    return f"{m:+.{d}f}" + (f" (± {se:.{d}f})" if np.isfinite(se) else "")


def report_tables(pr: pd.DataFrame, grid: "pd.DataFrame | None", docs: str) -> dict[str, pd.DataFrame]:
    os.makedirs(docs, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    cands = [c for c in CANDIDATES if c != "today"] + [c for c in UNGATED if f"F_{c}" in pr]
    cands += ["class"] if "F_class" in pr else []
    for c in cands:
        if f"F_{c}" in pr:
            pr[f"g_{c}"] = pr[f"F_{c}"] - pr["F_today"]
    b4 = pr[pr["beta"] == 4.0]
    # 1. The gain over today at beta 4, per bench and scenario.
    rows = []
    for (path, scen), h in b4[b4["scenario"] != "empty"].groupby(["path", "scenario"], sort=False):
        r = {"bench": BENCH.get(path, path), "scenario": SCEN_NAME[scen], "snapshots": len(h),
             "prevalence": float((h["n_pos"] / h["n"]).median()), "today": by_class(h, "F_today")[0],
             "best cut": by_class(h, "best_f")[0]}  # fmt: skip
        for c in cands:
            if f"g_{c}" in h and h[f"g_{c}"].notna().any():
                m, se, _ = by_class(h, f"g_{c}")
                r[c], r[c + "_se"] = m, se
        rows.append(r)
    out["offline_b4"] = pd.DataFrame(rows)
    # 2. By click window, the withheld half as it is.
    rows = []
    nat = b4[b4["scenario"] == "native"]
    for path, wins in WINDOWS.items():
        h0 = nat[nat["path"] == path]
        for lo, hi in wins:
            h = h0[(h0["t"] >= lo) & (h0["t"] <= hi)]
            if not len(h):
                continue
            r = {"bench": BENCH[path], "clicks": f"{lo}-{hi}", "snapshots": len(h), "today": by_class(h, "F_today")[0],
                 "best cut": by_class(h, "best_f")[0],
                 "set / best, today": float((h["k_today"] / h["best_k"]).median())}  # fmt: skip
            for c in cands:
                if f"g_{c}" in h and h[f"g_{c}"].notna().any():
                    m, se, _ = by_class(h, f"g_{c}")
                    r[c], r[c + "_se"] = m, se
                    r[f"set / best, {c}"] = float((h[f"k_{c}"] / h["best_k"]).median())
            rows.append(r)
    out["offline_windows"] = pd.DataFrame(rows)
    # 3. The corpus with no positives: images kept at beta 4.
    rows = []
    for path, h in b4[b4["scenario"] == "empty"].groupby("path", sort=False):
        r = {"bench": BENCH[path], "snapshots": len(h)}
        for c in ["today", *[c for c in CANDIDATES if c != "today"]]:
            r[f"{c} mean"] = float(h[f"k_{c}"].mean())
            r[f"{c} p90"] = float(h[f"k_{c}"].quantile(0.9))
        rows.append(r)
    out["empty_b4"] = pd.DataFrame(rows)
    # 4. How often the reach's gate opens: a held-out Good ranked below what stands out.
    rows = []
    for (path, scen), h in b4.groupby(["path", "scenario"], sort=False):
        has = h["deepest_good_rank"] >= 0
        r = {"bench": BENCH[path], "scenario": SCEN_NAME[scen], "snapshots": len(h),
             "gate open": float((h["deepest_good_rank"] > h["standouts"]).mean()),
             "standouts (median)": float(h["standouts"].median()),
             "deepest held-out Good's rank (median)": float(h.loc[has, "deepest_good_rank"].median()) if has.any() else math.nan,
             "held-out Goods (median)": float(h["n_goods"].median())}  # fmt: skip
        for c in ARMS:
            r[f"{c} moves the line"] = float((h[f"k_{c}"] != h["k_e13"]).mean())
        rows.append(r)
    out["gate_b4"] = pd.DataFrame(rows)
    # 4b. The bulk above 3 spreads on the corpus as it is: more than a Gaussian puts there is what the ungated reach keeps.
    if "above_z3" in pr:
        nb = b4[b4["scenario"] == "native"]
        out["bulk_z3"] = pd.DataFrame(
            [{"bench": BENCH[path], "snapshots": len(h), "above 3 spreads (median)": float(h["above_z3"].median()),
              "a Gaussian's": float(h["gauss_z3"].median()), "p90": float(h["above_z3"].quantile(0.9)),
              "max": int(h["above_z3"].max()), "more than a Gaussian's": float((h["above_z3"] > h["gauss_z3"]).mean()),
              "positives (median)": float(h["n_pos"].median())}
             for path, h in nb.groupby("path", sort=False)]
        )  # fmt: skip
    # 5. Below beta 1 and at it, every library candidate is today's line (they read beta > 1 only).
    low = pr[pr["beta"] <= 1.0]
    out["unchanged_low"] = pd.DataFrame(
        [{"beta": b, "rows": int((low["beta"] == b).sum()),
          **{c: int((low.loc[low["beta"] == b, f"k_{c}"] != low.loc[low["beta"] == b, "k_today"]).sum())
             for c in CANDIDATES if c != "today"}} for b in (0.25, 1.0)]
    )  # fmt: skip
    # 6. The multiplier grid: the best multiplier per bench, preset and scenario.
    if grid is not None:
        rows = []
        for (path, beta, scen), h in grid[grid["scenario"] != "empty"].groupby(
            ["path", "beta", "scenario"], sort=False
        ):
            gains = [by_class(h.assign(_g=h[f"F{j}"] - h["F_today"]), "_g")[0] for j in range(len(MULTIPLIERS))]
            j = int(np.nanargmax(gains))
            today_c = beta ** (math.log(3 / 8) / math.log(1 / 4))
            rows.append({"bench": BENCH[path], "beta": beta, "scenario": SCEN_NAME[scen], "today's multiplier": today_c,
                         "best multiplier": MULTIPLIERS[j], "its gain": gains[j],
                         **{f"x{c:g}": gains[int(np.argmin([abs(m - c) for m in MULTIPLIERS]))] for c in (0.2, 0.5, 1.3, 2.7, 10)}})  # fmt: skip
        out["grid_best"] = pd.DataFrame(rows)
        out["grid_check"] = pd.DataFrame([{"rows": len(grid), "chk_today nonzero": int((grid["chk_today"] > 0).sum())}])
    for name, df in out.items():
        df.to_csv(os.path.join(docs, f"{name}.csv"), index=False, float_format="%.4f")
    return out


#: The clicks each bench's snapshots were listed at: a snapshot is read for the largest one at or below its step.
SNAP_CLICKS = {
    "binary": (35, 40, 50, 75, 100, 125, 150),
    "region": (10, 25, 50, 75, 100, 125, 150),
    "face": (20, 30, 40, 60, 80, 100, 150),
}
#: Series colours, the categorical slots in fixed order (validated: dataviz validate_palette, light surface).
INK, MUTED_INK = "#0b0b0b", "#8a8985"
SLOT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.set_axisbelow(True)


def report_figures(pr: pd.DataFrame, grid: "pd.DataFrame | None", figs: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figs, exist_ok=True)
    scen_col = {"native": SLOT[0], "pos1/4": SLOT[1], "neg0.44": SLOT[2]}
    if grid is not None:
        paths = [p for p in ("binary", "region", "face") if p in set(grid["path"])]
        fig, axes = plt.subplots(len(paths), 3, figsize=(12, 3.0 * len(paths)), squeeze=False)
        for i, path in enumerate(paths):
            for j, beta in enumerate(BETAS):
                ax = axes[i][j]
                _style(ax)
                for scen, col in scen_col.items():
                    h = grid[(grid["path"] == path) & (grid["beta"] == beta) & (grid["scenario"] == scen)]
                    if not len(h):
                        continue
                    g = [by_class(h.assign(_g=h[f"F{q}"] - h["F_today"]), "_g")[0] for q in range(len(MULTIPLIERS))]
                    ax.plot(MULTIPLIERS, g, color=col, lw=2, label=SCEN_NAME[scen])
                ax.axhline(0, color=MUTED_INK, lw=0.8)
                ax.axvline(beta ** (math.log(3 / 8) / math.log(1 / 4)), color=MUTED_INK, ls=":", lw=1.2)
                if beta > 1:
                    ax.axvline(1.3, color=SLOT[0], ls=":", lw=1.2)
                ax.set_xscale("log")
                ax.set_ylim(-0.15, 0.1)
                ax.set_title(f"{BENCH[path]}, beta {beta:g}", fontsize=9, color=INK)
                if i == len(paths) - 1:
                    ax.set_xlabel("multiplier of what stands out", fontsize=8, color="#52514e")
                if j == 0:
                    ax.set_ylabel("F-beta gain over today", fontsize=8, color="#52514e")
        axes[0][0].legend(fontsize=7, frameon=False, title="corpus", title_fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "multiplier_grid.png"), dpi=130)
        plt.close(fig)
    nat = pr[(pr["beta"] == 4.0) & (pr["scenario"] == "native")].copy()
    paths = [p for p in ("binary", "region", "face") if p in set(nat["path"])]
    series = (
        ("e13z3", "1.3x + reach from z 3", SLOT[0], "-", 2.0),
        ("e13z275", "1.3x + reach from z 2.75", SLOT[1], "-", 2.0),
        ("class", "the class model's cut", SLOT[2], "-", 1.4),
        ("z3", "reach from z 3, today's count", SLOT[3], "-", 1.4),
        ("e13", "1.3x, no reach", MUTED_INK, "--", 1.4),
    )
    fig, axes = plt.subplots(1, len(paths), figsize=(4.2 * len(paths), 3.3), squeeze=False)
    for ax, path in zip(axes[0], paths):
        _style(ax)
        h = nat[nat["path"] == path].copy()
        listed = np.array(SNAP_CLICKS[path])
        h["click"] = [int(listed[listed <= t].max()) if (listed <= t).any() else int(listed[0]) for t in h["t"]]
        for c, name, col, ls, lw in series:
            if f"F_{c}" not in h or h[f"F_{c}"].isna().all():
                continue
            xs, ys = [], []
            for click, w in h.groupby("click"):
                xs.append(click)
                ys.append(by_class(w.assign(_g=w[f"F_{c}"] - w["F_today"]), "_g")[0])
            ax.plot(xs, ys, color=col, ls=ls, lw=lw, marker="o", ms=4, label=name)
        ax.axhline(0, color=MUTED_INK, lw=0.8)
        ax.set_title(f"{BENCH[path]}, beta 4", fontsize=9, color=INK)
        ax.set_xlabel("click", fontsize=8, color="#52514e")
    axes[0][0].set_ylabel("F-beta gain over today", fontsize=8, color="#52514e")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "b4_over_clicks.png"), dpi=130)
    plt.close(fig)


def faces_ab_figure(curves: pd.DataFrame, figs: str) -> None:
    """The closed loop on faces: each arm's paired F-beta gain over the control per vote (``prior_4776.py`` curves)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"e13z3": "1.3x + reach from z 3", "e13z275": "1.3x + reach from z 2.75"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, k, title in zip(axes, ("k1", "k4"), ("sessions from 1 photo", "sessions from 4 photos")):
        _style(ax)
        for arm, col in (("e13z3", SLOT[0]), ("e13z275", SLOT[1])):
            h = curves[(curves["set"] == f"{k}-{arm}") & (curves["metric"] == "F")].sort_values("vote")
            ax.plot(h["vote"], h["diff"], color=col, lw=2, label=names[arm])
            ax.fill_between(h["vote"], h["diff"] - h["diff_se"], h["diff"] + h["diff_se"], color=col, alpha=0.15, lw=0)
        ax.axhline(0, color=MUTED_INK, lw=0.8)
        ax.set_title(f"FHIBE faces, beta 4: {title}", fontsize=9, color=INK)
        ax.set_xlabel("vote", fontsize=8, color="#52514e")
    axes[0].set_ylabel("paired F-beta gain over the app (± 1 SE)", fontsize=8, color="#52514e")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "faces_ab_over_votes.png"), dpi=130)
    plt.close(fig)


def cmd_report(a) -> None:
    pr = pd.read_csv(os.path.join(a.priced, "priced.csv.gz"), low_memory=False)
    grid = pd.read_csv(os.path.join(a.grid, "grid.csv.gz"), low_memory=False) if a.grid else None
    out = report_tables(pr, grid, a.docs)
    report_figures(pr, grid, a.figs)
    if a.faces_ab:
        faces_ab_figure(pd.read_csv(os.path.join(a.faces_ab, "prior_curves.csv")), a.figs)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        for name, df in out.items():
            print(f"\n== {name}")
            print(df.round(4).to_string(index=False))


def pick_examples(pr: pd.DataFrame, arm: str) -> list[tuple[str, pd.Series]]:
    """Per bench, at beta 4 on the withheld half as it is, among snapshots the arm moves: the one it helps most, the
    median among those whose F-beta it changes, the one it hurts most."""
    out = []
    h = pr[(pr["beta"] == 4.0) & (pr["scenario"] == "native") & (pr[f"k_{arm}"] != pr["k_today"])].copy()
    h["gain"] = h[f"F_{arm}"] - h["F_today"]
    for path in ("binary", "region", "face"):
        g = h[h["path"] == path].sort_values("gain", kind="stable")
        if not len(g):
            continue
        moved = g[g["gain"] != 0]  # the median among those whose F-beta it changes
        mid = moved.iloc[len(moved) // 2] if len(moved) else g.iloc[len(g) // 2]
        out += [(f"{BENCH[path]}: helped most", g.iloc[-1]), (f"{BENCH[path]}: the median change", mid),
                (f"{BENCH[path]}: hurt most", g.iloc[0])]  # fmt: skip
    return out


def cmd_examples(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vtscore.eval.live_threshold_rules import head_count_recall
    from vtscore.training.thresholds.labels_line import _logit, fit_labels_line, head_count_cut

    pr = pd.read_csv(os.path.join(a.priced, "priced.csv.gz"), low_memory=False)
    picks = pick_examples(pr, a.arm)
    snaps = {
        (run, int(m["idx"]), int(m["t"])): arrs[i] for run, _, _, meta, arrs in load_x(a.x) for i, m in meta.iterrows()
    }
    rows = []
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.3 * len(picks)), squeeze=False)
    e, z = CANDIDATES[a.arm]
    for ax, (label, r) in zip(axes[:, 0], picks):
        sd, yd, g, b = snaps[(r["run"], int(r["idx"]), int(r["t"]))]
        line = fit_labels_line([(np.r_[g, b], np.r_[np.ones(g.size), np.zeros(b.size)])], sd)
        held = None if line is None else line.model.pos_logits
        with head_count_recall("off", "off"):
            k0 = int(np.count_nonzero(sd >= head_count_cut(sd, 4.0, held)))
        with head_count_recall(e, z):
            k1 = int(np.count_nonzero(sd >= head_count_cut(sd, 4.0, held)))
        x = _logit(sd)
        top = int(min(sd.size, max(k0, k1, int(r["best_k"])) * 1.6 + 20))
        rank = np.arange(1, top + 1)
        ax.scatter(rank[yd[:top] == 0], x[:top][yd[:top] == 0], s=3, color="#bbbbbb", lw=0)
        ax.scatter(rank[yd[:top] == 1], x[:top][yd[:top] == 1], s=12, color="#1baf7a", lw=0, label="positives")
        for k, col, ls, nm in (
            (k0, "#222222", "-", "today"),
            (k1, "#2a78d6", "-", a.arm),
            (int(r["best_k"]), "#8e44ad", "--", "best cut"),
        ):
            ax.axvline(k + 0.5, color=col, ls=ls, lw=1.2, label=f"{nm}: {k}")
        if held is not None and held.size:
            ax.axvline(
                int(r["deepest_good_rank"]) + 0.5, color="#eb6834", ls=":", lw=1.2, label="deepest held-out Good"
            )
        ax.set_xscale("log")
        ax.set_title(f"{label} - {r['cell']} seed {int(r['seed'])}, click {int(r['t'])}", fontsize=9)
        ax.set_ylabel("head's logit")
        ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper right")
        rows.append({"example": label, "session": r["cell"], "seed": int(r["seed"]), "click": int(r["t"]),
                     "held-out Goods": int(r["n_goods"]), "standouts": int(r["standouts"]),
                     "deepest held-out Good's rank": int(r["deepest_good_rank"]), "positives": int(r["n_pos"]),
                     "kept today": k0, f"kept {a.arm}": k1, "best set": int(r["best_k"]), "F today": float(r["F_today"]),
                     f"F {a.arm}": float(r[f"F_{a.arm}"]), "F best": float(r["best_f"]), "check": int(k1 == r[f"k_{a.arm}"])})  # fmt: skip
    axes[-1, 0].set_xlabel("rank in the withheld half (log)")
    fig.tight_layout()
    os.makedirs(a.figs, exist_ok=True)
    fig.savefig(os.path.join(a.figs, f"examples_{a.arm}.png"), dpi=130)
    plt.close(fig)
    df = pd.DataFrame(rows)
    os.makedirs(a.docs, exist_ok=True)
    df.to_csv(os.path.join(a.docs, f"examples_{a.arm}.csv"), index=False, float_format="%.3f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--snapshots", required=True)
    e.add_argument("--results", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=8)
    f = sub.add_parser("extract-faces")
    f.add_argument("--runs", action="append", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--procs", type=int, default=16)
    pr = sub.add_parser("price")
    pr.add_argument("--x", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--procs", type=int, default=16)
    pr.add_argument("--class-cut", action="store_true")
    pr.add_argument("--limit", type=int, default=0, help="price an even sample of this many snapshots (a test)")
    gr = sub.add_parser("grid")
    gr.add_argument("--x", required=True)
    gr.add_argument("--out", required=True)
    gr.add_argument("--procs", type=int, default=16)
    rp = sub.add_parser("report")
    rp.add_argument("--priced", required=True)
    rp.add_argument("--grid", default=None)
    rp.add_argument("--docs", required=True)
    rp.add_argument("--figs", required=True)
    rp.add_argument("--faces-ab", default=None, help="prior_4776.py --pair's output on the face arms")
    ex = sub.add_parser("examples")
    ex.add_argument("--x", required=True)
    ex.add_argument("--priced", required=True)
    ex.add_argument("--arm", default="e13z3")
    ex.add_argument("--docs", required=True)
    ex.add_argument("--figs", required=True)
    a = p.parse_args(argv)
    cmds = {"extract": cmd_extract, "extract-faces": cmd_extract_faces, "price": cmd_price, "grid": cmd_grid}
    {**cmds, "report": cmd_report, "examples": cmd_examples}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
