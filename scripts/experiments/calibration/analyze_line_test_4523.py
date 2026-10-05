#!/usr/bin/env python3
"""Price Test mode's stop rule: the Test arm replayed over targets and budgets (#4523).

Test mode (``docs/plans/test-mode.md`` §2) stops each phase when its range is
narrower than a target, at a pick budget, or when its bands run out.  The
plan's values (a precision width of 0.20, 40 picks a phase) are proposals;
this prices them.  Every number here is a function of ``(the withheld half's
ranking, the line Find draws on it, the model's posteriors, the truth)``, so
the arm is **replayed** from the saved withheld-half snapshots
(``task_NNNN__testscores.npz``, the ``last`` entry: the final model's scores
and labels on the withheld half, the labels' class model, the run's balance)
over a grid of precision-width targets and per-phase budgets, several Test
seeds per session, without retraining anything.  The in-run arm's own rows
(``task_NNNN__linetest.csv``, the app's default budgets) are the control: the
replay at the same budgets and seed must reproduce them exactly, and
``harness_check.json`` says whether it did.

Worlds (``--world label=results_dir``) are the launcher's arms: COCO Better at
its default pool, positives thinned to 0.1%, and the Train pool thinned to
5%.  The last trains in a 5% world but its withheld half is still at 0.44%,
so ``--thin label=0.05`` thins that world's withheld half in the replay (its
negatives subsampled, seeded per cell) before the line is drawn, as #4383's
corpus draws did.  A document class (``--docs label=frames_dir:beta,...``) is
read from ``sota_documents.py --frames`` frames carrying the app's structural
ranking and line (#4523): the test half's pages in rank order, the line's
count, and no model (the structural detector has no class model), so its
recall estimate is the Jeffreys prior's alone.

Outputs under ``--out``: ``rows.csv.gz`` (one row per session x grid point x
Test seed), ``summary.csv`` (per world x beta x width x budget: cost, stops,
coverage, widths, errors, verdict agreement), ``pick.csv`` (the grid point
each world's sessions would settle on under :data:`COVERAGE_BAR`, beside the
default), ``harness_check.json`` and ``provenance.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.setup_env()

from _cells_paths import side_frame_files  # noqa: E402
from vtscore.eval.line_test_arm import row_from_snapshot, simulate_line_test  # noqa: E402
from vtscore.eval.voting_columns import LINE_TEST_COLUMNS  # noqa: E402
from vtscore.training.thresholds import DEFAULT_BUDGETS, PHASE_DONE, PHASE_NOTHING, LineBudgets  # noqa: E402

#: The grid the issue names: precision-width targets and per-phase pick budgets.
WIDTHS: tuple[float, ...] = (0.15, 0.20, 0.30)
BUDGETS: tuple[int, ...] = (20, 40, 80)
#: The coverage a 95% range must show for its grid point to be eligible when
#: ``pick.csv`` settles on one: the nominal level less the slack a few hundred
#: sessions' standard error allows.
COVERAGE_BAR = 0.93
#: The study's seed base (the issue number), mixed with each cell's own.
SEED_BASE = 4523

IDENT = ("seed", "dataset", "category", "calibration_seed", "style")
STOPS_MATCHES = ("width", "budget", "exhausted")
STOPS_MISSES = ("dry_run", "budget", "exhausted", "width")


def _hash(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def grid(widths: tuple[float, ...], budgets: tuple[int, ...]) -> list[LineBudgets]:
    return [LineBudgets(matches_width=w, matches_picks=b, misses_picks=b) for w in widths for b in budgets]


def _default_point(b: LineBudgets) -> bool:
    return (
        b.matches_width == DEFAULT_BUDGETS.matches_width
        and b.matches_picks == DEFAULT_BUDGETS.matches_picks
        and b.misses_picks == DEFAULT_BUDGETS.misses_picks
    )


def thin_mask(labels: np.ndarray, prevalence: float, rng: np.random.Generator) -> np.ndarray:
    """Keep every positive and a seeded subsample of the negatives so positives are *prevalence* of the corpus."""
    y = np.asarray(labels) >= 0.5
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    keep_neg = int(round(len(pos) * (1.0 - prevalence) / prevalence))
    keep = np.zeros(y.size, dtype=bool)
    keep[pos] = True
    if keep_neg >= len(neg):
        keep[neg] = True
    else:
        keep[rng.choice(neg, size=keep_neg, replace=False)] = True
    return keep


def _snapshot(z: np.lib.npyio.NpzFile) -> dict | None:
    """The ``last`` snapshot in a cell's test-score archive, as :func:`row_from_snapshot` reads it."""
    prefixes = sorted({k.rsplit("/", 1)[0] for k in z.files if k.endswith("/last/meta")})
    if not prefixes:
        return None
    p = prefixes[0]
    meta = json.loads(str(z[f"{p}/meta"]))
    return {**meta, "scores": z[f"{p}/scores"], "labels": z[f"{p}/labels"]}


def _harness_row(linetest_csv: Path | None) -> dict | None:
    if linetest_csv is None:
        return None
    df = pd.read_csv(linetest_csv)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return str(a) == str(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, str) or isinstance(b, str):
        return str(a if not (isinstance(a, float) and math.isnan(a)) else "") == str(
            b if not (isinstance(b, float) and math.isnan(b)) else ""
        )
    return a == b


def cell_rows(job: tuple) -> tuple[list[dict], dict]:
    """One cell's replay: its grid rows, and the harness check (the in-run row against the replay)."""
    world, npz_path, linetest_path, widths, budgets, test_seeds, thin = job
    harness = _harness_row(linetest_path)
    z = np.load(npz_path)
    snap = _snapshot(z)
    check = {
        "cell": Path(npz_path).name,
        "harness_row": harness is not None,
        "replayed": False,
        "same": None,
        "diff": [],
    }
    if snap is None:
        return [], {**check, "error": "no last snapshot"}
    if harness is not None:
        seed, cat = int(harness["seed"]), str(harness["category"])
    else:
        # The arm was off in this run: identity from the main frame.
        main = Path(npz_path).with_name(Path(npz_path).name.replace("__testscores.npz", ".csv"))
        if not main.exists():
            main = main.with_name(main.name + ".gz")
        first = pd.read_csv(main, nrows=1)
        seed, cat = int(first["seed"].iloc[0]), str(first["category"].iloc[0])
    keep = None
    if thin is not None:
        keep = thin_mask(snap["labels"], thin, np.random.default_rng([SEED_BASE, seed, _hash(cat)]))
    if harness is not None and thin is None:
        replay = row_from_snapshot(snap, budgets=DEFAULT_BUDGETS, seed=seed)
        diff = [k for k in replay if k in harness and not _same(replay[k], harness[k])]
        check.update({"replayed": True, "same": not diff, "diff": diff})
    ident = {
        "world": world,
        "category": cat,
        "band": cat.split("@")[-1] if "@" in cat else "",
        "seed": seed,
        "thinned_to": float("nan") if thin is None else thin,
    }
    rows: list[dict] = []
    for b in grid(widths, budgets):
        for s in range(test_seeds):
            row = row_from_snapshot(snap, budgets=b, seed=_hash(SEED_BASE, seed, cat, s), keep=keep)
            rows.append({**ident, "test_seed_index": s, "width": b.matches_width, "budget": b.matches_picks, **row})
    return rows, check


def doc_rows(job: tuple) -> list[dict]:
    """One document frame's replay at each beta: the structural ranking of the test half, no model."""
    world, frame_path, betas, widths, budgets, test_seeds = job
    z = np.load(frame_path)
    if "order" not in z.files or "line" not in z.files:
        return []
    order = np.asarray(z["order"], dtype=np.int64)
    score = np.asarray(z["score"], dtype=np.float64)
    line = float(z["line"])
    test = np.asarray(z["test"], dtype=bool)
    positive = np.asarray(z["positive"], dtype=bool)
    in_test = test[order]
    ranking = order[in_test]
    ranked_score = score[in_test]
    truth = positive[ranking]
    line_count = int(np.count_nonzero(ranked_score >= line))
    name = Path(frame_path).stem  # <source>__<class>__vNNN
    cid, v = name.rsplit("__v", 1)
    rows: list[dict] = []
    for beta in betas:
        for b in grid(widths, budgets):
            for s in range(test_seeds):
                out = simulate_line_test(
                    truth, line_count, beta, posteriors=None, budgets=b, seed=_hash(SEED_BASE, cid, beta, s)
                )
                rows.append(
                    {
                        "world": world,
                        "category": cid.replace("__", "/", 1),
                        "band": "",
                        "seed": 0,
                        "thinned_to": float("nan"),
                        "test_seed_index": s,
                        "width": b.matches_width,
                        "budget": b.matches_picks,
                        "t": int(v),
                        "beta": float(beta),
                        "line_source": "structural",
                        "has_model": 0,
                        **{k: v_ for k, v_ in b.as_dict().items() if k not in ("alpha", "draws")},
                        "test_seed": _hash(SEED_BASE, cid, beta, s),
                        **out,
                    }
                )
    return rows


# ------------------------------------------------------------------ summaries


def cluster_se(x: pd.Series, cells: pd.Series) -> float:
    """SE of a mean with (category, seed) clusters: the cluster means' SE."""
    df = pd.DataFrame({"x": x, "c": cells}).dropna()
    if df.empty:
        return float("nan")
    means = df.groupby("c")["x"].mean()
    return float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Per world x beta x width x budget (x click for documents): cost, stops, coverage, widths, errors, verdicts."""
    keys = ["world", "beta", "width", "budget"]
    out: list[dict] = []
    for key, g in rows.groupby(keys, sort=True):
        done = g[g["phase"] == PHASE_DONE]
        cells = done["category"].astype(str) + "/" + done["seed"].astype(str)
        rec: dict = dict(zip(keys, key))
        rec.update(
            {
                "n": len(g),
                "n_sessions": g[["category", "seed"]].drop_duplicates().shape[0],
                "n_done": len(done),
                "share_nothing": float((g["phase"] == PHASE_NOTHING).mean()),
                "line_count_mean": float(g["line_count"].mean()),
                "n_test_pos_mean": float(g["n_test_pos"].mean()),
                "picks_mean": float(done["picks_total"].mean()),
                "picks_p50": float(done["picks_total"].median()),
                "picks_p90": float(done["picks_total"].quantile(0.9)),
                "picks_above_mean": float(done["picks_above"].mean()),
                "picks_below_mean": float(done["picks_below"].mean()),
                "rounds_mean": float(done["rounds"].mean()),
            }
        )
        for s in STOPS_MATCHES:
            rec[f"matches_{s}"] = float((done["matches_stop"] == s).mean())
        for s in STOPS_MISSES:
            rec[f"misses_{s}"] = float((done["misses_stop"] == s).mean())
        for m in ("precision", "recall", "fbeta"):
            held = done[f"{m}_held"].astype(float)
            rec[f"{m}_cov"] = float(held.mean())
            rec[f"{m}_cov_se"] = cluster_se(held, cells)
            rec[f"{m}_width"] = float((done[f"{m}_hi"] - done[f"{m}_lo"]).mean())
            rec[f"{m}_abs_err"] = float((done[f"{m}_point"] - done[f"{m}_true"]).abs().mean())
            rec[f"{m}_bias"] = float((done[f"{m}_point"] - done[f"{m}_true"]).mean())
            rec[f"edges_{m}_cov"] = float(done[f"edges_{m}_held"].sum() / max(done["n_edges"].sum(), 1))
        rec["found_match"] = float(done["found_match"].mean())
        rec["tail_from_model"] = float(done["tail_from_model"].mean())
        rec["tail_err"] = float((done["tail_positives_model"] - done["tail_positives_true"]).mean())
        rec["tail_abs_err"] = float((done["tail_positives_model"] - done["tail_positives_true"]).abs().mean())
        rec["below_abs_err"] = float((done["positives_below_point"] - done["positives_below_true"]).abs().mean())
        rec["below_true_mean"] = float(done["positives_below_true"].mean())
        rec["verdict_match"] = float(done["verdict_match"].mean())
        rec["verdict_match_se"] = cluster_se(done["verdict_match"].astype(float), cells)
        for v in ("ship", "lean", "retrain"):
            rec[f"verdict_{v}"] = float((done["verdict"] == v).mean())
            rec[f"oracle_{v}"] = float((done["oracle_verdict"] == v).mean())
        lean = done[done["verdict"] == "lean"]
        rec["lean_gain_true"] = (
            float((lean["fbeta_at_best_edge_est_true"] - lean["fbeta_true"]).mean()) if len(lean) else float("nan")
        )
        rec["fbeta_true_mean"] = float(done["fbeta_true"].mean())
        rec["fbeta_best_cut_true_mean"] = float(done["fbeta_best_cut_true"].mean())
        out.append(rec)
    return pd.DataFrame(out)


def pick(summary: pd.DataFrame, bar: float = COVERAGE_BAR) -> pd.DataFrame:
    """Per world x beta: the cheapest grid point whose precision range covers at the bar, beside the default.

    Eligible points cover the truth at least *bar* of the time; among them the
    one with the fewest mean picks wins, ties to the wider target (fewer picks
    for the user).  A world where no point is eligible names the best coverage.
    """
    out: list[dict] = []
    for (world, beta), g in summary.groupby(["world", "beta"], sort=True):
        default = g[(g["width"] == DEFAULT_BUDGETS.matches_width) & (g["budget"] == DEFAULT_BUDGETS.matches_picks)]
        ok = g[g["precision_cov"] >= bar].sort_values(["picks_mean", "width"], ascending=[True, False])
        chosen = ok.iloc[0] if len(ok) else g.sort_values("precision_cov", ascending=False).iloc[0]
        out.append(
            {
                "world": world,
                "beta": beta,
                "eligible": len(ok),
                "width": chosen["width"],
                "budget": chosen["budget"],
                "picks_mean": chosen["picks_mean"],
                "picks_p90": chosen["picks_p90"],
                "precision_cov": chosen["precision_cov"],
                "recall_cov": chosen["recall_cov"],
                "fbeta_cov": chosen["fbeta_cov"],
                "verdict_match": chosen["verdict_match"],
                "default_picks_mean": float(default["picks_mean"].iloc[0]) if len(default) else float("nan"),
                "default_precision_cov": float(default["precision_cov"].iloc[0]) if len(default) else float("nan"),
                "default_recall_cov": float(default["recall_cov"].iloc[0]) if len(default) else float("nan"),
            }
        )
    return pd.DataFrame(out)


def sha_dir(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--world", action="append", default=[], help="label=results_dir (holding cells/)")
    ap.add_argument("--thin", action="append", default=[], help="label=prevalence: thin that world's withheld half")
    ap.add_argument("--docs", action="append", default=[], help="label=frames_dir:beta[,beta] (document frames)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--test-seeds", type=int, default=4, help="Test seeds per session and grid point")
    ap.add_argument("--limit", type=int, default=0, help="cells per world (0 = all); for a smoke run")
    args = ap.parse_args(argv)
    widths = tuple(float(w) for w in args.widths.split(","))
    budgets = tuple(int(b) for b in args.budgets.split(","))
    thin = {kv.split("=", 1)[0]: float(kv.split("=", 1)[1]) for kv in args.thin}
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs: list[tuple] = []
    prov: dict = {"inputs": {}, "widths": widths, "budgets": budgets, "test_seeds": args.test_seeds, "thin": thin}
    for spec in args.world:
        label, d = spec.split("=", 1)
        cells = Path(d) / "cells"
        npzs = sorted(cells.glob("task_*__testscores.npz"))
        linetests = {p.name.split("__")[0]: p for p in side_frame_files(cells, "__linetest")}
        if args.limit:
            npzs = npzs[: args.limit]
        prov["inputs"][label] = {"dir": str(cells), "snapshots": len(npzs), "sha": sha_dir(npzs)}
        for p in npzs:
            jobs.append(
                (
                    "cell",
                    (
                        label,
                        str(p),
                        linetests.get(p.name.split("__")[0]),
                        widths,
                        budgets,
                        args.test_seeds,
                        thin.get(label),
                    ),
                )
            )
    for spec in args.docs:
        label, rest = spec.split("=", 1)
        d, betas = rest.split(":", 1)
        frames = sorted(Path(d).glob("*__v*.npz"))
        if args.limit:
            frames = frames[: args.limit]
        prov["inputs"][label] = {"dir": str(d), "frames": len(frames), "sha": sha_dir(frames), "betas": betas}
        for p in frames:
            jobs.append(
                ("doc", (label, str(p), tuple(float(b) for b in betas.split(",")), widths, budgets, args.test_seeds))
            )

    rows: list[dict] = []
    checks: list[dict] = []

    def run(job):
        kind, payload = job
        if kind == "cell":
            return cell_rows(payload)
        return doc_rows(payload), None

    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r, c in ex.map(run, jobs, chunksize=4):
                rows.extend(r)
                if c is not None:
                    checks.append(c)
    else:
        for job in jobs:
            r, c = run(job)
            rows.extend(r)
            if c is not None:
                checks.append(c)
    df = pd.DataFrame(rows)
    missing = [c for c in LINE_TEST_COLUMNS if c not in df.columns and c not in IDENT]
    if missing:
        raise SystemExit(f"replay rows lack {missing}")
    df.to_csv(args.out / "rows.csv.gz", index=False)
    summary = summarise(df)
    summary.to_csv(args.out / "summary.csv", index=False)
    pick(summary).to_csv(args.out / "pick.csv", index=False)
    replayed = [c for c in checks if c["replayed"]]
    same = [c for c in replayed if c["same"]]
    harness = {
        "cells": len(checks),
        "with_harness_row": sum(1 for c in checks if c["harness_row"]),
        "replayed": len(replayed),
        "identical": len(same),
        "differing": [c for c in replayed if not c["same"]][:20],
        "errors": [c for c in checks if "error" in c][:20],
    }
    (args.out / "harness_check.json").write_text(json.dumps(harness, indent=2))
    prov["rows"] = len(df)
    prov["seconds"] = round(time.time() - t0, 1)
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"{len(df)} rows from {len(jobs)} inputs in {prov['seconds']}s -> {args.out}")
    print(f"harness check: {harness['identical']}/{harness['replayed']} replayed cells identical")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pick(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
