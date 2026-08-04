"""Runs whose held-out cost gets — and STAYS — worse as the user labels more (issue #2825).

The opposite of ``deep_spikes.py``. A deep spike is a *transient* excursion that snaps back
within a step or two; this tool hunts **persistent** wrong-way trends: cost climbing over a
run of >=K consecutive steps by >=delta and not giving the rise back, or a late/final cost
sitting meaningfully above an earlier best that never recovers. Those are cases where the
user is actively making the detector worse by labelling.

Works straight off the per-step sweep rows — **no new runs**:

* ``--jsonl-root DIR`` — any tree of ``results.jsonl`` (whole/boolean *and* region-voting
  ``hac`` rows). Arm/head are taken from the **path** (a ``mlp``/``linear``/``svm``/``reg-mlp``
  or ``arm_*`` component), because the row's own ``head`` field records the sweep default
  rather than the arm actually run — see ``--strict-head``.
* ``--csv-source PATH`` — the max-patch ``cells.csv.gz`` schema (DINOv3-patch region voting on
  Visual Genome).

The metric is **cost = FNR + FPR** throughout — the same headline the sweeps were run on.
Detection scans ``cost``; the ranking-vs-calibration split compares it with ``oracle_cost``,
which is that same cost evaluated at the best possible cut, so the whole analysis stays in one
unit. ``average_precision``/``auroc`` are reported where a source happens to carry them, but
they never classify a failure — a source without ``oracle_cost`` is simply reported as
undecided rather than judged on a different metric.

A **run** is one ``(source, group, head, embedder, proposal, class, seed)`` cost-vs-``t``
trajectory. Detection is on a median-smoothed series so single-step spikes cannot trigger it,
and every detection must beat a **per-run permutation test**: the same run's step-to-step cost
moves are reshuffled ``--per-run-null`` times (a random walk of exactly this run's volatility)
and the detection is kept only if its severity lands in the top ``--null-alpha`` tail. Without
that test a magnitude-only rule mostly measures how jumpy a run is — ``--permute-null`` reports
the pooled false-positive rate that makes the point.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

#: Model-class arms the sweeps write as a path component (the row field is unreliable).
HEADS = {"mlp", "linear", "svm", "reg-mlp", "anneal-svm"}
#: Which of those are linear-boundary models — the head-independence split.
LINEAR_FAMILY = {"linear", "svm", "anneal-svm"}
MLP_FAMILY = {"mlp", "reg-mlp"}
COLDSTART = "cosine_coldstart"

# Fields carried through from a row when present; everything else is ignored.
STEP_FIELDS = (
    "t", "cost", "fnr", "fpr", "f1", "threshold", "n_good", "n_bad",
    "phase", "calib_mode", "select_mode", "oracle_cost", "average_precision", "auroc",
)  # fmt: skip


@dataclass
class Run:
    """One labeling trajectory: the unit #2825 asks about."""

    key: tuple
    meta: dict
    steps: list = field(default_factory=list)

    def series(self, name):
        return [s.get(name) for s in self.steps]


# --------------------------------------------------------------------------- loading


def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _meta_from_path(root: Path, path: Path) -> dict:
    """Arm/head live in the directory tree, not in the row (see module docstring)."""
    parts = path.relative_to(root).parts[:-1]
    head = next((p for p in parts if p in HEADS), None)
    arm = next((p for p in parts if p.startswith("arm_")), None)
    group = "/".join(parts) or root.name
    return {"head": head, "arm": arm[4:] if arm else None, "group": group}


def load_jsonl_tree(root: Path, source: str, strict_head: bool = False):
    """Every ``results.jsonl`` under *root*, grouped into runs."""
    runs: dict[tuple, Run] = {}
    for f in sorted(root.rglob("results.jsonl")):
        pm = _meta_from_path(root, f)
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if pm["head"] is None and strict_head:
                raise SystemExit(f"--strict-head: no head component in {f}")
            head = pm["head"] or r.get("head")
            key = (
                source,
                pm["group"],
                pm["arm"],
                head,
                r.get("dataset"),
                r.get("embedder"),
                r.get("proposal"),
                r.get("class"),
                r.get("seed"),
            )
            run = runs.get(key)
            if run is None:
                run = runs[key] = Run(
                    key=key,
                    meta={
                        "source": source,
                        "group": pm["group"],
                        "arm": pm["arm"],
                        "head": head,
                        "dataset": r.get("dataset"),
                        "embedder": r.get("embedder"),
                        "proposal": r.get("proposal"),
                        "region_voting": bool(r.get("region_voting")),
                        "class": r.get("class"),
                        "seed": r.get("seed"),
                        "n_test_pos": r.get("n_test_pos"),
                        "n_test": r.get("n_test"),
                    },
                )
            step = {k: r.get(k) for k in STEP_FIELDS if k in r}
            for k in ("cost", "fnr", "fpr", "threshold", "oracle_cost", "f1"):
                if k in step:
                    step[k] = _num(step[k])
            run.steps.append(step)
    for run in runs.values():
        run.steps.sort(key=lambda s: s.get("t") or 0)
    return list(runs.values())


def load_maxpatch_csv(path: Path, source: str):
    """The max-patch ``cells.csv.gz`` schema: region-voting styles, AP/AUROC, no oracle."""
    runs: dict[tuple, Run] = {}
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as fh:
        for r in csv.DictReader(fh):
            style = r.get("style")
            key = (source, style, r.get("trainer"), r.get("embedder"), r.get("category"), r.get("seed"))
            run = runs.get(key)
            if run is None:
                run = runs[key] = Run(
                    key=key,
                    meta={
                        "source": source,
                        "group": style,
                        "arm": r.get("prevalence_arm"),
                        "head": r.get("trainer"),
                        "dataset": r.get("dataset"),
                        "embedder": r.get("embedder"),
                        "proposal": style,
                        "region_voting": style in {"max_patch", "max_patch_hac", "max_hac"},
                        "class": r.get("category"),
                        "seed": int(r["seed"]),
                    },
                )
            run.steps.append(
                {
                    "t": int(r["t"]),
                    "cost": _num(r.get("cost")),
                    "fnr": _num(r.get("fnr")),
                    "fpr": _num(r.get("fpr")),
                    "n_good": _num(r.get("n_good")),
                    "n_bad": _num(r.get("n_bad")),
                    "average_precision": _num(r.get("average_precision")),
                    "auroc": _num(r.get("auroc")),
                }
            )
    for run in runs.values():
        run.steps.sort(key=lambda s: s["t"])
    return list(runs.values())


# --------------------------------------------------------------------------- detection


def smooth_median(xs, w: int = 3):
    """Centred rolling median — kills 1-step spikes so only sustained moves survive."""
    if w <= 1:
        return list(xs)
    half = w // 2
    out = []
    for i in range(len(xs)):
        win = [v for v in xs[max(0, i - half) : i + half + 1] if v is not None]
        out.append(statistics.median(win) if win else None)
    return out


def find_sustained_rise(s, k: int, delta: float, hold: int, min_up_frac: float):
    """The worst floor->peak rise that lasts.

    Walks the running minimum, so every candidate segment reads "from this run's best-so-far
    up to here". A candidate survives only if it spans >=*k* steps, rises >=*delta*, is mostly
    upward, and the elevated level then **persists** without falling back below
    ``floor + delta/2`` — that persistence is what separates a wrong-way trend from the
    transient deep spike of #2790. A run that simply *ends* elevated has nowhere left to
    recover, so the hold requirement is capped by the steps that remain. Among survivors we
    keep the one with the largest ``rise x hold``, which is how detections are ranked.
    """
    n = len(s)
    if n < k + 1:
        return None
    best = None
    min_i = None
    for j in range(n):
        if s[j] is None:
            continue
        if min_i is None:
            min_i = j
            continue
        if j - min_i >= k:
            cand = _validate_rise(s, min_i, j, delta, hold, min_up_frac)
            if cand is not None and (best is None or cand["rise"] * cand["hold"] > best["rise"] * best["hold"]):
                best = cand
        # ``<=`` on purpose: anchor the floor at the LAST step that was at the best-so-far, not
        # the first. With ``<`` a long flat stretch gets swallowed into the segment and drags
        # ``up_frac`` down to the rejection boundary, so real climbs off a plateau are missed.
        if s[j] <= s[min_i]:
            min_i = j
    return best


def _validate_rise(s, i, j, delta, hold, min_up_frac):
    rise = s[j] - s[i]
    if rise < delta:
        return None
    seg = [v for v in s[i : j + 1] if v is not None]
    ups = sum(1 for a, b in zip(seg, seg[1:]) if b > a)
    up_frac = ups / max(1, len(seg) - 1)
    if up_frac < min_up_frac:
        return None
    floor = s[i] + delta / 2.0
    n = len(s)
    held = 0
    for m in range(j, n):
        if s[m] is None or s[m] < floor:
            break
        held += 1
    if held < min(hold, n - j):
        return None
    return {
        "start_idx": i,
        "peak_idx": j,
        "rise": rise,
        "span": j - i,
        "up_frac": up_frac,
        "hold": held,
        "held_to_end": j + held >= n,
    }


def find_best_late_gap(s, k: int, delta: float, late_window: int):
    """Late plateau sitting above an earlier best that never came back."""
    n = len(s)
    if n < k + late_window:
        return None
    head = [v for v in s[: n - late_window] if v is not None]
    if not head:
        return None
    best_v = min(head)
    best_i = next(i for i, v in enumerate(s[: n - late_window]) if v == best_v)
    if best_i > n - 1 - k:
        return None
    tail = [v for v in s[n - late_window :] if v is not None]
    if not tail:
        return None
    late = statistics.median(tail)
    gap = late - best_v
    if gap < delta:
        return None
    return {"best_idx": best_i, "best": best_v, "late": late, "gap": gap, "steps_after_best": n - 1 - best_i}


def analysis_window(run: Run, regime: str):
    """Indices of the steps to scan. ``learned`` drops the pre-MLP cosine cold-start."""
    steps = run.steps
    if regime == "all":
        return list(range(len(steps)))
    have_mode = any("calib_mode" in s for s in steps)
    if not have_mode:
        return list(range(len(steps)))
    if regime == "coldstart":
        return [i for i, s in enumerate(steps) if s.get("calib_mode") == COLDSTART]
    return [i for i, s in enumerate(steps) if s.get("calib_mode") != COLDSTART]


def detect(run: Run, cfg) -> dict | None:
    idx = analysis_window(run, cfg.regime)
    if len(idx) < cfg.k + 2:
        return None
    raw = [run.steps[i].get("cost") for i in idx]
    if sum(1 for v in raw if v is not None) < cfg.k + 2:
        return None
    s = smooth_median(raw, cfg.smooth)
    rise = find_sustained_rise(s, cfg.k, cfg.delta, cfg.hold, cfg.min_up_frac)
    gap = find_best_late_gap(s, cfg.k, cfg.delta, cfg.late_window)
    if rise is not None:
        a, b = idx[rise["start_idx"]], idx[rise["peak_idx"]]
        severity = rise["rise"] * rise["hold"]
        kind = "rise"
    elif gap is not None:
        a, b = idx[gap["best_idx"]], idx[-1]
        severity = gap["gap"] * gap["steps_after_best"]
        kind = "late-gap"
    else:
        return None
    return {
        "meta": run.meta,
        "kind": kind,
        "rise": rise,
        "late_gap": gap,
        "severity": severity,
        "seg_i": a,
        "seg_j": b,
        "n_steps": len(idx),
    }


# --------------------------------------------------------------------------- characterisation


def _regret(step):
    """cost - oracle_cost at one step: the calibration gap, when both are recorded."""
    c, o = step.get("cost"), step.get("oracle_cost")
    return None if c is None or o is None else c - o


def _delta(run: Run, name, a, b):
    va, vb = run.steps[a].get(name), run.steps[b].get(name)
    if va is None or vb is None:
        return None
    return vb - va


def characterise(run: Run, det: dict) -> dict:
    a, b = det["seg_i"], det["seg_j"]
    sa, sb = run.steps[a], run.steps[b]
    d_cost = _delta(run, "cost", a, b)
    d_fnr = _delta(run, "fnr", a, b)
    d_fpr = _delta(run, "fpr", a, b)
    d_thr = _delta(run, "threshold", a, b)
    d_or = _delta(run, "oracle_cost", a, b)
    d_ap = _delta(run, "average_precision", a, b)
    d_auroc = _delta(run, "auroc", a, b)

    driver = None
    if d_fnr is not None and d_fpr is not None:
        if d_fnr > 0 and d_fpr <= 0:
            driver = "fnr"
        elif d_fpr > 0 and d_fnr <= 0:
            driver = "fpr"
        elif d_fnr > 0 and d_fpr > 0:
            driver = "both"
        else:
            driver = "neither"

    # Ranking vs calibration, decided **only** on cost=FNR+FPR: oracle_cost is the same cost at
    # the best achievable cut, so its rise is the part of the damage no threshold could undo.
    # Sources that never recorded oracle_cost stay undecided — AP/AUROC are a different metric
    # and are not allowed to stand in for this call.
    failure = None
    ranking_share = None
    if d_or is not None and d_cost:
        ranking_share = d_or / d_cost
        failure = "ranking" if ranking_share >= 0.5 else ("calibration" if ranking_share <= 0.2 else "mixed")
    elif d_cost:
        failure = "undecided-no-oracle-cost"

    thr_mono = None
    thr_vals = [run.steps[i].get("threshold") for i in range(a, b + 1)]
    thr_vals = [v for v in thr_vals if v is not None]
    if len(thr_vals) > 2:
        ups = sum(1 for x, y in zip(thr_vals, thr_vals[1:]) if y > x)
        thr_mono = ups / (len(thr_vals) - 1)

    # What the user was actually feeding the loop across the segment. The mechanism claim is
    # that a wrong-way stretch is a stretch where the bad votes keep coming and the good ones
    # stop, so the cut has nothing holding it down and marches up through the test positives.
    d_good = _delta(run, "n_good", a, b)
    d_bad = _delta(run, "n_bad", a, b)
    votes = (d_good or 0) + (d_bad or 0)
    good_share = (d_good / votes) if votes else None
    run_good = run.steps[-1].get("n_good")
    run_bad = run.steps[-1].get("n_bad")
    run_votes = (run_good or 0) + (run_bad or 0)
    good_share_run = (run_good / run_votes) if run_votes else None
    # Longest stretch inside the segment with no new positive at all.
    dry = best_dry = 0
    for i in range(a + 1, b + 1):
        prev, cur = run.steps[i - 1].get("n_good"), run.steps[i].get("n_good")
        if prev is not None and cur is not None and cur > prev:
            dry = 0
        else:
            dry += 1
            best_dry = max(best_dry, dry)

    return {
        "t_start": sa.get("t"),
        "t_end": sb.get("t"),
        "cost_start": sa.get("cost"),
        "cost_end": sb.get("cost"),
        "cost_final": run.steps[-1].get("cost"),
        "d_cost": d_cost,
        "d_fnr": d_fnr,
        "d_fpr": d_fpr,
        "driver": driver,
        "d_threshold": d_thr,
        "thr_up_frac": thr_mono,
        "d_oracle": d_or,
        "ranking_share": ranking_share,
        "d_ap": d_ap,
        "d_auroc": d_auroc,
        "failure": failure,
        "n_good_start": sa.get("n_good"),
        "n_bad_start": sa.get("n_bad"),
        "n_good_end": sb.get("n_good"),
        "n_bad_end": sb.get("n_bad"),
        "d_n_good": d_good,
        "d_n_bad": d_bad,
        "good_share_segment": good_share,
        "good_share_run": good_share_run,
        "longest_no_positive_stretch": best_dry,
        "phase_start": sa.get("phase"),
        "calib_start": sa.get("calib_mode"),
        "regret_start": _regret(sa),
        "regret_end": _regret(sb),
    }


# --------------------------------------------------------------------------- null control


def null_severities(run: Run, cfg, n: int, rng: random.Random):
    """Severity the detector reports on *n* surrogates of this run's own cost moves."""
    out = []
    for _ in range(n):
        d = detect(permuted_copy(run, rng, cfg.null_mode, cfg.null_block), cfg)
        out.append(d["severity"] if d is not None else 0.0)
    return out


def permutation_p(observed: float, nulls) -> float:
    """Fraction of the null draws that match or beat the observed severity (add-one)."""
    return (1 + sum(1 for v in nulls if v >= observed)) / (len(nulls) + 1)


def _shuffle_blocks(deltas, block: int, rng: random.Random):
    """Reorder contiguous blocks of moves instead of individual moves.

    Single-move shuffling is the wrong null here: a #2790 deep spike is a ``+0.9`` step
    immediately followed by a ``-0.9`` recovery, and shuffling separates the pair, so the
    surrogate manufactures exactly the sustained level shifts we are testing for. Keeping
    moves in blocks preserves that local pairing, which is what makes the test usable.
    """
    if block <= 1:
        rng.shuffle(deltas)
        return deltas
    blocks = [deltas[i : i + block] for i in range(0, len(deltas), block)]
    rng.shuffle(blocks)
    return [d for b in blocks for d in b]


def permuted_copy(run: Run, rng: random.Random, mode: str = "demean", block: int = 5) -> Run:
    """A same-volatility surrogate of this run's cost curve.

    ``demean`` (default) re-centres the step-to-step moves to zero net drift before
    reshuffling: the surrogate is a **driftless** walk that is exactly as jumpy as the real
    run, which is the null a *trend* claim has to beat. ``raw`` reshuffles the moves as they
    are — but that preserves the run's start and end points, so it can only ask whether the
    ordering is unusual, never whether the run ended worse. Use it as a sensitivity check.
    *block* is the block length for the reshuffle (see :func:`_shuffle_blocks`).
    """
    idx = list(range(len(run.steps)))
    costs = [run.steps[i].get("cost") for i in idx]
    have = [i for i, v in enumerate(costs) if v is not None]
    if len(have) < 3:
        return run
    deltas = [costs[b] - costs[a] for a, b in zip(have, have[1:])]
    if mode == "demean":
        drift = sum(deltas) / len(deltas)
        deltas = [d - drift for d in deltas]
    deltas = _shuffle_blocks(list(deltas), block, rng)
    new = list(costs)
    cur = costs[have[0]]
    for pos, d in zip(have[1:], deltas):
        cur += d
        new[pos] = cur
    steps = [dict(s) for s in run.steps]
    for i, v in enumerate(new):
        if v is not None:
            steps[i]["cost"] = v
    return Run(key=run.key, meta=run.meta, steps=steps)


def ramped_copy(run: Run, total_rise: float, cfg) -> Run | None:
    """This run with a synthetic sustained rise injected over the back half of its window.

    Used for the power check: if a planted wrong-way trend of a size we care about would not
    survive the detector on *this* run, then "no detections here" means "too noisy to tell",
    not "nothing is wrong".
    """
    idx = analysis_window(run, cfg.regime)
    if len(idx) < 4:
        return None
    back = idx[len(idx) // 2 :]
    steps = [dict(st) for st in run.steps]
    for j, i in enumerate(back):
        c = steps[i].get("cost")
        if c is not None:
            steps[i]["cost"] = c + total_rise * (j + 1) / len(back)
    return Run(key=run.key, meta=run.meta, steps=steps)


def gated_detect(run: Run, cfg, rng: random.Random):
    """detect() plus the per-run permutation gate — the full pipeline, as applied to real data."""
    d = detect(run, cfg)
    if d is None:
        return None
    if cfg.per_run_null:
        if permutation_p(d["severity"], null_severities(run, cfg, cfg.per_run_null, rng)) > cfg.null_alpha:
            return None
    return d


def fp_report(runs, cfg, reps: int, rng: random.Random):
    """Type-I error: feed the pipeline surrogates that contain no trend by construction."""
    groups = defaultdict(lambda: [0, 0])
    for r in runs:
        for _ in range(reps):
            sur = permuted_copy(r, rng, cfg.null_mode, cfg.null_block)
            g = groups[(r.meta["dataset"], r.meta["proposal"], r.meta["embedder"])]
            g[1] += 1
            if gated_detect(sur, cfg, rng) is not None:
                g[0] += 1
    print(f"\n== type-I error (pipeline run on trend-free surrogates, target <= {cfg.null_alpha:.2f}) ==")
    print(f"  {'group':<50} {'trials':>7} {'false pos':>10} {'rate':>7}")
    for g, (hit, tot) in sorted(groups.items()):
        print(f"  {str(g):<50} {tot:>7} {hit:>10} {hit / max(1, tot):>7.3f}")


def power_report(runs, cfg, rise: float, rng: random.Random):
    """Per-source: how volatile the cost curves are, and whether a planted trend is findable.

    Two columns, because they answer different questions. ``on run`` plants the rise on the
    real trajectory, so it measures whether this much *extra* worsening would show up on top
    of whatever the run already does — and most runs are busy improving, which cancels much
    of it. ``on flat`` plants the same rise on a trendless surrogate of the same volatility,
    which isolates the detector's own sensitivity. Read ``on flat`` when deciding whether a
    source's zero detections mean "nothing there" or "could not have seen it".
    """
    groups = defaultdict(list)
    for r in runs:
        groups[(r.meta["dataset"], r.meta["proposal"], r.meta["embedder"])].append(r)
    print(f"\n== detection power (planted sustained rise of +{rise:.2f} in cost over the back half) ==")
    print(f"  {'group':<50} {'runs':>5} {'|dcost| med':>11} {'on run':>8} {'on flat':>8} {'drift':>8}")
    for g, rs in sorted(groups.items()):
        vols, drifts = [], []
        found = found_flat = tried = 0
        for r in rs:
            idx = analysis_window(r, cfg.regime)
            cs = [r.steps[i].get("cost") for i in idx]
            cs = [c for c in cs if c is not None]
            if len(cs) > 2:
                vols.append(statistics.median(abs(b - a) for a, b in zip(cs, cs[1:])))
                drifts.append(cs[-1] - cs[0])
            ramped = ramped_copy(r, rise, cfg)
            if ramped is None:
                continue
            tried += 1
            if gated_detect(ramped, cfg, rng) is not None:
                found += 1
            flat = ramped_copy(permuted_copy(r, rng, "demean", cfg.null_block), rise, cfg)
            if flat is not None and gated_detect(flat, cfg, rng) is not None:
                found_flat += 1
        vol = statistics.median(vols) if vols else float("nan")
        dr = statistics.median(drifts) if drifts else float("nan")
        rate = found / tried if tried else float("nan")
        rate_f = found_flat / tried if tried else float("nan")
        print(f"  {str(g):<50} {len(rs):>5} {vol:>11.4f} {rate:>7.0%} {rate_f:>8.0%} {dr:>+8.3f}")


# --------------------------------------------------------------------------- reporting


def _rate_table(title, dets, runs, keyfn, min_runs: int = 1, limit: int | None = None):
    """Wrong-way rate per group. *min_runs* hides cells too small to read a rate off."""
    tot, hit = defaultdict(int), defaultdict(int)
    for r in runs:
        tot[keyfn(r.meta)] += 1
    for d in dets:
        hit[keyfn(d["meta"])] += 1
    shown = [k for k in tot if tot[k] >= min_runs]
    hidden = len(tot) - len(shown)
    print(f"\n== {title} ==" + (f"   (>= {min_runs} runs; {hidden} smaller cells hidden)" if hidden else ""))
    print(f"  {'group':<50} {'runs':>6} {'wrong-way':>10} {'rate':>7}")
    order = sorted(shown, key=lambda k: -(hit[k] / max(1, tot[k])))
    for k in order[:limit] if limit else order:
        print(f"  {str(k):<50} {tot[k]:>6} {hit[k]:>10} {hit[k] / max(1, tot[k]):>7.3f}")


def _share(items, pred):
    items = list(items)
    return (sum(1 for x in items if pred(x)) / len(items)) if items else float("nan")


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else float("nan")


def report(runs, dets, cfg, null_rate=None):
    print(f"runs scanned: {len(runs)}   wrong-way runs: {len(dets)} ({len(dets) / max(1, len(runs)):.3f})")
    print(
        f"  criteria: K>={cfg.k} steps, delta>={cfg.delta}, hold>={cfg.hold}, up_frac>={cfg.min_up_frac}, "
        f"smooth={cfg.smooth}, late_window={cfg.late_window}, regime={cfg.regime}"
    )
    if null_rate is not None:
        print(f"  permuted-null rate (same thresholds, temporal order destroyed): {null_rate:.3f}")
    if not dets:
        return
    _rate_table("by dataset x proposal x embedder", dets, runs,
                lambda m: (m["dataset"], m["proposal"], m["embedder"]))  # fmt: skip
    _rate_table("by head / arm", dets, runs, lambda m: (m["dataset"], m["proposal"], m["head"], m["arm"]))
    _rate_table("by class (worst 20)", dets, runs, lambda m: (m["dataset"], m["proposal"], m["class"]),
                min_runs=10, limit=20)  # fmt: skip

    ch = [d["char"] for d in dets]
    print("\n== what is going wrong ==")
    print(f"  kind: rise {_share(dets, lambda d: d['kind'] == 'rise'):.0%}  "
          f"late-gap {_share(dets, lambda d: d['kind'] == 'late-gap'):.0%}")  # fmt: skip
    for name in ("fnr", "fpr", "both", "neither"):
        sh = _share(ch, lambda c, n=name: c["driver"] == n)
        if sh == sh and sh > 0:
            print(f"  driver={name:<8} {sh:.0%}")
    print(f"  median rise in cost      : {_med(c['d_cost'] for c in ch):+.3f}")
    never = _share(ch, lambda c: c["cost_final"] is not None and c["cost_start"] is not None
                   and c["cost_final"] >= c["cost_start"] + 0.5 * cfg.delta)  # fmt: skip
    print(f"  still worse at the LAST step of the run: {never:.0%}   "
          f"median cost at t_end-of-run {_med(c['cost_final'] for c in ch):.3f} "
          f"vs {_med(c['cost_start'] for c in ch):.3f} at onset")  # fmt: skip
    print(f"  median onset step t      : {_med(c['t_start'] for c in ch):.0f}")
    print(f"  median d_fnr / d_fpr     : {_med(c['d_fnr'] for c in ch):+.3f} / {_med(c['d_fpr'] for c in ch):+.3f}")
    print(f"  median d_threshold       : {_med(c['d_threshold'] for c in ch):+.3f}  "
          f"(share monotone-up >0.7: {_share(ch, lambda c: (c['thr_up_frac'] or 0) > 0.7):.0%})")  # fmt: skip
    print(f"  n_good at onset (median) : {_med(c['n_good_start'] for c in ch):.0f}   "
          f"n_bad {_med(c['n_bad_start'] for c in ch):.0f}")  # fmt: skip
    ph = defaultdict(int)
    for c in ch:
        ph[c["phase_start"]] += 1
    print(f"  phase at onset           : {dict(sorted(ph.items(), key=lambda kv: -kv[1]))}")

    print("\n== what the loop was being fed across the wrong-way segment ==")
    print(f"  votes added: good {_med(c['d_n_good'] for c in ch):.0f}  bad {_med(c['d_n_bad'] for c in ch):.0f}"
          "   (median)")  # fmt: skip
    gs = [c["good_share_segment"] for c in ch if c["good_share_segment"] is not None]
    gr = [c["good_share_run"] for c in ch if c["good_share_run"] is not None]
    print(f"  share of votes that were GOOD: {_med(gs):.2f} inside the segment "
          f"vs {_med(gr):.2f} over the whole run")  # fmt: skip
    print(f"  segments where NOT ONE new positive arrived: "
          f"{_share(ch, lambda c: (c['d_n_good'] or 0) == 0):.0%}")  # fmt: skip
    print(f"  longest run of steps with no new positive  : median "
          f"{_med(c['longest_no_positive_stretch'] for c in ch):.0f} steps")  # fmt: skip

    withor = [c for c in ch if c["ranking_share"] is not None]
    if withor:
        print("\n== calibration failure vs genuine RANKING degradation ==")
        print(f"  runs with oracle_cost    : {len(withor)}")
        print(f"  median d_oracle_cost     : {_med(c['d_oracle'] for c in withor):+.3f}  "
              f"(median d_cost {_med(c['d_cost'] for c in withor):+.3f})")  # fmt: skip
        print(f"  median ranking share     : {_med(c['ranking_share'] for c in withor):.2f}")
        print("    (share of the cost rise that survives an oracle threshold = ranking damage)")
        for lab in ("ranking", "mixed", "calibration"):
            print(f"    {lab:<12} {_share(withor, lambda c, L=lab: c['failure'] == L):.0%}")
        print(f"  ORACLE ITSELF ROSE >0.05 : {_share(withor, lambda c: (c['d_oracle'] or 0) > 0.05):.0%}"
              " <- more labels degraded the RANKING")  # fmt: skip
    noor = [c for c in ch if c["ranking_share"] is None]
    if noor:
        print(f"\n  ({len(noor)} detections come from a source with no oracle_cost column, so the "
              "ranking/calibration split is undecided for them.)")  # fmt: skip
    withap = [c for c in ch if c["d_ap"] is not None]
    if withap:
        print("\n== SECONDARY (not the #2825 metric): average_precision over the same segment ==")
        print("  Context only — cost=FNR+FPR above is what the study is about; AP is a different metric")
        print("  and is never used to classify a failure.")
        print(f"  runs with AP             : {len(withap)}")
        print(f"  median d_AP over segment : {_med(c['d_ap'] for c in withap):+.4f}   "
              f"share AP fell: {_share(withap, lambda c: c['d_ap'] < 0):.0%}")  # fmt: skip
        print(f"  median d_AUROC           : {_med(c['d_auroc'] for c in withap):+.4f}")

    print(f"\n== worst {min(cfg.top, len(dets))} runs (severity = rise x persistence) ==")
    for d in sorted(dets, key=lambda d: -d["severity"])[:cfg.top]:  # fmt: skip
        m, c = d["meta"], d["char"]
        orx = f" oracle {c['d_oracle']:+.3f}" if c["d_oracle"] is not None else ""
        apx = f" AP {c['d_ap']:+.3f}" if c["d_ap"] is not None else ""
        print(
            f"  [{d['severity']:6.2f}] {m['dataset']}/{m['proposal']}/{m['embedder']}/{m['head'] or m['group']} "
            f"{m['class']:<14} s{m['seed']:<3} {d['kind']:<9} "
            f"t{c['t_start']}->{c['t_end']} cost {c['cost_start']:.3f}->{c['cost_end']:.3f} "
            f"(fnr {c['d_fnr']:+.3f} fpr {c['d_fpr']:+.3f}){orx}{apx} g{c['n_good_start']}b{c['n_bad_start']}"
        )


def head_independence(runs, dets):
    """Same (embedder, class, seed) across model families: data problem or model problem?"""
    by_cell = defaultdict(dict)
    for r in runs:
        m = r.meta
        if m["head"] not in HEADS:
            continue
        by_cell[(m["source"], m["dataset"], m["proposal"], m["embedder"], m["class"], m["seed"])][m["head"]] = False
    for d in dets:
        m = d["meta"]
        if m["head"] not in HEADS:
            continue
        cell = (m["source"], m["dataset"], m["proposal"], m["embedder"], m["class"], m["seed"])
        if cell in by_cell:
            by_cell[cell][m["head"]] = True
    full = {c: h for c, h in by_cell.items() if len(h) >= 3}
    if not full:
        return
    hit = {c: [k for k, v in h.items() if v] for c, h in full.items()}
    affected = {c: hs for c, hs in hit.items() if hs}
    print("\n== head-independence (cells with >=3 heads run) ==")
    print(f"  cells: {len(full)}   cells with >=1 wrong-way head: {len(affected)}")
    dist = defaultdict(int)
    for hs in affected.values():
        dist[len(hs)] += 1
    for n in sorted(dist):
        print(f"    {n} of the heads affected: {dist[n]} cells ({dist[n] / max(1, len(affected)):.0%})")
    both = sum(1 for hs in affected.values() if set(hs) & LINEAR_FAMILY and set(hs) & MLP_FAMILY)
    only_mlp = sum(1 for hs in affected.values() if not (set(hs) & LINEAR_FAMILY))
    only_lin = sum(1 for hs in affected.values() if not (set(hs) & MLP_FAMILY))
    print(f"  hits BOTH a linear and an MLP family head : {both} ({both / max(1, len(affected)):.0%})"
          "  <- head-independent => data/acquisition")  # fmt: skip
    print(f"  MLP-family only                           : {only_mlp} ({only_mlp / max(1, len(affected)):.0%})"
          "  <- model-flexibility")  # fmt: skip
    print(f"  linear-family only                        : {only_lin} ({only_lin / max(1, len(affected)):.0%})")


def show_runs(runs, spec: str, cfg):
    """Dump the per-step table for the runs matching ``key=value,...`` — the deep-dive view."""
    want = dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)
    sel = [r for r in runs if all(str(r.meta.get(k)) == v for k, v in want.items())]
    print(f"\n== per-step detail for {spec}: {len(sel)} run(s) ==")
    for run in sel:
        det = detect(run, cfg)
        mark = set(range(det["seg_i"], det["seg_j"] + 1)) if det else set()
        print(f"\n  {run.meta}")
        if det:
            print(f"  detected: {det['kind']} severity {det['severity']:.3f} "
                  f"(segment marked '>')")  # fmt: skip
        cols = ["t", "cost", "oracle_cost", "fnr", "fpr", "threshold", "n_good", "n_bad",
                "phase", "calib_mode", "average_precision"]  # fmt: skip
        cols = [c for c in cols if any(c in st for st in run.steps)]
        print("   " + " ".join(f"{c[:9]:>10}" for c in cols))
        for i, st in enumerate(run.steps):
            cells = []
            for c in cols:
                v = st.get(c)
                cells.append(f"{v:>10.4f}" if isinstance(v, float) else f"{str(v):>10}")
            print(("  >" if i in mark else "   ") + " ".join(cells))


# --------------------------------------------------------------------------- cli


def build_parser():
    ap = argparse.ArgumentParser(description="Sustained wrong-way cost trends (#2825).")
    ap.add_argument("--jsonl-root", action="append", default=[], metavar="LABEL=DIR",
                    help="tree of results.jsonl; repeatable")  # fmt: skip
    ap.add_argument("--csv-source", action="append", default=[], metavar="LABEL=PATH",
                    help="max-patch cells.csv(.gz); repeatable")  # fmt: skip
    ap.add_argument("--k", type=int, default=5, help="min consecutive steps in the rising segment")
    ap.add_argument("--delta", type=float, default=0.10, help="min sustained rise in cost")
    ap.add_argument("--hold", type=int, default=5, help="min steps the elevated cost must persist")
    ap.add_argument("--min-up-frac", type=float, default=0.5, help="min fraction of upward steps in the segment")
    ap.add_argument("--smooth", type=int, default=3, help="rolling-median window (>=3 kills 1-step spikes)")
    ap.add_argument("--late-window", type=int, default=10, help="steps averaged for the late/final level")
    ap.add_argument("--regime", choices=("learned", "all", "coldstart"), default="learned")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--permute-null", type=int, default=0,
                    help="pooled null replicates per run, for the false-positive-rate line (0 = skip)")  # fmt: skip
    ap.add_argument("--per-run-null", type=int, default=99,
                    help="permutation replicates used to significance-test each detection (0 = magnitude only)")  # fmt: skip
    ap.add_argument("--null-alpha", type=float, default=0.05, help="permutation p-value a detection must clear")
    ap.add_argument("--sample", type=int, default=0,
                    help="deterministically subsample this many runs (calibration only, 0 = all)")  # fmt: skip
    ap.add_argument("--null-block", type=int, default=5,
                    help="block length for the surrogate reshuffle; 1 = shuffle single moves")  # fmt: skip
    ap.add_argument("--fp-check", type=int, default=0,
                    help="run the FULL pipeline on N surrogates per run and report the type-I error rate")  # fmt: skip
    ap.add_argument("--null-mode", choices=("demean", "raw"), default="demean",
                    help="surrogate construction; see permuted_copy")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None, help="write the detected runs here")
    ap.add_argument("--strict-head", action="store_true", help="fail if a head cannot be read from the path")
    ap.add_argument("--power-rise", type=float, default=0.0,
                    help="plant a sustained rise of this size in every run and report how often it is found")  # fmt: skip
    ap.add_argument("--show", action="append", default=[], metavar="k=v,k=v",
                    help="dump the per-step table for matching runs (deep-dive); repeatable")  # fmt: skip
    return ap


def _split(spec, default_label):
    if "=" in spec:
        lab, _, p = spec.partition("=")
        return lab, Path(p)
    return default_label, Path(spec)


def main(argv=None):
    cfg = build_parser().parse_args(argv)
    runs = []
    for spec in cfg.jsonl_root:
        lab, p = _split(spec, "jsonl")
        got = load_jsonl_tree(p, lab, cfg.strict_head)
        print(f"[load] {lab}: {len(got)} runs from {p}")
        runs += got
    for spec in cfg.csv_source:
        lab, p = _split(spec, "csv")
        got = load_maxpatch_csv(p, lab)
        print(f"[load] {lab}: {len(got)} runs from {p}")
        runs += got
    if not runs:
        raise SystemExit("no runs loaded")
    if cfg.sample and cfg.sample < len(runs):
        step = len(runs) / cfg.sample  # even stride keeps every source represented
        runs = [runs[int(i * step)] for i in range(cfg.sample)]
        print(f"[sample] calibrating on {len(runs)} runs (even stride)")

    rng = random.Random(cfg.seed)
    dets, raw_hits = [], 0
    for r in runs:
        d = detect(r, cfg)
        if d is None:
            continue
        raw_hits += 1
        if cfg.per_run_null:
            d["p"] = permutation_p(d["severity"], null_severities(r, cfg, cfg.per_run_null, rng))
            if d["p"] > cfg.null_alpha:
                continue
        d["char"] = characterise(r, d)
        d["cost_traj"] = [round(v, 4) for v in r.series("cost") if v is not None]
        dets.append(d)
    if cfg.per_run_null:
        print(f"[null] {raw_hits} magnitude hits -> {len(dets)} survive the per-run permutation test "
              f"(N={cfg.per_run_null}, alpha={cfg.null_alpha})")  # fmt: skip

    null_rate = None
    if cfg.permute_null:
        hits = trials = 0
        for r in runs:
            for _ in range(cfg.permute_null):
                trials += 1
                if detect(permuted_copy(r, rng), cfg) is not None:
                    hits += 1
        null_rate = hits / max(1, trials)

    report(runs, dets, cfg, null_rate)
    head_independence(runs, dets)
    if cfg.power_rise:
        power_report(runs, cfg, cfg.power_rise, random.Random(cfg.seed + 1))
    if cfg.fp_check:
        fp_report(runs, cfg, cfg.fp_check, random.Random(cfg.seed + 2))
    for spec in cfg.show:
        show_runs(runs, spec, cfg)

    if cfg.out_json:
        Path(cfg.out_json).write_text(
            json.dumps(
                {
                    "config": vars(cfg),
                    "n_runs": len(runs),
                    "null_rate": null_rate,
                    "detections": [{k: v for k, v in d.items()} for d in dets],
                },
                indent=1,
                default=str,
            )
        )
        print(f"\nwrote {cfg.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
