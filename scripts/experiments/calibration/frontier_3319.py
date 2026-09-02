"""#3319: where the acquisition-offset frontier turns, and at what resolution.

`analyze_acq` answers "is this arm better than the control?" for each arm
independently.  #3319 asks three questions that are about the SHAPE of the arm
sequence rather than about any one arm, and none of them is a by-product of the
per-arm table:

* **H1 - does the frontier turn?**  Read off the paired-vs-control deltas WITH
  their CIs.  Deliberately not a fitted curve: a parabola through twelve points
  manufactures an interior optimum whether or not one exists, which is exactly
  the claim under test.
* **H2 - is the optimum resolvable at finer than one step?**  A half step has to
  beat BOTH its integer neighbours, paired arm-to-arm on the same cells, by more
  than the tolerance.  It also has to BE a distinct operating point: an arm whose
  `acq_pool_percentile` matches a neighbour's cell-for-cell is a duplicate, and
  its comparison is refused rather than reported.
* **H4 - does the turn land on the posterior-flip landmark?**  Inclusion is a
  log2 likelihood-ratio threshold, so at prevalence pi the selector's picks
  become more likely Good than Bad at k* = -log2((1-pi)/pi).  On `vg_scale_any`
  (pi = 7.1% by construction) that is -3.71, which sits between two half steps
  and is invisible to an integer grid.

Reuses `analyze_acq`'s `_paired`/`trajectory_stats` so these are the same
statistics by the same code path, not a second implementation that could drift.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
import analyze_acq as A  # noqa: E402

ap = argparse.ArgumentParser(description="#3319 frontier shape, half-step resolution and the deep regime.")
ap.add_argument("--analysis", default="/expscratch/sgreenberg/acq-3319/analysis")
ap.add_argument("--deep", default=None, help="the 400-click wave's analysis dir, if it has run")
ap.add_argument("--incumbent", default="acq_m3", help="the arm a ship recommendation is made against")
ap.add_argument("--prevalence", type=float, default=0.071, help="designed prevalence, for the H4 landmark")
ap.add_argument("--markdown", default=None)
# The environment to read. Defaults to the SHIPPED arm; the region cross-check
# passes the pair. Never pooled: #3318 showed these two disagree, so a mean over
# them is precisely the number that would hide the disagreement.
ap.add_argument("--embedder", default="siglip")
ap.add_argument("--style", default="whole_image")
args = ap.parse_args()

TOL = A.COST_REGRESSION_TOLERANCE
ANALYSIS = pathlib.Path(args.analysis)
traj = pd.read_csv(ANALYSIS / "agg" / "trajectories.csv")

# The shipped arm only.  The region half is a separate wave and pooling it here
# would average the frontier over two environments that #3318 showed disagree.
traj = traj[(traj["embedder"] == args.embedder) & (traj["style"] == args.style)]
arms = [a for a in A.ARMS if a in set(traj["arm"])]
K = {a: A.ARM_K[a] for a in arms if a in A.ARM_K}
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


out(f"# #3319 frontier — `{args.embedder} x {args.style}` — {len(traj)} trajectories, {len(arms)} arms\n")
out(f"Arms present: {', '.join(arms)}")
out(f"Cost-regression tolerance: ±{TOL:g}\n")

# --- the lever, before anything is interpreted -------------------------------
# #2876's discipline: an arm whose sampling position did not move measured
# nothing, and reading its cost delta as a result is the failure mode.
out("## Did the lever move?\n")
out("| arm | k | median `acq_pool_percentile` | shift vs prod | cells |")
out("|---|---:|---:|---:|---:|")
ctl_pct = float(traj[traj["arm"] == A.CONTROL]["acq_pct"].median())
pct_by_arm = {}
for a in arms:
    sub = traj[traj["arm"] == a]
    pct = float(sub["acq_pct"].median())
    pct_by_arm[a] = pct
    out(f"| `{a}` | {K.get(a, float('nan')):g} | {pct:.4f} | {pct - ctl_pct:+.4f} | {len(sub)} |")
out()

# --- H2 prerequisite: are the half steps DISTINCT arms? ----------------------
out("## H2 prerequisite — are the half steps distinct operating points?\n")
out("An arm whose per-cell `acq_pool_percentile` matches a neighbour's is a")
out("duplicate produced by the quantile snap, not a finer grid point. Refused")
out("above 10% of cells, per the plan.\n")
out("| arm | neighbour | cells compared | identical | verdict |")
out("|---|---|---:|---:|---|")
keys = [k for k in A.PAIR_KEYS if k in traj.columns]
dup_refused: set[str] = set()
ordered = sorted([a for a in arms if a in K], key=lambda a: -K[a])
for a, b in zip(ordered, ordered[1:]):
    x = traj[traj["arm"] == a].set_index(keys)["acq_pct"]
    y = traj[traj["arm"] == b].set_index(keys)["acq_pct"]
    j = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if j.empty:
        continue
    same = float((np.abs(j["a"] - j["b"]) < 1e-12).mean())
    bad = same > 0.10
    if bad:
        dup_refused.update({a, b})
    out(f"| `{a}` | `{b}` | {len(j)} | {100 * same:.1f}% | {'**DUPLICATE — refused**' if bad else 'distinct'} |")
out()

# --- the frontier ------------------------------------------------------------
out("## The frontier — paired against `prod` (k=0)\n")
out("| arm | k | Δ final cost [95% CI] | Δ positives@100 | Δ AP | Δ oracle cost | pairs |")
out("|---|---:|---|---:|---:|---:|---:|")
front: dict[str, dict] = {}
for a in ordered:
    if a == A.CONTROL:
        continue
    c = A._paired(traj, "final_cost", a)
    p = A._paired(traj, "positives_100", a)
    ap_ = A._paired(traj, "final_ap", a)
    o = A._paired(traj, "final_oracle_cost", a)
    front[a] = {"k": K[a], "cost": c, "pos": p, "ap": ap_, "oracle": o}
    out(
        f"| `{a}` | {K[a]:g} | {c['mean_delta']:+.4f} [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] | "
        f"{p['mean_delta']:+.1f} | {ap_['mean_delta']:+.3f} | {o['mean_delta']:+.4f} | {c['n_pairs']} |"
    )
out()

# --- the falsifier -----------------------------------------------------------
f = front.get(A.FALSIFIER, {}).get("pos")
falsifier_ok = bool(f and f["mean_delta"] < 0)
out(
    f"**Falsification arm `{A.FALSIFIER}`**: Δ positives {f['mean_delta']:+.1f} — "
    f"{'behaves (positives fall, as required)' if falsifier_ok else '**DID NOT DEGRADE — verdict withheld**'}\n"
    if f
    else "**Falsification arm missing — verdict withheld.**\n"
)

# --- H1: does it turn? -------------------------------------------------------
neg = [a for a in ordered if K.get(a, 0) < 0 and a in front]
best = min(neg, key=lambda a: front[a]["cost"]["mean_delta"]) if neg else None
out("## H1 — does the frontier turn?\n")
if best is not None:
    kb = K[best]
    deeper = [a for a in neg if K[a] < kb]
    out(
        f"Minimum paired cost delta is at **`{best}` (k={kb:g})**, "
        f"{front[best]['cost']['mean_delta']:+.4f} "
        f"[{front[best]['cost']['ci95_lo']:+.4f}, {front[best]['cost']['ci95_hi']:+.4f}].\n"
    )
    # A turn is a RESOLVABLE rise past the minimum, not a point estimate that
    # happens to sit above it.  Contrasted arm-to-arm on the same cells, so the
    # comparison does not inherit the control's variance twice; the CI must
    # clear the tolerance, exactly as H2's test does.
    out("Is any arm deeper than the minimum **resolvably** worse than it?\n")
    out("| deeper arm | k | Δ cost vs the minimum [95% CI] | resolvably worse? |")
    out("|---|---:|---|---|")
    worse = []
    for a in deeper:
        d = A._paired(traj, "final_cost", a, control=best)
        if not d.get("n_pairs"):
            continue
        rose = d["ci95_lo"] > TOL
        if rose:
            worse.append(a)
        out(
            f"| `{a}` | {K[a]:g} | {d['mean_delta']:+.4f} "
            f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] | {'YES' if rose else 'no'} |"
        )
    out()
    turned = bool(worse)
    if not deeper:
        out(
            "The minimum is at the EDGE of the grid — the frontier has not turned "
            "within it, and H1 is **not settled** by this run.\n"
        )
    elif turned:
        out(
            f"Resolvably worse past the minimum: {', '.join('`%s`' % w for w in worse)}. "
            f"**The frontier turns — H1 supported.**\n"
        )
    else:
        out(
            "No arm deeper than the minimum is resolvably worse than it: the frontier "
            "is **flat past the optimum**, not turning. H1 falsified in its strong "
            "form — the knob has a plateau, not a peak, and the practical reading is "
            "that anything past the plateau's near edge buys positives for free.\n"
        )

# --- H4: the posterior-flip landmark ----------------------------------------
pi = args.prevalence
k_star = -math.log2((1.0 - pi) / pi)
out("## H4 — the posterior-flip landmark\n")
out(f"At prevalence π = {100 * pi:.1f}% the prior odds are {pi / (1 - pi):.4f}, so a")
out("selector's picks become more likely Good than Bad only at")
out(f"**k\\* = −log₂((1−π)/π) = {k_star:.2f}**.\n")
if best is not None:
    hit = -4.0 <= K[best] <= -3.5
    out(
        f"Measured minimum: k={K[best]:g}. Landmark: {k_star:.2f}. "
        f"**H4 {'SUPPORTED' if hit else 'NOT supported'}** "
        f"(pre-registered window [−4.0, −3.5]).\n"
    )

# --- H2: half-step resolution ------------------------------------------------
out("## H2 — is the optimum resolvable at finer than one step?\n")
out("Each half step paired **arm-to-arm** against both its integer neighbours on")
out("the same cells. A half step 'resolves' only if it beats both by more than")
out(f"the ±{TOL:g} tolerance.\n")
out("| half step | vs | Δ final cost [95% CI] | beats it? |")
out("|---|---|---|---|")
resolves = []
for h in [a for a in arms if a.endswith("h")]:
    if h in dup_refused:
        out(f"| `{h}` | — | — | **refused: duplicate of a neighbour** |")
        continue
    verdicts = []
    for nb in [a for a in arms if a in K and abs(K[a] - K[h]) == 0.5]:
        d = A._paired(traj, "final_cost", h, control=nb)
        if not d.get("n_pairs"):
            continue
        wins = d["ci95_hi"] < -TOL
        verdicts.append(wins)
        out(
            f"| `{h}` (k={K[h]:g}) | `{nb}` (k={K[nb]:g}) | "
            f"{d['mean_delta']:+.4f} [{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] | {'YES' if wins else 'no'} |"
        )
    if verdicts and all(verdicts):
        resolves.append(h)
out()
out(
    f"**H2 {'SUPPORTED' if resolves else 'FALSIFIED'}** — "
    + (
        f"{', '.join('`%s`' % r for r in resolves)} beats both integer neighbours.\n"
        if resolves
        else "no half step beats both of its integer neighbours; the knob's "
        "usable resolution is one bit and the integer grid was right.\n"
    )
)

# --- the ship comparison, against the incumbent ------------------------------
inc = args.incumbent
out(f"## The ship comparison — every arm against the incumbent `{inc}` (k={K.get(inc, float('nan')):g})\n")
out("| arm | k | Δ final cost [95% CI] | Δ positives | Δ AP | deep spikes | passes ship rule |")
out("|---|---:|---|---:|---:|---|---|")
ship: dict[str, bool] = {}
for a in ordered:
    if a == inc or K.get(a, 0) >= 0:
        continue
    c = A._paired(traj, "final_cost", a, control=inc)
    p = A._paired(traj, "positives_100", a, control=inc)
    apd = A._paired(traj, "final_ap", a, control=inc)
    sp_ = A._mcnemar(traj, a, "has_genuine", control=inc)
    if not c.get("n_pairs"):
        continue
    spike_rose = bool(sp_.get("arm_rate", 0) > sp_.get("control_rate", 0) and sp_.get("p_exact", 1.0) < A.ALPHA)
    passes = (c["ci95_hi"] < TOL) and (p["mean_delta"] > 0) and not spike_rose and a not in dup_refused
    ship[a] = passes
    out(
        f"| `{a}` | {K[a]:g} | {c['mean_delta']:+.4f} [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] | "
        f"{p['mean_delta']:+.1f} | {apd['mean_delta']:+.3f} | "
        f"{100 * sp_.get('control_rate', float('nan')):.1f}% → {100 * sp_.get('arm_rate', float('nan')):.1f}%"
        f"{' **(rise)**' if spike_rose else ''} | {'YES' if passes else 'no'} |"
    )
out()

# --- H3: the deep regime -----------------------------------------------------
if args.deep:
    dpath = pathlib.Path(args.deep) / "agg" / "trajectories.csv"
    if dpath.exists():
        dt = pd.read_csv(dpath)
        dt = dt[(dt["embedder"] == args.embedder) & (dt["style"] == args.style)]
        out("## H3 — the deep regime (400 clicks)\n")
        out("| arm | k | Δ final cost vs prod [95% CI] | Δ positives | pairs |")
        out("|---|---:|---|---:|---:|")
        darms = [a for a in A.ARMS if a in set(dt["arm"]) and a != A.CONTROL]
        dbest = None
        for a in sorted(darms, key=lambda a: -A.ARM_K.get(a, 0)):
            c = A._paired(dt, "final_cost", a)
            p = A._paired(dt, "positives_100", a)
            if not c.get("n_pairs"):
                continue
            out(
                f"| `{a}` | {A.ARM_K.get(a, float('nan')):g} | "
                f"{c['mean_delta']:+.4f} [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] | "
                f"{p['mean_delta']:+.1f} | {c['n_pairs']} |"
            )
            if A.ARM_K.get(a, 0) < 0 and (dbest is None or c["mean_delta"] < dbest[1]):
                dbest = (a, c["mean_delta"])
        out()
        if dbest and best:
            deep_k, shallow_k = A.ARM_K[dbest[0]], K[best]
            out(f"Optimum at 400 clicks: **k={deep_k:g}** (`{dbest[0]}`); at 100 clicks: **k={shallow_k:g}**.")
            out(
                f"**H3 {'SUPPORTED' if deep_k <= shallow_k else 'FALSIFIED'}** — the deep optimum is "
                f"{'at least as deep' if deep_k <= shallow_k else 'SHALLOWER'} as the 100-click one.\n"
            )
        # exhaustion: the artefact that would masquerade as 'the offset stops mattering'
        if "positives_100" in dt.columns:
            out("### Positive exhaustion\n")
            for a in sorted(darms + [A.CONTROL], key=lambda a: -A.ARM_K.get(a, 0)):
                s = dt[dt["arm"] == a]["positives_100"]
                if len(s):
                    out(
                        f"* `{a}`: median {s.median():.0f} positives found by t=400 "
                        f"(max {s.max():.0f}); the sim half holds ~150."
                    )
            out()
    else:
        out(f"## H3 — the deep regime\n\nNot yet available at `{dpath}`.\n")

summary = {
    "arms": arms,
    "k": K,
    "frontier": {a: {"k": v["k"], "cost": v["cost"], "positives": v["pos"]} for a, v in front.items()},
    "argmin_k": K.get(best) if best else None,
    "landmark_k_star": k_star,
    "h2_resolving_half_steps": resolves,
    "duplicates_refused": sorted(dup_refused),
    "falsifier_ok": falsifier_ok,
    "ship_vs_incumbent": ship,
}
(ANALYSIS / "frontier_3319.json").write_text(json.dumps(summary, indent=2, default=float))
if args.markdown:
    pathlib.Path(args.markdown).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.markdown}")
