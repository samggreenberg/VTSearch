"""The ship rule per ENVIRONMENT, not just per voting mode.

The per-mode split was built to stop a mean over two voting modes hiding a
disagreement between them.  It does -- and then the `binary` group turns out to
hold TWO environments (`siglip x whole_image` and the pair's `whole_image`)
which disagree with each other, so the same hazard reappears one level down.

Reuses `analyze_acq`'s own `_core_summary` on subsets of the trajectory frame
the report was written from, so these are the same statistics by the same code
path -- not a second implementation that could drift from it.
"""

import argparse
import pathlib
import sys

import common

common.setup_env()

import pandas as pd

sys.path.insert(0, ".")
import analyze_acq as A

ap = argparse.ArgumentParser(description="Per-environment ship rule for the #2877 pile run.")
ap.add_argument("--analysis", default="/expscratch/sgreenberg/acq-2877/analysis")
ap.add_argument("--figures", action="store_true", help="also draw one figure set per environment")
ap.add_argument("--markdown", default=None, help="write the report's per-environment tables here")
args = ap.parse_args()

ANALYSIS = pathlib.Path(args.analysis)
traj = pd.read_csv(ANALYSIS / "agg" / "trajectories.csv")
print(f"{len(traj)} trajectories\n")

# `traj["style"]`, never `traj.style`: `.style` is pandas' Styler accessor, so
# attribute access silently yields a Styler and `== "whole_image"` collapses to
# a scalar False -- which selects nothing and reports "0 trajectories" rather
# than raising.  Two groups came back empty before the third one crashed loudly
# enough to notice.
emb = traj["embedder"]
sty = traj["style"]
groups = [
    ("siglip x whole_image  (binary)", (emb == "siglip") & (sty == "whole_image"), "env_siglip_"),
    ("pair   x whole_image  (binary)", (emb != "siglip") & (sty == "whole_image"), "env_pair_wi_"),
    ("pair   x max_patch    (REGION)", sty == "max_patch", "env_pair_region_"),
]

for label, mask, prefix in groups:
    sub = traj[mask]
    s = A._core_summary(sub)
    sz = A.sizing(sub)
    n_per_arm = int(len(sub) / max(1, sub.arm.nunique()))
    print(f"=== {label}   {len(sub)} trajectories, {n_per_arm}/arm")
    print(
        f"{'arm':10s} {'pos@100':>8s} {'cost':>7s} {'costCI':>20s} "
        f"{'AP':>7s} {'oracle':>7s} {'blips':>6s} {'acq_pct':>8s}  ADOPT"
    )
    for arm in A.ARMS:
        v = s["per_arm"].get(arm)
        if not v:
            continue
        c = s["contrasts_vs_control"].get(arm, {})
        cost = c.get("final_cost", {})
        ci = f"[{cost['ci95_lo']:+.4f},{cost['ci95_hi']:+.4f}]" if cost.get("n_pairs") else "-- control --"
        ship = s["ship_rule"].get(arm, {})
        mark = "YES" if ship.get("ADOPT") else ("" if arm == A.CONTROL else "no")
        if arm == A.FALSIFIER:
            mark = "falsifier"
        print(
            f"{arm:10s} {v['median_positives_100']:8.1f} {v['median_final_cost']:7.3f} {ci:>20s} "
            f"{v['median_final_ap']:7.3f} {v['median_final_oracle_cost']:7.3f} "
            f"{100 * v['genuine_blip_rate']:5.1f}% "
            f"{s['lever_verification'][arm]['median_acq_pool_percentile']:8.4f}  {mark}"
        )
    # Deep-spike incidence is what the ship rule reads, and on this study it is
    # the BINDING criterion -- so print the paired contrast rather than the base
    # rate.  Both flags, because they are different claims: `has_deep` counts
    # every cost spike over the oracle, `has_genuine` keeps only those on a
    # ranking good enough for the threshold to be blamed (oracle <= 0.30, the
    # #2825 distinction).  An environment whose oracle cost sits above that line
    # has NO genuine spikes by construction, which is a fact about the
    # environment and not a clean bill of health.
    print(f"{'':10s} {'deep: ctl':>10s} {'arm':>7s} {'p':>8s}   {'genuine: ctl':>13s} {'arm':>7s} {'p':>8s}")
    for arm in A.ARMS:
        if arm == A.CONTROL or arm not in s["per_arm"]:
            continue
        c = s["contrasts_vs_control"][arm]
        d, g = c["deep_incidence"], c["genuine_incidence"]
        print(
            f"{arm:10s} {d.get('control_rate', 0):10.3f} {d.get('arm_rate', 0):7.3f} "
            f"{d.get('p_exact', 1):8.4f}   {g.get('control_rate', 0):13.3f} "
            f"{g.get('arm_rate', 0):7.3f} {g.get('p_exact', 1):8.4f}"
        )
    print(f"  falsifier behaved: {s['falsifier_behaved']}")
    print(f"  ADOPT: {s['adopt'] or 'none'}")
    if "n_for_target" in sz:
        print(f"  power: n≈{sz['n_for_target']} needed for ±0.010, {n_per_arm} delivered (SD {sz['binding_sd']:.4f})")
    # Figures per ENVIRONMENT, from this same frame -- the per-mode set the
    # analyzer draws pools the two binary environments, and the frontier's whole
    # point is its shape, so a curve averaged over two environments that
    # disagree is a curve of neither.
    if args.figures:
        made = A.make_figures(sub, s, ANALYSIS / "figures", prefix=prefix)
        print(f"  figures: {', '.join(made)}")
    print()


# --- the report's tables, GENERATED ---------------------------------------
# Hand-transcribing these is how a `genuine_blip_rate` ends up in a column
# headed "deep spikes": the two are different statistics (a genuine blip is a
# deep spike additionally restricted to rankings good enough for the threshold
# to be blamed, oracle <= 0.30) and they differ by an order of magnitude in the
# environment where the verdict turns on them.  So the tables the report prints
# are emitted from the same frame the ship rule was computed on.
def _p(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _stars(p):
    if p is None:
        return ""
    return " ***" if p < 1e-3 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))


if args.markdown:
    L = []
    for label, mask, _prefix in groups:
        sub = traj[mask]
        s = A._core_summary(sub)
        sz = A.sizing(sub)
        n_per_arm = int(len(sub) / max(1, sub["arm"].nunique()))
        L.append(f"\n### {label} — {n_per_arm} pairs/arm\n")
        L.append(
            "| arm | pos@100 | final cost | 95% CI on mean Δ cost | AP | oracle | "
            "deep spikes | p (spikes) | genuine blips | **ADOPT** |"
        )
        L.append("|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|")
        for arm in A.ARMS:
            v = s["per_arm"].get(arm)
            if not v:
                continue
            c = s["contrasts_vs_control"].get(arm, {})
            cost = c.get("final_cost", {})
            ci = f"[{cost['ci95_lo']:+.4f}, {cost['ci95_hi']:+.4f}]" if cost.get("n_pairs") else "—"
            deep = c.get("deep_incidence", {})
            drate = deep.get("arm_rate", s["per_arm"][arm]["deep_spike_rate"])
            dp = deep.get("p_exact")
            ship = s["ship_rule"].get(arm, {})
            if arm == A.CONTROL:
                mark = "—"
            elif arm == A.FALSIFIER:
                mark = "falsifier ✓" if s["falsifier_behaved"] else "**falsifier FAILED**"
            else:
                mark = "**yes**" if ship.get("ADOPT") else "no"
            L.append(
                f"| `{arm}` ({A.ARM_K.get(arm, 'pin')}) | {v['median_positives_100']:.0f} | "
                f"{_p(v['median_final_cost'])} | {ci} | {_p(v['median_final_ap'])} | "
                f"{_p(v['median_final_oracle_cost'])} | {100 * drate:.1f}% | "
                f"{('%.4f' % dp) if dp is not None else '—'} | "
                f"{100 * v['genuine_blip_rate']:.1f}% | {mark} |"
            )
        L.append(
            f"\nAdopted: **{', '.join(s['adopt']) if s['adopt'] else 'none'}**. "
            f"Falsifier behaved: {s['falsifier_behaved']}. "
            f"Power: n ≈ {sz.get('n_for_target', '?')} needed for ±0.010, {n_per_arm} delivered "
            f"(paired SD {_p(sz.get('binding_sd'), 4)}).\n"
        )
    pathlib.Path(args.markdown).write_text("\n".join(L) + "\n")
    print(f"wrote {args.markdown}")
