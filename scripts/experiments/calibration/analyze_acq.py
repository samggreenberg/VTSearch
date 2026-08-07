"""Does decoupling the acquisition cut buy back the positives the fused threshold costs?

Design and pre-registered decision rules:
``docs/plans/acquisition-inclusion-decoupling.md``.  Background: #2847 / PR #2873
found production finds half as many positives as the conformal path it replaced
(median 9 -> 4 per 100 votes, p=1e-20) with final cost unchanged (p=0.09),
because one number is both the reported decision line and the rank position
Autopilot's ``hard`` pick samples around.

Arms move **only** the selector's cut; reporting stays at inclusion 0 throughout,
so cost is comparable across arms.

Three things this analyzer refuses to do quietly:

1. **Report a result without checking the lever moved.** Every arm's
   ``acq_pool_percentile`` is compared against its ``report_pool_percentile``.
   If the sampling position did not shift, the arm measured nothing and says so.
2. **Call a null a null without power.** The ship rule turns on cost *not*
   regressing, so the analyzer reports the CI on that delta, not just its
   p-value - a wide null and a tight null are different findings, and #2847's
   near-miss came from treating the first as the second.
3. **Read the falsification arm as decoration.** ``acq_p2`` must move positives
   the wrong way; if it does not, the mechanism is wrong and the verdict is
   withheld.

Writes ``agg/*.csv``, ``acq_summary.json``, ``figures/*.png`` and
``REPORT_acq.md`` under ``$CALIB_EXP/analysis``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_spikes as sp  # noqa: E402  (reuse the #2847 loader + spike rule)

try:
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # noqa: BLE001
    _wilcoxon = None

#: Arms in sweep order; ``prod`` is the control.
ARMS: tuple[str, ...] = ("acq_m4", "acq_m3", "acq_m2", "acq_m1", "prod", "acq_p2", "rank_pin")
CONTROL = "prod"
FALSIFIER = "acq_p2"
#: Nominal acquisition inclusion per arm, for the frontier's x-axis. ``rank_pin``
#: is not on the inclusion scale and is plotted separately.
ARM_K: dict[str, float] = {"acq_m4": -4, "acq_m3": -3, "acq_m2": -2, "acq_m1": -1, "prod": 0, "acq_p2": 2}
ARM_LABEL: dict[str, str] = {
    "prod": "prod (k=0, shipped)",
    "acq_m1": "k=-1",
    "acq_m2": "k=-2",
    "acq_m3": "k=-3",
    "acq_m4": "k=-4",
    "acq_p2": "k=+2 (falsifier)",
    "rank_pin": "rank-pinned 0.959",
}

#: Ship rule (pre-registered): positives must rise, cost must not regress by more
#: than this at the 95% upper bound, and deep-spike incidence must not rise.
COST_REGRESSION_TOLERANCE = float(os.environ.get("ACQ_COST_TOL", "0.01"))
ALPHA = 0.05

OUT = Path(os.environ.get("ACQ_OUT", str(common.EXP / "analysis")))


def load_all(root: Path) -> tuple[pd.DataFrame, dict]:
    parts, prov = [], {}
    for arm in ARMS:
        df, p = sp.load_arm(root / arm)
        prov[arm] = p
        if not df.empty:
            df = df.copy()
            df["arm"] = arm
            parts.append(df)
    if not parts:
        return pd.DataFrame(), prov
    return pd.concat(parts, ignore_index=True), prov


def trajectory_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(arm, category, seed)``, carrying every endpoint at once."""
    keys = [k for k in ("arm", "dataset", "embedder", "category", "seed") if k in df.columns]
    out = []
    for key, g in df.groupby(keys, dropna=False):
        g = g.sort_values("t")
        cost = g["cost"].to_numpy(dtype=float)
        orc = g["oracle_cost"].to_numpy(dtype=float)
        t = g["t"].to_numpy(dtype=float)
        warm = t >= sp.WARM_T
        excess = cost - orc
        deep = warm & (cost >= sp.DEEP_COST) & (excess >= sp.DEEP_EXCESS)
        # Ranking-limited steps are #2825's problem, not a threshold blip - the
        # #2847 report's distinction, kept so the guardrail means the same thing.
        genuine = deep & (orc <= 0.30)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=False))
        rec.update(
            n_steps=int(len(g)),
            # --- mechanism ---
            positives_100=int(g["n_good"].iloc[-1]),
            positives_50=int(g[g["t"] <= 50]["n_good"].max()) if (g["t"] <= 50).any() else 0,
            # --- decision ---
            final_cost=float(cost[-1]),
            mean_cost_warm=float(np.nanmean(cost[warm])) if warm.any() else np.nan,
            final_oracle_cost=float(orc[-1]),
            final_ap=float(g["average_precision"].iloc[-1]) if "average_precision" in g.columns else np.nan,
            # --- guardrails ---
            has_deep=bool(deep.any()),
            has_genuine=bool(genuine.any()),
            max_excess_warm=float(np.nanmax(excess[warm])) if warm.any() else np.nan,
            # --- did the lever actually move? ---
            acq_pct=float(g["acq_pool_percentile"].median()) if "acq_pool_percentile" in g.columns else np.nan,
            report_pct=float(g["report_pool_percentile"].median()) if "report_pool_percentile" in g.columns else np.nan,
            acq_moved_frac=(
                float((g["acq_threshold"] != g["threshold"]).mean()) if "acq_threshold" in g.columns else np.nan
            ),
        )
        out.append(rec)
    return pd.DataFrame(out)


def _paired(traj: pd.DataFrame, metric: str, arm: str, control: str = CONTROL) -> dict:
    """Wilcoxon + a bootstrap CI on the paired ``(category, seed)`` delta.

    The CI is what the ship rule reads: "cost did not regress" is a claim about
    an interval, not a p-value.
    """
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in traj.columns]
    a = traj[traj["arm"] == control].set_index(keys)[metric]
    b = traj[traj["arm"] == arm].set_index(keys)[metric]
    j = pd.concat([a.rename("ctl"), b.rename("arm")], axis=1).dropna()
    if j.empty:
        return {"n_pairs": 0}
    d = (j["arm"] - j["ctl"]).to_numpy(dtype=float)
    rng = np.random.default_rng(12345)  # fixed: a CI must not move between runs
    boot = np.array([np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(4000)])
    res = {
        "n_pairs": int(len(j)),
        "control_median": float(j["ctl"].median()),
        "arm_median": float(j["arm"].median()),
        "median_delta": float(np.median(d)),
        "mean_delta": float(np.mean(d)),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "frac_arm_higher": float((d > 0).mean()),
    }
    if _wilcoxon is not None and np.any(d != 0):
        try:
            res["p"] = float(_wilcoxon(j["ctl"], j["arm"]).pvalue)
        except Exception:  # noqa: BLE001
            pass
    return res


def _mcnemar(traj: pd.DataFrame, arm: str, flag: str, control: str = CONTROL) -> dict:
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in traj.columns]
    a = traj[traj["arm"] == control].set_index(keys)[flag]
    b = traj[traj["arm"] == arm].set_index(keys)[flag]
    j = pd.concat([a.rename("ctl"), b.rename("arm")], axis=1).dropna()
    if j.empty:
        return {"n_pairs": 0}
    n01 = int(((~j["ctl"].astype(bool)) & j["arm"].astype(bool)).sum())
    n10 = int((j["ctl"].astype(bool) & (~j["arm"].astype(bool))).sum())
    out = {
        "n_pairs": int(len(j)),
        "control_rate": float(j["ctl"].astype(bool).mean()),
        "arm_rate": float(j["arm"].astype(bool).mean()),
        "only_control": n10,
        "only_arm": n01,
    }
    n = n01 + n10
    if n:
        from math import comb

        k = min(n01, n10)
        out["p_exact"] = float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2.0**n)))
    return out


def verify_lever(traj: pd.DataFrame) -> dict:
    """Did each arm's sampling position actually move, and which way?

    Without this the whole run is uninterpretable: an arm whose acquisition cut
    never budged looks exactly like an arm where the lever does nothing.
    """
    out = {}
    ctl_pct = float(traj[traj["arm"] == CONTROL]["acq_pct"].median())
    for arm in ARMS:
        a = traj[traj["arm"] == arm]
        if a.empty:
            continue
        pct = float(a["acq_pct"].median())
        out[arm] = {
            "median_acq_pool_percentile": pct,
            "median_report_pool_percentile": float(a["report_pct"].median()),
            "shift_vs_control": pct - ctl_pct,
            "frac_steps_acq_differs": float(a["acq_moved_frac"].median()),
            "moved": bool(arm == CONTROL or abs(pct - ctl_pct) > 1e-6),
        }
    return out


def build_summary(traj: pd.DataFrame, prov: dict) -> dict:
    per_arm = {}
    for arm in ARMS:
        a = traj[traj["arm"] == arm]
        if a.empty:
            continue
        per_arm[arm] = {
            "label": ARM_LABEL.get(arm, arm),
            "k_acq": ARM_K.get(arm),
            "n_trajectories": int(len(a)),
            "median_positives_100": float(a["positives_100"].median()),
            "median_positives_50": float(a["positives_50"].median()),
            "median_final_cost": float(a["final_cost"].median()),
            "median_mean_cost_warm": float(a["mean_cost_warm"].median()),
            "median_final_oracle_cost": float(a["final_oracle_cost"].median()),
            "median_final_ap": float(a["final_ap"].median()),
            "deep_spike_rate": float(a["has_deep"].mean()),
            "genuine_blip_rate": float(a["has_genuine"].mean()),
            "median_max_excess_warm": float(a["max_excess_warm"].median()),
        }

    contrasts = {}
    for arm in ARMS:
        if arm == CONTROL or arm not in per_arm:
            continue
        contrasts[arm] = {
            "positives_100": _paired(traj, "positives_100", arm),
            "positives_50": _paired(traj, "positives_50", arm),
            "final_cost": _paired(traj, "final_cost", arm),
            "mean_cost_warm": _paired(traj, "mean_cost_warm", arm),
            "final_oracle_cost": _paired(traj, "final_oracle_cost", arm),
            "deep_incidence": _mcnemar(traj, arm, "has_deep"),
            "genuine_incidence": _mcnemar(traj, arm, "has_genuine"),
        }

    lever = verify_lever(traj)

    # --- the pre-registered ship rule ---
    falsifier_ok = False
    if FALSIFIER in contrasts:
        f = contrasts[FALSIFIER]["positives_100"]
        falsifier_ok = f.get("median_delta", 0) < 0 and f.get("p", 1.0) < ALPHA

    ship = {}
    for arm, c in contrasts.items():
        if arm == FALSIFIER:
            continue
        pos, cost, deep = c["positives_100"], c["final_cost"], c["deep_incidence"]
        c1 = pos.get("median_delta", 0) > 0 and pos.get("p", 1.0) < ALPHA
        c2 = cost.get("ci95_hi", float("inf")) < COST_REGRESSION_TOLERANCE
        c3 = not (deep.get("arm_rate", 0) > deep.get("control_rate", 0) and deep.get("p_exact", 1.0) < ALPHA)
        ship[arm] = {
            "positives_rose": bool(c1),
            "cost_did_not_regress": bool(c2),
            "spikes_did_not_rise": bool(c3),
            "lever_moved": bool(lever.get(arm, {}).get("moved")),
            "ADOPT": bool(c1 and c2 and c3 and lever.get(arm, {}).get("moved")),
        }

    adopt = [a for a, v in ship.items() if v["ADOPT"]]
    return {
        "config": {
            "cost_regression_tolerance": COST_REGRESSION_TOLERANCE,
            "alpha": ALPHA,
            "warm_t": sp.WARM_T,
            "deep_cost": sp.DEEP_COST,
            "deep_excess": sp.DEEP_EXCESS,
        },
        "provenance": prov,
        "lever_verification": lever,
        "falsifier_behaved": falsifier_ok,
        "per_arm": per_arm,
        "contrasts_vs_control": contrasts,
        "ship_rule": ship,
        "adopt": adopt,
    }


def make_figures(traj: pd.DataFrame, summary: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    dpi = int(os.environ.get("ACQ_FIG_DPI", "200"))
    order = [a for a in ARMS if a in summary["per_arm"]]

    # Fig 1: the frontier - the deliverable. Positives against cost, per arm.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ks = [summary["per_arm"][a]["k_acq"] for a in order if summary["per_arm"][a]["k_acq"] is not None]
    karms = [a for a in order if summary["per_arm"][a]["k_acq"] is not None]

    axes[0].plot(ks, [summary["per_arm"][a]["median_positives_100"] for a in karms], "o-", color="#1f6f78")
    axes[0].set_xlabel("acquisition inclusion $k$"), axes[0].set_ylabel("median positives found by t=100")
    axes[0].set_title("mechanism: does the lever pull?", fontsize=9)
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.25, ls=":")

    axes[1].plot(ks, [summary["per_arm"][a]["median_final_cost"] for a in karms], "o-", color="#b0402e")
    axes[1].set_xlabel("acquisition inclusion $k$"), axes[1].set_ylabel("median final cost (FPR+FNR)")
    axes[1].set_title("decision: is it free?", fontsize=9)
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.25, ls=":")

    for a in order:
        x = summary["per_arm"][a]["median_positives_100"]
        y = summary["per_arm"][a]["median_final_cost"]
        m = "s" if a == "rank_pin" else ("D" if a == CONTROL else "o")
        axes[2].scatter(x, y, s=70, marker=m, zorder=4)
        axes[2].annotate(ARM_LABEL.get(a, a), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[2].set_xlabel("median positives found"), axes[2].set_ylabel("median final cost")
    axes[2].set_title("the frontier (down-and-right is better)", fontsize=9)
    axes[2].grid(alpha=0.25, ls=":")

    fig.tight_layout()
    p = outdir / "fig1_frontier.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    made.append(p.name)

    # Fig 2: the verification panel - where each arm actually sampled.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pcts = [summary["lever_verification"][a]["median_acq_pool_percentile"] for a in order]
    ax.bar(range(len(order)), pcts, color="#4c72b0")
    ax.axhline(
        summary["lever_verification"][CONTROL]["median_acq_pool_percentile"],
        color="#b0402e",
        ls="--",
        lw=1.2,
        label="production (control)",
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ARM_LABEL.get(a, a) for a in order], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median acq_pool_percentile")
    ax.set_title("verification: where in the ranking each arm actually sampled", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y", ls=":")
    fig.tight_layout()
    p = outdir / "fig2_lever_verification.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    made.append(p.name)

    # Fig 3: guardrails.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(range(len(order)), [100 * summary["per_arm"][a]["genuine_blip_rate"] for a in order], color="#c44e52")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([ARM_LABEL.get(a, a) for a in order], rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("% runs with a genuine threshold blip")
    axes[0].set_title("guardrail: #2847's fix must survive", fontsize=9)
    axes[0].grid(alpha=0.25, axis="y", ls=":")

    data = [traj[traj["arm"] == a]["final_cost"].dropna().to_numpy() for a in order]
    axes[1].boxplot(data, tick_labels=[ARM_LABEL.get(a, a) for a in order], showfliers=False)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].set_ylabel("final cost per trajectory")
    axes[1].set_title("decision endpoint, full distribution", fontsize=9)
    axes[1].grid(alpha=0.25, axis="y", ls=":")
    fig.tight_layout()
    p = outdir / "fig3_guardrails.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    made.append(p.name)
    return made


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def write_report(summary: dict, figs: list[str], outdir: Path) -> Path:
    L: list[str] = []
    A = L.append
    A("# Acquisition/reporting threshold decoupling — does it buy back the positives?\n")
    A(
        "Design: `docs/plans/acquisition-inclusion-decoupling.md`. Reporting is cut at "
        "inclusion 0 in **every** arm; only the selector's cut moves.\n"
    )

    if not summary["falsifier_behaved"]:
        A(
            "> **VERDICT WITHHELD.** The falsification arm `acq_p2` (k=+2) did not "
            "significantly *reduce* positives found. The mechanism this run assumes — "
            "that the acquisition cut's rank position drives positive yield — is "
            "therefore unsupported, and the remaining arms cannot be read as evidence "
            "for it. Everything below is descriptive.\n"
        )

    bad_levers = [a for a, v in summary["lever_verification"].items() if not v["moved"]]
    if bad_levers:
        A(f"> **Arms whose sampling position never moved: {', '.join(bad_levers)}.** These measured nothing.\n")

    A("\n## Lever verification — where each arm actually sampled\n")
    A("| arm | median `acq_pool_percentile` | shift vs control | steps where acq ≠ reporting |")
    A("|---|---:|---:|---:|")
    for a in ARMS:
        v = summary["lever_verification"].get(a)
        if not v:
            continue
        A(
            f"| `{a}` — {ARM_LABEL.get(a, a)} | {_f(v['median_acq_pool_percentile'], 4)} | "
            f"{v['shift_vs_control']:+.4f} | {100 * v['frac_steps_acq_differs']:.0f}% |"
        )

    A("\n## Per-arm\n")
    A(
        "| arm | trajectories | positives @100 | positives @50 | final cost | mean warm cost | "
        "final AP | oracle cost | genuine blips |"
    )
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARMS:
        v = summary["per_arm"].get(a)
        if not v:
            continue
        A(
            f"| `{a}` — {ARM_LABEL.get(a, a)} | {v['n_trajectories']} | "
            f"{_f(v['median_positives_100'], 1)} | {_f(v['median_positives_50'], 1)} | "
            f"{_f(v['median_final_cost'])} | {_f(v['median_mean_cost_warm'])} | "
            f"{_f(v['median_final_ap'])} | {_f(v['median_final_oracle_cost'])} | "
            f"{100 * v['genuine_blip_rate']:.1f}% |"
        )

    A(f"\n## Paired against `{CONTROL}` — cells are `(category, seed)`, never steps\n")
    A("| arm | metric | n | control | arm | median Δ | 95% CI on mean Δ | p |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        c = summary["contrasts_vs_control"].get(arm)
        if not c:
            continue
        for metric in ("positives_100", "final_cost", "mean_cost_warm", "final_oracle_cost"):
            r = c[metric]
            if not r.get("n_pairs"):
                continue
            A(
                f"| `{arm}` | {metric} | {r['n_pairs']} | {_f(r['control_median'])} | {_f(r['arm_median'])} | "
                f"{r['median_delta']:+.3f} | [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {_f(r.get('p'), 5)} |"
            )

    A("\n## Ship rule (pre-registered)\n")
    A(
        f"Adopt iff positives rise (p<{ALPHA}) **and** the 95% upper bound on the "
        f"final-cost delta is below +{COST_REGRESSION_TOLERANCE} **and** deep-spike "
        "incidence does not rise **and** the lever actually moved.\n"
    )
    A("| arm | positives rose | cost did not regress | spikes did not rise | lever moved | **ADOPT** |")
    A("|---|:--:|:--:|:--:|:--:|:--:|")
    for arm, v in summary["ship_rule"].items():
        y = lambda b: "yes" if b else "no"  # noqa: E731
        A(
            f"| `{arm}` | {y(v['positives_rose'])} | {y(v['cost_did_not_regress'])} | "
            f"{y(v['spikes_did_not_rise'])} | {y(v['lever_moved'])} | **{y(v['ADOPT'])}** |"
        )
    A(f"\n**Arms passing every criterion:** {', '.join(summary['adopt']) if summary['adopt'] else '_none_'}\n")

    A("\n## Data read\n")
    A("| arm | cell files | trajectories | never found a positive | unreadable | zero-byte |")
    A("|---|---:|---:|---:|---:|---:|")
    for arm, v in summary["provenance"].items():
        A(
            f"| `{arm}` | {v.get('n_files', 0)} | {v.get('n_read', 0)} | "
            f"{len(v.get('no_positive_found', []))} | {len(v.get('unreadable', []))} | "
            f"{len(v.get('zero_byte', []))} |"
        )

    if figs:
        A("\n## Figures\n")
        for f in figs:
            A(f"![{f}](figures/{f})\n")

    out = outdir / "REPORT_acq.md"
    out.write_text("\n".join(L) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Acquisition-inclusion decoupling analysis.")
    ap.add_argument("--results-root", default=str(common.EXP / "results"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    (outdir / "agg").mkdir(parents=True, exist_ok=True)

    df, prov = load_all(Path(args.results_root))
    if df.empty:
        print(f"no cells under {args.results_root}")
        return 1
    print(f"loaded {len(df)} base rows across {df['arm'].nunique()} arms")

    traj = trajectory_stats(df)
    traj.to_csv(outdir / "agg" / "trajectories.csv", index=False)
    summary = build_summary(traj, prov)
    (outdir / "acq_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    figs = make_figures(traj, summary, outdir / "figures")
    rep = write_report(summary, figs, outdir)

    print(f"wrote {rep}")
    for a in ARMS:
        v = summary["per_arm"].get(a)
        if not v:
            continue
        print(
            f"  {a:9s} pos@100={v['median_positives_100']:5.1f}  final_cost={v['median_final_cost']:.3f}  "
            f"genuine_blips={100 * v['genuine_blip_rate']:5.1f}%  "
            f"acq_pct={summary['lever_verification'][a]['median_acq_pool_percentile']:.4f}"
        )
    print(f"  falsifier behaved: {summary['falsifier_behaved']}")
    print(f"  ADOPT: {summary['adopt'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
