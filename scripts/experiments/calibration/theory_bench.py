"""Stage 0 (cut-rule study, #2836): the cut rules against a *known* truth.

The real-data arm of this study can say which rule scored better on Visual
Genome.  It cannot say why, because on real data every quantity in the
derivation is estimated at once: the mixture is fitted, the classes are unknown,
the score distribution has no closed form, and the sim set is finite.  This bench
removes all four confounds by generating scores from a model of region voting
whose class-conditional distributions are known in closed form, so the exact
rate-optimal cut is computable and every rule's **excess loss is measured
against the truth rather than against a held-out sample**.

The generative model mirrors the geometry the production arm actually has:

* an image has ``m`` region nodes, each scored independently by the detector;
* a **negative** image's regions are all Bad: logits ``~ N(mu_bad, sd_bad)``;
* a **positive** image has one object region, ``~ N(mu_good, sd_good)``, and
  ``m - 1`` Bad ones;
* the image's score is the **max** over its regions (inference max-pool), pushed
  through a sigmoid — the score scale the mixture is actually fitted on.

That makes both class-conditional CDFs elementary::

    FPR(t) = 1 - Phi_bad(t)**m
    FNR(t) = Phi_good(t) * Phi_bad(t)**(m - 1)

so ``wf*FPR + wn*FNR`` can be minimised exactly on a grid, and any candidate cut
can be scored on the true loss with no Monte-Carlo noise at all.  Note what is
*absent* from those two lines: prevalence.  A rate loss is prevalence-free by
construction, which is the whole of the #2836 argument — so the prevalence sweep
below measures how much a rule that smuggles priors in pays for it, with the
correct answer held fixed.

``m = 1`` degenerates to the single-vector (whole-image) geometry, which is the
bench's internal control: the extreme-value story must vanish there.

Sweeps ``(m, prevalence, separation, sd ratio, sample size)`` and writes a tidy
per-configuration CSV, a phase diagram of the winning rule, and a decomposition
of every rule's error into misspecification (the same rule fitted on a huge
sample) versus estimation noise (the shortfall at realistic sample sizes).

Usage: ``python theory_bench.py [--reps 40] [--out DIR]``
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vtscore.eval.cut_rules import decomposition_cuts  # noqa: E402

#: Region counts.  1 = whole-image control (no max-pool, so no extreme-value
#: statistic); 24 ~ the production patch grid's usable region count.
M_VALUES: tuple[int, ...] = (1, 6, 24)
#: Positive-class prevalences.  The truth does not depend on these; a rule that
#: uses the mixture weights as priors does.
PREVALENCES: tuple[float, ...] = (0.005, 0.02, 0.05, 0.15)
#: Class separation in per-region logit units (mu_good - mu_bad).
SEPARATIONS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
#: sd_good / sd_bad.  The equal-variance case is where the midpoint and the
#: prior-free crossing coincide, so this sweep is what separates them.
SD_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)
#: Sim-set sizes.  The large one is the population proxy that isolates
#: misspecification from estimation noise.
SAMPLE_SIZES: tuple[int, ...] = (500, 2000)
POPULATION_N: int = 200_000

MU_BAD = 0.0
SD_BAD = 1.0

#: Cost weights.  Inclusion 0 -> (1, 1); the bench keeps the shipped default so
#: ``rate`` and ``priorfree`` coincide and the comparison is about priors alone.
FPR_WEIGHT = 1.0
FNR_WEIGHT = 1.0

#: Rules scored here.  ``supervised``/``sim_oracle`` read labels and are reported
#: as bounds, not candidates.
RULES: tuple[str, ...] = (
    "mid",
    "cross",
    "priorfree",
    "gumbel_cross",
    "gumbel_priorfree",
    "supervised",
    "sim_oracle",
)
CANDIDATE_RULES: tuple[str, ...] = ("mid", "cross", "priorfree", "gumbel_cross", "gumbel_priorfree")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return float(np.log(p) - np.log1p(-p))


def _normal_cdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    from scipy.special import ndtr  # noqa: PLC0415

    return ndtr((x - mu) / sd)


def true_rates(t_logit: np.ndarray, m: int, mu_good: float, sd_good: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``(FPR, FNR)`` at each logit threshold under the generative model."""
    phi_bad = _normal_cdf(t_logit, MU_BAD, SD_BAD)
    phi_good = _normal_cdf(t_logit, mu_good, sd_good)
    fpr = 1.0 - phi_bad**m
    fnr = phi_good * phi_bad ** (m - 1)
    return fpr, fnr


def true_loss(t_logit: np.ndarray, m: int, mu_good: float, sd_good: float) -> np.ndarray:
    fpr, fnr = true_rates(t_logit, m, mu_good, sd_good)
    return FPR_WEIGHT * fpr + FNR_WEIGHT * fnr


def population_optimum(m: int, mu_good: float, sd_good: float) -> tuple[float, float]:
    """``(tau_star_logit, min_loss)`` on a fine grid spanning both classes."""
    grid = np.linspace(MU_BAD - 6.0 * SD_BAD, mu_good + 6.0 * sd_good, 20_001)
    loss = true_loss(grid, m, mu_good, sd_good)
    i = int(np.argmin(loss))
    return float(grid[i]), float(loss[i])


def sample_scores(
    rng: np.random.Generator, n: int, m: int, prevalence: float, mu_good: float, sd_good: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(scores, labels)`` for *n* images: max over regions, then sigmoid."""
    labels = (rng.random(n) < prevalence).astype(np.float64)
    n_pos = int(labels.sum())
    bad = rng.normal(MU_BAD, SD_BAD, size=(n, m))
    logits = bad.max(axis=1)
    if n_pos and m >= 1:
        obj = rng.normal(mu_good, sd_good, size=n_pos)
        pos_idx = np.flatnonzero(labels == 1.0)
        # A positive's object region replaces one of its Bad regions.
        others = bad[pos_idx, 1:].max(axis=1) if m > 1 else np.full(n_pos, -np.inf)
        logits[pos_idx] = np.maximum(obj, others)
    return _sigmoid(logits), labels


def evaluate_config(
    rng: np.random.Generator,
    m: int,
    prevalence: float,
    separation: float,
    sd_ratio: float,
    n: int,
) -> dict:
    """One replicate: fit every rule on a sample, score each on the *true* loss."""
    mu_good = MU_BAD + separation
    sd_good = SD_BAD * sd_ratio
    tau_star, best_loss = population_optimum(m, mu_good, sd_good)

    scores, labels = sample_scores(rng, n, m, prevalence, mu_good, sd_good)
    cuts, params = decomposition_cuts(scores, labels, FPR_WEIGHT, FNR_WEIGHT)

    row: dict = {
        "m": m,
        "prevalence": prevalence,
        "separation": separation,
        "sd_ratio": sd_ratio,
        "n": n,
        "n_pos": int(labels.sum()),
        "tau_star_logit": tau_star,
        "best_loss": best_loss,
        "gmm_ok": params["gmm_ok"],
        "evt_ok": params["evt_ok"],
        "evt_loglik_gain": params["evt_loglik_gain"],
        "w_lo": params["w_lo"],
        "w_hi": params["w_hi"],
        "var_lo": params["var_lo"],
        "var_hi": params["var_hi"],
        "pred_offset_equal_var": params["pred_offset_equal_var"],
        "oracle_lo_sf_gauss": params["oracle_lo_sf_gauss"],
    }
    for rule in RULES:
        cut = cuts.get(rule, float("nan"))
        row[f"tau_{rule}"] = cut
        if not np.isfinite(cut):
            row[f"excess_{rule}"] = float("nan")
            continue
        t = _logit(float(cut))
        row[f"excess_{rule}"] = float(true_loss(np.array([t]), m, mu_good, sd_good)[0] - best_loss)
    # The offset identity, on a fit whose truth we know.
    if np.isfinite(row.get("tau_cross", np.nan)) and np.isfinite(row.get("tau_mid", np.nan)):
        row["actual_offset"] = row["tau_cross"] - row["tau_mid"]
        row["offset_residual"] = row["actual_offset"] - row["pred_offset_equal_var"]
    return row


def run_sweep(reps: int, seed: int = 20836) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = list(itertools.product(M_VALUES, PREVALENCES, SEPARATIONS, SD_RATIOS, SAMPLE_SIZES))
    common.log(f"theory bench: {len(grid)} configurations x {reps} reps = {len(grid) * reps} fits")
    rows = []
    for i, (m, prev, sep, sdr, n) in enumerate(grid):
        for _rep in range(reps):
            rows.append(evaluate_config(rng, m, prev, sep, sdr, n))
        if (i + 1) % 20 == 0:
            common.log(f"  {i + 1}/{len(grid)} configurations")
    return pd.DataFrame(rows)


def population_sweep(seed: int = 20837) -> pd.DataFrame:
    """The same rules at ``POPULATION_N``: what each rule costs with *no* estimation noise.

    Subtracting these from the finite-sample excesses splits every rule's error
    into the part that is the rule's own (misspecification + wrong loss) and the
    part that is small-sample jitter — the issue's third hypothesis, that the
    crossing may be the better rule and the worse estimator.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m, sep, sdr in itertools.product(M_VALUES, SEPARATIONS, SD_RATIOS):
        # Prevalence still matters here: it sets the mixture weights, and the
        # count-optimal rule reads those as priors even in the population limit.
        for prev in PREVALENCES:
            rows.append(evaluate_config(rng, m, prev, sep, sdr, POPULATION_N))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, pop: pd.DataFrame, outdir: Path) -> dict:
    excess_cols = [f"excess_{r}" for r in RULES]
    cand_cols = [f"excess_{r}" for r in CANDIDATE_RULES]

    by_config = df.groupby(["m", "prevalence", "separation", "sd_ratio", "n"])[excess_cols].mean().reset_index()
    by_config["winner"] = by_config[cand_cols].idxmin(axis=1).str.replace("excess_", "", regex=False)
    by_config.to_csv(outdir / "theory_by_config.csv", index=False)

    by_m_prev = df.groupby(["m", "prevalence"])[excess_cols].mean().reset_index()
    by_m_prev.to_csv(outdir / "theory_by_m_prevalence.csv", index=False)

    pop_by = pop.groupby(["m", "prevalence", "separation", "sd_ratio"])[excess_cols].mean().reset_index()
    pop_by.to_csv(outdir / "theory_population.csv", index=False)

    # Estimation-noise split: finite-sample excess minus population excess, on
    # the configurations both sweeps share.
    keys = ["m", "prevalence", "separation", "sd_ratio"]
    finite = df[df["n"] == min(SAMPLE_SIZES)].groupby(keys)[cand_cols].mean()
    noise = (finite - pop_by.set_index(keys)[cand_cols]).reset_index()
    noise.to_csv(outdir / "theory_estimation_noise.csv", index=False)

    winners = by_config["winner"].value_counts(normalize=True).to_dict()
    prod_like = by_config[(by_config["m"] == 24) & (by_config["prevalence"] <= 0.05)]
    return {
        "n_fits": int(len(df)),
        "winner_share_overall": {k: float(v) for k, v in winners.items()},
        "winner_share_production_like": {
            k: float(v) for k, v in prod_like["winner"].value_counts(normalize=True).to_dict().items()
        },
        "mean_excess_overall": {r: float(df[f"excess_{r}"].mean()) for r in RULES},
        "mean_excess_production_like": {
            r: float(df[(df["m"] == 24) & (df["prevalence"] <= 0.05)][f"excess_{r}"].mean()) for r in RULES
        },
        "mean_excess_population": {r: float(pop[f"excess_{r}"].mean()) for r in RULES},
        "mean_estimation_noise": {
            r: float(noise[f"excess_{r}"].mean()) for r in CANDIDATE_RULES if f"excess_{r}" in noise
        },
        "evt_loglik_gain_by_m": {str(int(m)): float(sub["evt_loglik_gain"].mean()) for m, sub in df.groupby("m")},
        "offset_identity": {
            "mean_abs_residual": float(df["offset_residual"].abs().mean()) if "offset_residual" in df else None,
            "corr": (
                float(df[["actual_offset", "pred_offset_equal_var"]].dropna().corr().iloc[0, 1])
                if "actual_offset" in df
                else None
            ),
        },
    }


def make_figures(df: pd.DataFrame, outdir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        common.log(f"matplotlib unavailable ({e}); skipping figures")
        return []
    saved = []

    # Excess loss vs prevalence, one panel per region count.  The prior-bearing
    # rule should fan out as prevalence falls; the prior-free ones should not.
    fig, axes = plt.subplots(1, len(M_VALUES), figsize=(5 * len(M_VALUES), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], M_VALUES, strict=False):
        sub = df[df["m"] == m]
        for rule in CANDIDATE_RULES:
            g = sub.groupby("prevalence")[f"excess_{rule}"].mean()
            ax.plot(g.index, g.to_numpy(), marker="o", lw=1.3, label=rule)
        g = sub.groupby("prevalence")["excess_sim_oracle"].mean()
        ax.plot(g.index, g.to_numpy(), "k--", lw=1.0, label="sim_oracle (labels)")
        ax.set_xscale("log")
        ax.set_title(f"m = {m} region(s)")
        ax.set_xlabel("prevalence")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("excess true rate loss")
    axes[0][-1].legend(fontsize=7)
    p = outdir / "theory_excess_vs_prevalence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    # Phase diagram: winning rule over (separation, sd_ratio) at each m.
    by = df.groupby(["m", "separation", "sd_ratio"])[[f"excess_{r}" for r in CANDIDATE_RULES]].mean().reset_index()
    by["winner"] = by[[f"excess_{r}" for r in CANDIDATE_RULES]].idxmin(axis=1).str.replace("excess_", "", regex=False)
    codes = {r: i for i, r in enumerate(CANDIDATE_RULES)}
    fig, axes = plt.subplots(1, len(M_VALUES), figsize=(4.6 * len(M_VALUES), 3.8), squeeze=False)
    for ax, m in zip(axes[0], M_VALUES, strict=False):
        sub = by[by["m"] == m]
        grid = sub.pivot(index="sd_ratio", columns="separation", values="winner")
        num = grid.map(lambda r: codes.get(r, -1))
        ax.imshow(num.to_numpy(), aspect="auto", cmap="tab10", vmin=0, vmax=9, origin="lower")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{i:g}" for i in grid.index])
        ax.set_xlabel("separation")
        ax.set_title(f"m = {m}")
        for yi, yv in enumerate(grid.index):
            for xi, xv in enumerate(grid.columns):
                ax.text(xi, yi, str(grid.loc[yv, xv])[:9], ha="center", va="center", fontsize=6, color="w")
    axes[0][0].set_ylabel("sd_good / sd_bad")
    p = outdir / "theory_phase_diagram.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=40, help="replicates per configuration")
    parser.add_argument("--out", default=str(common.RESULTS / "theory"))
    args = parser.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(args.reps)
    df.to_csv(outdir / "theory_raw.csv", index=False)
    common.log("finite-sample sweep done; running the population sweep")
    pop = population_sweep()
    summary = summarize(df, pop, outdir)
    summary["figures"] = make_figures(df, outdir)
    (outdir / "theory_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    common.log(f"theory bench complete -> {outdir}")
    common.log(json.dumps(summary["mean_excess_production_like"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
