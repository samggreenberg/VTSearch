"""Stage 2 (#2865): which cut rule should answer the Inclusion knob?

Consumes the ``results/cells/task_*__cutincl.csv`` side frames from a run
launched with ``CALIB_CUT_INCL_KS`` set - one row per (step, fold-anchored arm,
inclusion ``k``), each scored under the cost weights of *its own* ``k`` and
against the oracle cut at that same ``k``.

The issue names two decision numbers, and this script computes exactly those.

**(a) Paired regret at each k, against the incumbent.**  The incumbent is the
shipped rule (``mid_tilt`` at the production anchor weight and combine).  Every
arm re-cuts the *same* per-step anchored fit against the *same* test scores, so
the pairing unit is a full identity - (dataset, embedder, style, category, seed,
step, k) - and the difference isolates the cut rule alone.  Significance is a
paired bootstrap over **cells**, not over steps: consecutive steps of one
trajectory share a model and are nowhere near independent, so a step-level
interval would be badly over-confident.

**(b) How much of the knob's nominal range survives as distinct admitted sets.**
A rule that moves the threshold without moving the admitted set has not fixed
anything.  Because the cut is carried to the final model as a *quantile*, a cut
that lands inside an empty band between two well-separated modes realizes to the
same threshold no matter where in the band it sits - so ``mid_tilt`` can look
perfectly monotone in ``fold_quantile`` while every user-visible verdict stays
frozen.  Reported three ways per arm: distinct admitted sets across the swept
``k`` (the headline), the admitted-fraction span end to end, and the share of
adjacent ``k`` steps that moved the set at all (the "dead steps" rate, which is
what a user experiences as a slider that does nothing).

This second table is also the answer to the plan's *"Inclusion resolution on
cleanly separated haystacks"* item: ``flat_rate`` per environment says how often
real data sits in the regime where no cut rule can help.

Writes ``results/agg/cutincl_*.csv``, ``results/cutincl_summary.json`` and a
``results/REPORT_cutincl.md`` draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiment_config as cfg  # noqa: E402
from _cells_io import side_frame_files  # noqa: E402

#: Bootstrap resamples for the paired CIs.  Resampling is over cells, so the
#: unit count is in the hundreds and this is cheap.
N_BOOT = 2000

#: Deep regime: the vote count past which the estimator has enough anchors for
#: the cut rule (rather than the anchor supply) to be what is being measured.
DEEP_VOTES_MIN = 100

#: Non-inferiority margin on regret, in cost units.  An arm counts as *harmed*
#: at a ``(env, k)`` only when its whole CI sits above this - not merely above
#: zero.
#:
#: The margin is not slack, it is what makes the rule decidable at all.  This
#: sweep produces one interval per (arm, env, k) - order 100 of them - so a
#: "significantly worse anywhere" test rejects every arm, including a *perfect*
#: one, on multiplicity alone: 100 independent 95% intervals around a true zero
#: throw ~5 false alarms by construction.  Requiring the loss to be **material**
#: rather than merely detectable is what separates "this rule costs users
#: something" from "this sweep had enough steps to resolve a rounding error".
#:
#: 0.01 is the same tolerance PR #2891 pre-registered for the acquisition-offset
#: decision (see :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET`),
#: kept identical so two threshold decisions in the same subsystem are not
#: quietly held to different bars.
HARM_TOLERANCE = 0.01


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a fixed-width dump."""
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def incumbent_arm() -> str:
    """The shipped rule's arm name, built from the production constants.

    Read off :mod:`vtscore.training.thresholds` rather than hard-coded, so that
    if production's ``(kappa, rule, combine)`` moves, this analyzer compares
    against what actually ships instead of against a stale literal.
    """
    from vtscore.training.thresholds import (
        FOLD_ANCHOR_COMBINE,
        FOLD_ANCHOR_CUT_RULE,
        FOLD_ANCHOR_WEIGHT,
    )

    return f"fold_anchored_w{FOLD_ANCHOR_WEIGHT:g}_{FOLD_ANCHOR_CUT_RULE}_{FOLD_ANCHOR_COMBINE}"


def load_cutincl(cells_dir: Path) -> pd.DataFrame:
    files = side_frame_files(cells_dir, "__cutincl")
    frames = [f for f in (pd.read_csv(p) for p in files) if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["env"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["cell"] = df["env"] + "/" + df["category"] + "/s" + df["seed"].astype(str)
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {len(df):,} cut-inclusion rows from {len(files)} cells, {df['arm'].nunique()} arms")
    return df


# --------------------------------------------------------------- (a) regret


def _paired_bootstrap(per_cell: pd.Series, rng: np.random.Generator) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` of a per-cell mean difference, resampling cells."""
    values = per_cell.to_numpy(dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    draws = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def regret_vs_incumbent(df: pd.DataFrame, incumbent: str, agg: Path) -> pd.DataFrame:
    """Paired ``arm - incumbent`` regret at each ``k``, per environment.

    Averaged to one number per (cell, k) *before* the bootstrap so that a cell
    with more steps does not dominate, and so the resampling unit is genuinely
    exchangeable.
    """
    keys = ["env", "cell", "category", "seed", "t", "inclusion_k"]
    base = df[df["arm"] == incumbent].set_index(keys)["cut_regret"]
    if base.empty:
        common.log(f"WARNING: incumbent arm {incumbent!r} absent; no paired contrasts")
        return pd.DataFrame()

    rng = np.random.default_rng(12345)
    rows = []
    for arm, a in df.groupby("arm", observed=True):
        if arm == incumbent:
            continue
        joined = a.set_index(keys)["cut_regret"].to_frame("arm_regret").join(base.rename("inc_regret"), how="inner")
        if joined.empty:
            continue
        joined = joined.reset_index()
        joined["d"] = joined["arm_regret"] - joined["inc_regret"]
        for (env, k), g in joined.groupby(["env", "inclusion_k"], observed=True):
            per_cell = g.groupby("cell", observed=True)["d"].mean()
            mean, lo, hi = _paired_bootstrap(per_cell, rng)
            rows.append(
                {
                    "arm": arm,
                    "env": env,
                    "inclusion_k": int(k),
                    "n_cells": int(per_cell.size),
                    "n_steps": int(len(g)),
                    "d_regret": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "win_rate": float((g["d"] < 0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["env", "inclusion_k", "d_regret"])
        out.to_csv(agg / "cutincl_regret_vs_incumbent.csv", index=False)
    return out


# ------------------------------------------------------- (b) knob liveness


def knob_liveness(df: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """How much of the knob each arm actually delivers, per (arm, env).

    Computed per *step* (one trajectory point is one realization of the whole
    knob) and then averaged, because that is the unit a user experiences: at a
    given moment, how many distinct answers does dragging the slider produce?

    * ``distinct_admitted`` - distinct admitted sets across the swept ``k``.
    * ``knob_yield`` - that count as a share of the swept ``k``; 1.0 means every
      stop of the slider is its own answer, ``1/len(ks)`` means the knob is inert.
    * ``dead_step_rate`` - share of *adjacent* ``k`` pairs that admitted exactly
      the same set.  Distinguishes "inert everywhere" from "live in the middle,
      saturated at the ends", which want different fixes.
    * ``admitted_span`` - end-to-end range of the admitted fraction.
    * ``quantile_span`` - the same for the pre-realization fold quantile.  A big
      ``quantile_span`` with a small ``admitted_span`` is the empty-band failure:
      the rule is moving the cut, the haystack has nothing there.
    """
    rows = []
    for (arm, env, t, cell), g in df.groupby(["arm", "env", "t", "cell"], observed=True):
        g = g.sort_values("inclusion_k")
        adm = g["n_admitted"].to_numpy()
        rows.append(
            {
                "arm": arm,
                "env": env,
                "cell": cell,
                "t": t,
                "n_votes": int(g["n_votes"].iloc[0]),
                "n_ks": int(len(g)),
                "distinct_admitted": int(len(set(adm.tolist()))),
                "dead_steps": int(np.count_nonzero(np.diff(adm) == 0)),
                "adjacent_pairs": int(max(0, len(adm) - 1)),
                "admitted_span": float(g["admitted_frac"].max() - g["admitted_frac"].min()),
                "quantile_span": float(g["fold_quantile"].max() - g["fold_quantile"].min()),
            }
        )
    per_step = pd.DataFrame(rows)
    if per_step.empty:
        return per_step
    per_step["knob_yield"] = per_step["distinct_admitted"] / per_step["n_ks"]
    per_step["dead_step_rate"] = per_step["dead_steps"] / per_step["adjacent_pairs"].replace(0, np.nan)
    per_step.to_csv(agg / "cutincl_liveness_per_step.csv", index=False)

    g = (
        per_step.groupby(["arm", "env"], observed=True)
        .agg(
            n_steps=("t", "size"),
            distinct_admitted=("distinct_admitted", "mean"),
            knob_yield=("knob_yield", "mean"),
            dead_step_rate=("dead_step_rate", "mean"),
            admitted_span=("admitted_span", "mean"),
            quantile_span=("quantile_span", "mean"),
            # The fully-inert share: steps where the whole knob admits one set.
            inert_rate=("distinct_admitted", lambda s: float((s <= 1).mean())),
        )
        .reset_index()
        .sort_values(["env", "knob_yield"], ascending=[True, False])
    )
    g.to_csv(agg / "cutincl_liveness.csv", index=False)
    return g


def flatness_by_env(per_env: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """The plan's *"Inclusion resolution on cleanly separated haystacks"* item.

    Takes the **best** knob yield any rule achieves in each environment: what is
    left unmoved there is a property of the haystack, not of the rule, so it
    bounds what any cut rule could deliver.
    """
    if per_env.empty:
        return per_env
    g = (
        per_env.sort_values("knob_yield", ascending=False)
        .groupby("env", observed=True)
        .head(1)
        .rename(columns={"arm": "best_arm", "knob_yield": "best_knob_yield"})[
            ["env", "best_arm", "best_knob_yield", "dead_step_rate", "admitted_span", "quantile_span"]
        ]
        .sort_values("best_knob_yield")
    )
    g.to_csv(agg / "cutincl_env_flatness.csv", index=False)
    return g


# ------------------------------------------------------------------ verdict


def verdicts(regret: pd.DataFrame, liveness: pd.DataFrame, incumbent: str) -> dict:
    """The decision, mechanically stated.  The tables still get read.

    A challenger has to clear **both** bars: not lose materially on regret
    anywhere across the knob (a rule that is only better at one end is a rule
    that is worse at the other), and deliver strictly more of the knob than the
    incumbent.  A rule that clears only the liveness bar has bought motion with
    accuracy, which is the trade #2865 exists to price rather than to assume.

    The regret bar is deliberately *pointwise*, not pooled.  Pooling hides the
    failure this whole sweep exists to catch: an arm can win on average across
    the knob while being worse everywhere a user would actually park the slider,
    and the pooled mean would ship it.  ``d_regret_pooled`` is reported for
    orientation and is **not** what gates the recommendation.
    """
    out: dict = {"incumbent": incumbent, "n_boot": N_BOOT, "harm_tolerance": HARM_TOLERANCE}
    if regret.empty or liveness.empty:
        out["decidable"] = False
        return out

    pooled_regret = regret.groupby("arm", observed=True)["d_regret"].mean()
    # "Never *materially* worse at any k in any environment" - see HARM_TOLERANCE.
    harmed = regret[regret["ci_lo"] > HARM_TOLERANCE].groupby("arm", observed=True).size()
    pooled_yield = liveness.groupby("arm", observed=True)["knob_yield"].mean()
    inc_yield = float(pooled_yield.get(incumbent, float("nan")))

    table = pd.DataFrame(
        {
            "d_regret_pooled": pooled_regret,
            "n_harmed_cells": harmed.reindex(pooled_regret.index).fillna(0).astype(int),
            "knob_yield": pooled_yield.reindex(pooled_regret.index),
        }
    )
    table["beats_incumbent_knob"] = table["knob_yield"] > inc_yield
    table["no_regret_harm"] = table["n_harmed_cells"] == 0
    table["ships"] = table["beats_incumbent_knob"] & table["no_regret_harm"]

    out["incumbent_knob_yield"] = inc_yield
    out["arms"] = json.loads(table.reset_index().to_json(orient="records"))
    winners = table[table["ships"]].sort_values("d_regret_pooled")
    out["decidable"] = True
    out["recommended"] = str(winners.index[0]) if len(winners) else None
    out["reason"] = (
        f"no challenger both beat the incumbent's knob yield and stayed within {HARM_TOLERANCE} regret at every k"
        if not len(winners)
        else (
            "lowest pooled regret among arms that beat the incumbent's knob yield "
            f"without exceeding the {HARM_TOLERANCE} regret tolerance at any k"
        )
    )
    return out


def write_report(results: Path, regret: pd.DataFrame, liveness: pd.DataFrame, flat: pd.DataFrame, v: dict) -> None:
    lines = [
        "# Cut rule x Inclusion sweep (issue #2865) - draft",
        "",
        f"Incumbent (the shipped rule): `{v['incumbent']}`.",
        f"Swept k: {cfg.CUT_INCLUSION_KS}.  Anchor weights: {cfg.ANCHORED_WEIGHTS}.  "
        f"Rules: {cfg.ANCHORED_RULES}.",
        "",
        "Every row is scored at its own `k` against the oracle at that `k`, so",
        "regret is comparable along the knob as well as across arms.",
        "",
        "## Verdict (mechanical; read the tables before believing it)",
        "",
        "```json",
        json.dumps(v, indent=2),
        "```",
        "",
        "## (a) Paired regret vs the incumbent, per k",
        "",
        "Negative favours the challenger.  CI is a paired bootstrap over cells.",
        f"An arm counts as harmed at a `(env, k)` only when `ci_lo > {HARM_TOLERANCE}`",
        "- see `HARM_TOLERANCE` for why a bare significance test cannot decide this.",
        "",
        _md(regret),
        "",
        "## (b) How much of the knob survives as distinct admitted sets",
        "",
        "`knob_yield` 1.0 = every slider stop is its own answer; 1/len(ks) = inert.",
        "A large `quantile_span` beside a small `admitted_span` is the empty-band",
        "case: the rule moves the cut, the haystack has nothing there to move past.",
        "",
        _md(liveness),
        "",
        "## Haystack-imposed ceiling per environment (best rule's yield)",
        "",
        "What no cut rule can fix - the plan's *Inclusion resolution on cleanly",
        "separated haystacks* item.",
        "",
        _md(flat),
        "",
    ]
    (results / "REPORT_cutincl.md").write_text("\n".join(lines))


def main() -> int:
    results = common.RESULTS
    agg = results / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    df = load_cutincl(results / "cells")
    if df.empty:
        common.log("no cut-inclusion rows found; was CALIB_CUT_INCL_KS set on the run?")
        return 1

    deep = df[df["n_votes"] >= DEEP_VOTES_MIN]
    if deep.empty:
        common.log(f"WARNING: no steps past {DEEP_VOTES_MIN} votes; analyzing the whole trajectory instead")
        deep = df

    incumbent = incumbent_arm()
    regret = regret_vs_incumbent(deep, incumbent, agg)
    liveness = knob_liveness(deep, agg)
    flat = flatness_by_env(liveness, agg)
    v = verdicts(regret, liveness, incumbent)

    (results / "cutincl_summary.json").write_text(json.dumps(v, indent=2))
    write_report(results, regret, liveness, flat, v)
    common.log(f"wrote {results / 'cutincl_summary.json'} and {results / 'REPORT_cutincl.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
