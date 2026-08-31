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

Regret is reported on the **rate scale**, ``cut_regret / 2**abs(k)``, because
the cost the harness scores is ``fpr_weight*FPR + fnr_weight*FNR`` and
:func:`~vtscore.training.thresholds.inclusion_cost_weights` *doubles* one of
those weights per step of the knob.  Raw ``cut_regret`` at ``k=10`` is therefore
denominated in units 1024x the ones at ``k=0``: pooling it across the knob is
dominated entirely by the two end stops, and a fixed tolerance in cost units
means "a thousandth of an error rate" at one end of the slider and "a whole
error rate" at the other.  Dividing by the larger of the two weights - which is
``2**abs(k)``, and exactly 1 at inclusion 0 - restores a common unit (a weighted
mean of FPR and FNR, bounded like a rate) **without changing any number at
inclusion 0**, where every prior calibration study measured.  The raw
cost-unit difference rides along as ``d_regret_cost`` for anyone who wants it.

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
cleanly separated haystacks"* item: ``best_knob_yield`` per environment is the
ceiling the haystack itself imposes, so an environment where the *best* rule
still loses most of the knob is one where no cut rule can help.

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


#: Columns this analysis (and `make_cutincl_figs.py`) reads.  The frame is one
#: row per (step, arm, k) and a full run is ~10 M rows, so the loader takes only
#: these and hands the string columns to pandas as categories: the default
#: `read_csv` over the whole 34-column frame costs ~5x the memory for columns
#: nothing here groups on, which is how a 16G analysis step dies at the end of a
#: three-hour run rather than at its start.
_USECOLS: tuple[str, ...] = (
    "seed",
    "dataset",
    "embedder",
    "category",
    "head",
    "style",
    "t",
    "n_good",
    "n_bad",
    "arm",
    "cut_rule",
    "anchor_weight",
    "combine",
    "qtilt_step",
    "inclusion_k",
    "fold_quantile",
    "cut_threshold",
    "cut_cost",
    "cut_fpr",
    "cut_fnr",
    "k_oracle_cost",
    "cut_regret",
    "admitted_frac",
    "n_admitted",
    "n_test",
)

_DTYPES: dict[str, str] = {
    "dataset": "category",
    "embedder": "category",
    "category": "category",
    "head": "category",
    "style": "category",
    "arm": "category",
    "cut_rule": "category",
    "combine": "category",
    "seed": "int16",
    "t": "int32",
    "n_good": "int32",
    "n_bad": "int32",
    "inclusion_k": "int16",
    "n_admitted": "int32",
    "n_test": "int32",
    "anchor_weight": "float32",
    "qtilt_step": "float32",
    "fold_quantile": "float32",
    "cut_threshold": "float32",
    "cut_cost": "float32",
    "cut_fpr": "float32",
    "cut_fnr": "float32",
    "k_oracle_cost": "float32",
    "cut_regret": "float32",
    "admitted_frac": "float32",
}


def expected_cells(results: Path) -> int:
    """How many cells this grid *should* have produced, from the run's own config.

    An analysis that silently excludes cells is how a disk incident becomes a
    wrong verdict, so the count that matters is not "how many files did I find"
    but "how many did the grid define".
    """
    info_path = results / "prepare_info.json"
    if not info_path.exists():
        return 0
    info = json.loads(info_path.read_text())
    cats = {
        ds: {emb: entry.get("selected_categories", []) for emb, entry in per_emb.items()}
        for ds, per_emb in info.get("datasets", {}).items()
    }
    try:
        return len(cfg.array_cells(cats))
    except Exception:  # noqa: BLE001 - a config mismatch must not lose the analysis
        return 0


def load_cutincl(cells_dir: Path, provenance: dict | None = None) -> pd.DataFrame:
    files = side_frame_files(cells_dir, "__cutincl")
    prov = provenance if provenance is not None else {}
    prov["n_files"] = len(files)
    prov["unreadable"] = []
    prov["empty"] = []

    def _read(path: Path) -> pd.DataFrame:
        # A cell written by an older/other run may lack a column; fall back to
        # a plain read rather than dropping the cell silently.
        try:
            return pd.read_csv(path, usecols=list(_USECOLS), dtype=_DTYPES)
        except ValueError:
            return pd.read_csv(path)

    frames = []
    for path in files:
        try:
            f = _read(path)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not lose the rest
            prov["unreadable"].append(f"{path.name}: {exc}")
            continue
        if f.empty:
            prov["empty"].append(path.name)
            continue
        frames.append(f)
    if prov["unreadable"] or prov["empty"]:
        common.log(f"DROPPED {len(prov['unreadable'])} unreadable and {len(prov['empty'])} empty cell frames")
        for line in [*prov["unreadable"], *prov["empty"]]:
            common.log(f"  {line}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("dataset", "embedder", "style", "category", "arm", "cut_rule"):
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    env = df["dataset"].astype(str) + "/" + df["embedder"].astype(str) + "/" + df["style"].astype(str)
    df["env"] = env.astype("category")
    df["cell"] = (env + "/" + df["category"].astype(str) + "/s" + df["seed"].astype(str)).astype("category")
    df["n_votes"] = (df["n_good"] + df["n_bad"]).astype("int32")
    # The knob's own cost scale: `inclusion_cost_weights` doubles one weight per
    # step, so a cost at k=10 is denominated in units 1024x a cost at k=0.  See
    # the module docstring - without this every pooled number is the k=+-10 pair.
    df["k_scale"] = np.exp2(np.abs(df["inclusion_k"].to_numpy(dtype=np.float32)))
    df["regret_rate"] = (df["cut_regret"] / df["k_scale"]).astype("float32")
    common.log(
        f"loaded {len(df):,} cut-inclusion rows from {len(files)} cells, {df['arm'].nunique()} arms "
        f"({df.memory_usage(deep=True).sum() / 1e9:.1f} GB in memory)"
    )
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
    cols = ["regret_rate", "cut_regret"]
    base = df[df["arm"] == incumbent].set_index(keys)[cols]
    if base.empty:
        common.log(f"WARNING: incumbent arm {incumbent!r} absent; no paired contrasts")
        return pd.DataFrame()
    base = base.rename(columns={"regret_rate": "inc_rate", "cut_regret": "inc_cost"})

    rng = np.random.default_rng(12345)
    rows = []
    for arm, a in df.groupby("arm", observed=True):
        if arm == incumbent:
            continue
        joined = (
            a.set_index(keys)[cols]
            .rename(columns={"regret_rate": "arm_rate", "cut_regret": "arm_cost"})
            .join(base, how="inner")
        )
        if joined.empty:
            continue
        joined = joined.reset_index()
        joined["d"] = joined["arm_rate"] - joined["inc_rate"]
        joined["d_cost"] = joined["arm_cost"] - joined["inc_cost"]
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
                    # The same difference in raw cost units, for orientation:
                    # it is what the harness scored, and it is 2**abs(k) times
                    # the column the decision is taken on.
                    "d_regret_cost": float(g.groupby("cell", observed=True)["d_cost"].mean().mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["env", "inclusion_k", "d_regret"])
        out.to_csv(agg / "cutincl_regret_vs_incumbent.csv", index=False)
    return out


# ------------------------------------------------------- (b) knob liveness


def liveness_per_step(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, env, cell, step): what the whole slider did at that step.

    Split out of :func:`knob_liveness` so a cross-run analysis can pair these
    rows between two trajectories before they are averaged away - #3196 pairs
    them by cell across two *heads*, which cannot share a trajectory.  One
    implementation, because a second copy of this arithmetic is how two studies
    come to disagree about what "the knob is dead" means.

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
    # Vectorised: a full run is ~10 M rows and ~700 k (arm, env, cell, step)
    # groups, so a Python loop over `groupby` here ran ~40 min - longer than the
    # analysis step's own time limit.  Same quantities, one sort and one agg.
    cols = ["arm", "env", "cell", "t", "inclusion_k", "n_admitted", "admitted_frac", "fold_quantile", "n_votes"]
    g = df[cols].sort_values(["arm", "env", "cell", "t", "inclusion_k"], kind="stable")
    key = ["arm", "env", "cell", "t"]
    grp = g.groupby(key, observed=True, sort=False)
    # A "dead step" is an adjacent pair of k that admitted the same set.  The
    # diff is taken over the sorted frame, so the first row of each group has to
    # be masked out - otherwise a group whose first admitted count happens to
    # equal the previous group's last one counts a pair that does not exist.
    g = g.assign(_dead=(g["n_admitted"].diff().eq(0) & (grp.cumcount() > 0)).astype("int32"))
    per_step = (
        g.groupby(key, observed=True, sort=False)
        .agg(
            n_votes=("n_votes", "first"),
            n_ks=("inclusion_k", "size"),
            distinct_admitted=("n_admitted", "nunique"),
            dead_steps=("_dead", "sum"),
            adm_hi=("admitted_frac", "max"),
            adm_lo=("admitted_frac", "min"),
            q_hi=("fold_quantile", "max"),
            q_lo=("fold_quantile", "min"),
        )
        .reset_index()
    )
    per_step["adjacent_pairs"] = (per_step["n_ks"] - 1).clip(lower=0)
    per_step["admitted_span"] = per_step["adm_hi"] - per_step["adm_lo"]
    per_step["quantile_span"] = per_step["q_hi"] - per_step["q_lo"]
    per_step = per_step.drop(columns=["adm_hi", "adm_lo", "q_hi", "q_lo"])
    if per_step.empty:
        return per_step
    per_step["knob_yield"] = per_step["distinct_admitted"] / per_step["n_ks"]
    per_step["dead_step_rate"] = per_step["dead_steps"] / per_step["adjacent_pairs"].replace(0, np.nan)
    return per_step


def knob_liveness(df: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """How much of the knob each arm actually delivers, per (arm, env).

    Averages :func:`liveness_per_step` over steps and cells, because the per-step
    frame is the unit a user experiences: at a given moment, how many distinct
    answers does dragging the slider produce?
    """
    per_step = liveness_per_step(df)
    if per_step.empty:
        return per_step
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

    Regret enters here on the **rate scale** (`d_regret`, i.e. cost divided by
    `2**abs(k)`), so one tolerance means the same thing at every stop of the
    knob; see the module docstring.

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
        f"Swept k: {cfg.CUT_INCLUSION_KS}.  Anchor weights: {cfg.ANCHORED_WEIGHTS}.  Rules: {cfg.ANCHORED_RULES}.",
        "",
        (
            f"Cells: {v.get('cells', {}).get('n_files', '?')} read of "
            f"{v.get('cells', {}).get('n_expected', '?')} defined by the grid; "
            f"{len(v.get('cells', {}).get('unreadable', []))} unreadable, "
            f"{len(v.get('cells', {}).get('empty', []))} empty."
        ),
        "",
        "Every row is scored at its own `k` against the oracle at that `k`, and",
        "reported divided by `2**abs(k)`, so regret is comparable along the knob",
        "as well as across arms.",
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
        "`d_regret` is on the **rate scale** - raw cost divided by `2**abs(k)`, the",
        "larger of the two inclusion cost weights - so one tolerance means the same",
        "thing at every stop of the knob and inclusion 0 is unchanged.  The raw",
        "cost-unit difference the harness scored is `d_regret_cost`.",
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
    prov: dict = {}
    df = load_cutincl(results / "cells", prov)
    prov["n_expected"] = expected_cells(results)
    if prov["n_expected"]:
        missing = prov["n_expected"] - prov["n_files"]
        common.log(
            f"cells: {prov['n_files']} of {prov['n_expected']} expected ({missing} never wrote a cut-inclusion frame)"
        )
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
    v["cells"] = prov

    (results / "cutincl_summary.json").write_text(json.dumps(v, indent=2))
    write_report(results, regret, liveness, flat, v)
    common.log(f"wrote {results / 'cutincl_summary.json'} and {results / 'REPORT_cutincl.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
