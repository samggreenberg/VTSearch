"""Cross-head analysis (#3196): does the Inclusion knob still have authority?

Reads the ``__cutincl`` side frames of **both** head arms - the shipped linear
SVM and the logistic head it replaced - and answers the four pre-registered
questions in ``docs/experiments/inclusion-knob-3196/PLAN.md``:

* **H1** the head moved the knob: paired ``dead_step_rate(svm) - (linear)`` for
  the shipped rule, bootstrapped over cells.
* **H2** the knob has gone soft in absolute terms on the shipped head:
  ``dead_step_rate >= 0.5`` or ``admitted_span <= 0.05`` in the deep regime.
* **H3** whether ``q_tilt`` may ship: it must lower the dead-step rate where H2
  fired *and* stay inside the regret tolerance at every ``k`` everywhere.
* **H4** the acquisition offset's collapse: how often the selector's cut at
  ``k + ACQUISITION_INCLUSION_OFFSET`` admits exactly what reporting's cut at
  ``k`` admits.

**The pairing unit is the cell, not the step.** The threshold drives
acquisition, so the two heads collect different votes and step *t* of one arm is
not step *t* of the other; only the (environment, category, seed) identity
survives the head change. Everything is averaged to one number per cell before
any bootstrap, and the bootstrap resamples cells.

Everything shared with the #2865 analysis - the row loader, the per-step
liveness arithmetic, the paired regret table, the incumbent's name - is
*imported* from :mod:`analyze_cutincl` rather than reimplemented, so the two
studies cannot come to disagree about what their common words mean.

Writes ``<out>/incl3196_*.csv``, ``<out>/incl3196_summary.json`` and a
``<out>/REPORT_incl3196.md`` draft.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_cutincl import (  # noqa: E402
    DEEP_VOTES_MIN,
    HARM_TOLERANCE,
    N_BOOT,
    _md,
    _paired_bootstrap,
    expected_cells,
    incumbent_arm,
    liveness_per_step,
    load_cutincl,
    regret_vs_incumbent,
)

#: H2's two bars, pre-registered in the PLAN before the first cell landed.
#:
#: ``DEAD_MAX``: half the slider doing nothing is the point at which "the knob is
#: coarse" stops being a fair description of it.
#:
#: ``SPAN_MIN``: an admitted-fraction span of 0.05 means dragging the slider from
#: one end to the other changes fewer than one item in twenty.  Below that the
#: control is decorative whatever its threshold does, which is precisely the
#: distinction ``admitted_span`` exists to draw against ``quantile_span``.
DEAD_MAX = 0.5
SPAN_MIN = 0.05

#: The two head arms, in the order they are reported.  ``svm`` is the shipped
#: head and the one every product question is read off; ``linear`` is the
#: reference for what the knob used to do.
ARMS: tuple[str, ...] = ("svm", "linear")


def _band(category: pd.Series) -> pd.Series:
    """The box-size band of a `vg_scale` category (``bus@small`` -> ``small``).

    The band is this study's separability ladder: the tilt dies when the rate
    root stays inside the inter-mean interval, which is what a cleanly separated
    haystack produces, and #3255 measured cost roughly tripling from large
    targets to small.  Categories without a band (any other dataset) come back
    ``""`` and simply pool.
    """
    s = category.astype(str)
    return s.str.split("@").str[1].fillna("").where(s.str.contains("@"), "")


def per_k_curves(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Per ``k``: what the slider admits there, and whether it moved to get there.

    ``moved_rate`` is the share of steps whose admitted set differs from the one
    at ``k - 1``, so a run of zeros locates the **flat band** - which is the
    number the issue asks for, and the one an average over the whole knob cannot
    give.  A pooled dead-step rate says how much of the slider is dead; this says
    *which* part, and a band around ``k = 0`` is where users actually sit.
    """
    cols = ["arm", "env", "cell", "t", "inclusion_k", "n_admitted", "admitted_frac", "fold_quantile", "n_votes"]
    g = df[cols].sort_values(["arm", "env", "cell", "t", "inclusion_k"], kind="stable")
    grp = g.groupby(["arm", "env", "cell", "t"], observed=True, sort=False)
    # First k of each step has no predecessor: mask it out rather than let it
    # inherit the previous group's last value.
    moved = (~g["n_admitted"].diff().eq(0)) & (grp.cumcount() > 0)
    g = g.assign(_moved=moved.astype("float32"), _first=(grp.cumcount() == 0))
    g.loc[g["_first"], "_moved"] = np.nan
    out = (
        g.groupby(["arm", "env", "inclusion_k"], observed=True)
        .agg(
            n=("t", "size"),
            admitted_frac=("admitted_frac", "mean"),
            fold_quantile=("fold_quantile", "mean"),
            moved_rate=("_moved", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "head_arm", head)
    return out


def offset_gap(df: pd.DataFrame, head: str, incumbent: str, offset: int) -> pd.DataFrame:
    """H4: what the acquisition offset is still worth, per ``k``.

    ``ACQUISITION_INCLUSION_OFFSET`` cuts the threshold handed to the *selector*
    at ``k + offset`` while reporting stays at ``k``, so the offset is a **gap
    across the slider** and is worth exactly what the slider is worth over that
    span.  Where the knob is flat the two cuts are one cut and the offset buys
    nothing - #2896's collapse - so a wider flat band predicts a wider collapse.

    Returns one row per (env, k) with the mean gap in admitted fraction and the
    share of steps where it is **exactly** zero.
    """
    sub = df[df["arm"] == incumbent][["env", "cell", "t", "inclusion_k", "n_admitted", "admitted_frac"]]
    if sub.empty:
        return pd.DataFrame()
    sel = sub.copy()
    # The selector's own stop, joined onto the reporting stop it serves.
    sel["inclusion_k"] = sel["inclusion_k"] - offset
    merged = sub.merge(
        sel,
        on=["env", "cell", "t", "inclusion_k"],
        suffixes=("", "_sel"),
        how="inner",
    )
    if merged.empty:
        return merged
    merged["gap"] = merged["admitted_frac_sel"] - merged["admitted_frac"]
    merged["collapsed"] = merged["n_admitted_sel"].eq(merged["n_admitted"])
    out = (
        merged.groupby(["env", "inclusion_k"], observed=True)
        .agg(n=("gap", "size"), mean_gap=("gap", "mean"), collapse_rate=("collapsed", "mean"))
        .reset_index()
    )
    out.insert(0, "head_arm", head)
    out.insert(1, "offset", offset)
    return out


def per_cell(per_step: pd.DataFrame, head: str) -> pd.DataFrame:
    """One row per (arm, env, cell): the pairing unit for every contrast here."""
    g = (
        per_step.groupby(["arm", "env", "cell"], observed=True)
        .agg(
            n_steps=("t", "size"),
            dead_step_rate=("dead_step_rate", "mean"),
            knob_yield=("knob_yield", "mean"),
            admitted_span=("admitted_span", "mean"),
            quantile_span=("quantile_span", "mean"),
            inert_rate=("distinct_admitted", lambda s: float((s <= 1).mean())),
        )
        .reset_index()
    )
    g.insert(0, "head_arm", head)
    g["band"] = _band(g["cell"].astype(str).str.split("/").str[3])
    return g


def per_env(cells: pd.DataFrame) -> pd.DataFrame:
    """Average the per-cell frame to one row per (head, arm, env)."""
    return (
        cells.groupby(["head_arm", "arm", "env"], observed=True)
        .agg(
            n_cells=("cell", "size"),
            dead_step_rate=("dead_step_rate", "mean"),
            knob_yield=("knob_yield", "mean"),
            admitted_span=("admitted_span", "mean"),
            quantile_span=("quantile_span", "mean"),
            inert_rate=("inert_rate", "mean"),
        )
        .reset_index()
        .sort_values(["env", "arm", "head_arm"])
    )


def paired_delta(
    cells: pd.DataFrame,
    metric: str,
    left: dict[str, str],
    right: dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """``left - right`` on *metric*, paired by cell, bootstrapped over cells.

    *left* and *right* are column filters (e.g. ``{"head_arm": "svm"}``).  The
    join is on (env, cell) plus whichever of arm/head is not being contrasted, so
    the difference isolates exactly the one axis that differs.
    """
    keys = ["env", "cell"]
    for col in ("head_arm", "arm"):
        if left.get(col) == right.get(col) or (col not in left and col not in right):
            keys.append(col)

    def _side(f: dict[str, str]) -> pd.DataFrame:
        sub = cells
        for col, val in f.items():
            sub = sub[sub[col] == val]
        return sub[[*keys, metric]]

    merged = _side(left).merge(_side(right), on=keys, suffixes=("_l", "_r"))
    if merged.empty:
        return pd.DataFrame()
    merged["d"] = merged[f"{metric}_l"] - merged[f"{metric}_r"]
    rows = []
    for env, grp in merged.groupby("env", observed=True):
        mean, lo, hi = _paired_bootstrap(grp["d"], rng)
        rows.append(
            {
                "env": env,
                "metric": metric,
                "left": json.dumps(left, sort_keys=True),
                "right": json.dumps(right, sort_keys=True),
                "n_cells": int(len(grp)),
                "d": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "left_mean": float(grp[f"{metric}_l"].mean()),
                "right_mean": float(grp[f"{metric}_r"].mean()),
            }
        )
    return pd.DataFrame(rows)


def instrument_checks(env_tbl: pd.DataFrame, incumbent: str) -> dict:
    """The falsifiers, read before any headline number.

    ``mid`` never looks at the cost weights and #2865 measured one admitted set
    across the whole slider in all 65,671 cell-steps, so it must come back inert.
    And ``mid_tilt(k) - rate(k)`` is a **constant** in fold-quantile space (also
    #2865, exactly), so the two must move together however much either moves - a
    liveness gap between them contradicts the algebra rather than measuring
    anything.
    """
    out: dict = {}
    # Built from the incumbent's own name, not matched as a substring: every
    # `mid_tilt` arm *contains* `_mid_`, so a substring test would pool the rule
    # under test into its own null and the check would pass by dilution.
    mid_arm = incumbent.replace("_mid_tilt_", "_mid_")
    mid = env_tbl[env_tbl["arm"] == mid_arm]
    out["mid_arm"] = mid_arm
    out["mid_dead_step_rate_min"] = float(mid["dead_step_rate"].min()) if len(mid) else None
    out["mid_is_inert"] = bool(len(mid)) and bool((mid["dead_step_rate"] > 0.99).all())

    rate_arm = incumbent.replace("_mid_tilt_", "_rate_")
    pair = env_tbl[env_tbl["arm"].isin([incumbent, rate_arm])]
    wide = pair.pivot_table(index=["head_arm", "env"], columns="arm", values="dead_step_rate")
    if incumbent in wide.columns and rate_arm in wide.columns:
        gap = (wide[incumbent] - wide[rate_arm]).abs()
        out["mid_tilt_vs_rate_max_abs_gap"] = float(gap.max())
        # 0.02 is a tolerance on a MEAN over steps, not on the algebra: the two
        # rules differ by a constant quantile offset, so they can realize a
        # different admitted set only where that offset straddles a tie in the
        # score distribution.  A gap materially above this says the frame and the
        # algebra disagree, which is an instrument failure, not a finding.
        out["mid_tilt_tracks_rate"] = bool(gap.max() <= 0.02)
    else:
        out["mid_tilt_vs_rate_max_abs_gap"] = None
        out["mid_tilt_tracks_rate"] = None
    out["ok"] = bool(out["mid_is_inert"]) and out["mid_tilt_tracks_rate"] is not False
    return out


def verdict(
    env_tbl: pd.DataFrame,
    h1: pd.DataFrame,
    h3_dead: pd.DataFrame,
    regret: pd.DataFrame,
    incumbent: str,
) -> dict:
    """The pre-registered rules, applied mechanically.  The tables still get read."""
    out: dict = {
        "incumbent": incumbent,
        "n_boot": N_BOOT,
        "deep_votes_min": DEEP_VOTES_MIN,
        "bars": {"dead_max": DEAD_MAX, "span_min": SPAN_MIN, "harm_tolerance": HARM_TOLERANCE},
    }

    # H1 - the head moved the knob (svm minus linear, on the shipped rule).
    if h1.empty:
        out["H1"] = {"supported": None, "reason": "no paired cells"}
    else:
        hit = h1[h1["ci_lo"] > 0]
        out["H1"] = {
            "supported": bool(len(hit)),
            "envs_softer_under_svm": hit["env"].tolist(),
            "table": json.loads(h1.to_json(orient="records")),
        }

    # H2 - the knob has gone soft in absolute terms, on the SHIPPED head only.
    shipped = env_tbl[(env_tbl["head_arm"] == "svm") & (env_tbl["arm"] == incumbent)]
    soft = shipped[(shipped["dead_step_rate"] >= DEAD_MAX) | (shipped["admitted_span"] <= SPAN_MIN)]
    out["H2"] = {
        "fires": bool(len(soft)),
        "soft_envs": soft["env"].tolist(),
        "table": json.loads(shipped.to_json(orient="records")),
    }

    # H3 - q_tilt ships only on all three conditions.
    q_rows = h3_dead[h3_dead["ci_hi"] < 0] if not h3_dead.empty else h3_dead
    harmed = regret[regret["ci_lo"] > HARM_TOLERANCE] if not regret.empty else regret
    clean_arms = (
        sorted(set(h3_dead["left"].map(lambda s: json.loads(s)["arm"])) - set(harmed["arm"]))
        if not h3_dead.empty
        else []
    )
    helps_where_soft = (
        sorted({json.loads(r["left"])["arm"] for _, r in q_rows.iterrows() if r["env"] in out["H2"]["soft_envs"]})
        if len(q_rows)
        else []
    )
    ships = sorted(set(helps_where_soft) & set(clean_arms)) if out["H2"]["fires"] else []
    out["H3"] = {
        "ships": ships,
        "recommended": ships[0] if ships else None,
        "helps_where_soft": helps_where_soft,
        "regret_clean_arms": clean_arms,
        "n_harmed_env_k": int(len(harmed)),
        "reason": (
            "H2 did not fire: nothing needs fixing, keep the incumbent"
            if not out["H2"]["fires"]
            else (
                "no q_tilt step size both lowered the dead-step rate where the knob is soft "
                f"and stayed within {HARM_TOLERANCE} regret at every k"
                if not ships
                else "lowers the dead-step rate where the knob is soft, with no material regret at any k"
            )
        ),
    }
    return out


def write_report(out_dir: Path, v: dict, tables: dict[str, pd.DataFrame], prov: dict) -> None:
    lines = [
        "# Does the Inclusion knob still have authority under the linear SVM head? (#3196) - draft",
        "",
        "Generated by `analyze_incl_3196.py`.  Pre-registration:",
        "`docs/experiments/inclusion-knob-3196/PLAN.md`.",
        "",
        f"Incumbent (the shipped rule): `{v['incumbent']}`.  Deep regime: `n_votes >= {v['deep_votes_min']}`.",
        "",
        "## Cells read",
        "",
        _md(pd.DataFrame(prov["cells"])),
        "",
        "## Instrument checks (read these first)",
        "",
        "`mid` must be inert and `mid_tilt` must track `rate`; both are algebra,",
        "not findings.  A failure here means nothing below is readable.",
        "",
        "```json",
        json.dumps(v["instrument"], indent=2),
        "```",
        "",
        "## Verdict (mechanical)",
        "",
        "```json",
        json.dumps({k: v[k] for k in ("H1", "H2", "H3")}, indent=2),
        "```",
        "",
        "## The knob, per head and environment",
        "",
        "`dead_step_rate` is the share of adjacent slider stops that admit the",
        "identical set - the issue's *fraction of the knob over which*",
        "`admitted_frac` *is constant*.  `admitted_span` is what dragging the",
        "slider end to end is worth.  A large `quantile_span` beside a small",
        "`admitted_span` is the empty-band case: the rule moves the cut, the",
        "haystack has nothing there to move past.",
        "",
        _md(tables["env"]),
        "",
        "## H1: the head contrast, paired by cell",
        "",
        "Positive `d` = deader under the SVM head.",
        "",
        _md(tables["h1"]),
        "",
        "## H4: what the acquisition offset is still worth",
        "",
        "`collapse_rate` is the share of steps where the selector's cut admits",
        "exactly what reporting's cut admits, i.e. where the offset buys nothing.",
        "",
        _md(tables["h4"]),
        "",
    ]
    (out_dir / "REPORT_incl3196.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#3196 cross-head Inclusion-knob analysis.")
    ap.add_argument("--svm", required=True, help="results dir of the shipped-head arm")
    ap.add_argument("--linear", required=True, help="results dir of the logistic-head arm")
    ap.add_argument("--out", required=True, help="directory for tables, summary and report")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    incumbent = incumbent_arm()
    from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET  # noqa: PLC0415

    dirs = {"svm": Path(args.svm), "linear": Path(args.linear)}
    cells_frames, per_k_frames, gap_frames = [], [], []
    prov_rows, regret = [], pd.DataFrame()
    for head in ARMS:
        results = dirs[head]
        prov: dict = {}
        df = load_cutincl(results / "cells", prov)
        n_expected = expected_cells(results)
        prov_rows.append(
            {
                "head_arm": head,
                "n_read": prov.get("n_files", 0),
                "n_expected": n_expected,
                "n_missing": max(n_expected - prov.get("n_files", 0), 0),
                "n_unreadable": len(prov.get("unreadable", [])),
                "n_empty": len(prov.get("empty", [])),
            }
        )
        if df.empty:
            common.log(f"arm {head}: no cut-inclusion rows; was CALIB_CUT_INCL_KS set?")
            continue
        deep = df[df["n_votes"] >= DEEP_VOTES_MIN]
        if deep.empty:
            common.log(f"arm {head}: WARNING no steps past {DEEP_VOTES_MIN} votes; using the whole trajectory")
            deep = df
        cells_frames.append(per_cell(liveness_per_step(deep), head))
        per_k_frames.append(per_k_curves(deep, head))
        gap_frames.append(offset_gap(deep, head, incumbent, ACQUISITION_INCLUSION_OFFSET))
        if head == "svm":
            # H3's regret half is a question about the SHIPPED head only.
            regret = regret_vs_incumbent(deep, incumbent, out_dir)
        del df, deep

    if not cells_frames:
        common.log("no readable cells in either arm")
        return 1

    cells = pd.concat(cells_frames, ignore_index=True)
    cells.to_csv(out_dir / "incl3196_per_cell.csv", index=False)
    env_tbl = per_env(cells)
    env_tbl.to_csv(out_dir / "incl3196_per_env.csv", index=False)
    per_k = pd.concat(per_k_frames, ignore_index=True)
    per_k.to_csv(out_dir / "incl3196_per_k.csv", index=False)
    gaps = pd.concat([g for g in gap_frames if not g.empty], ignore_index=True) if gap_frames else pd.DataFrame()
    if not gaps.empty:
        gaps.to_csv(out_dir / "incl3196_offset_gap.csv", index=False)

    # The band breakdown: the separability ladder the mechanism runs on.
    bands = (
        cells[cells["arm"] == incumbent]
        .groupby(["head_arm", "env", "band"], observed=True)
        .agg(
            n_cells=("cell", "size"),
            dead_step_rate=("dead_step_rate", "mean"),
            admitted_span=("admitted_span", "mean"),
        )
        .reset_index()
    )
    bands.to_csv(out_dir / "incl3196_by_band.csv", index=False)

    rng = np.random.default_rng(0)
    h1 = paired_delta(
        cells[cells["arm"] == incumbent],
        "dead_step_rate",
        {"head_arm": "svm"},
        {"head_arm": "linear"},
        rng,
    )
    h1_span = paired_delta(
        cells[cells["arm"] == incumbent],
        "admitted_span",
        {"head_arm": "svm"},
        {"head_arm": "linear"},
        rng,
    )
    h1 = pd.concat([h1, h1_span], ignore_index=True)
    if not h1.empty:
        h1.to_csv(out_dir / "incl3196_head_contrast.csv", index=False)

    # H3's liveness half: every q_tilt step size against the incumbent, on the
    # shipped head, paired by cell.
    svm_cells = cells[cells["head_arm"] == "svm"]
    q_arms = sorted(a for a in svm_cells["arm"].unique() if "_q_tilt_" in a)
    h3_dead = (
        pd.concat(
            [paired_delta(svm_cells, "dead_step_rate", {"arm": a}, {"arm": incumbent}, rng) for a in q_arms],
            ignore_index=True,
        )
        if q_arms
        else pd.DataFrame()
    )
    if not h3_dead.empty:
        h3_dead.to_csv(out_dir / "incl3196_qtilt_liveness.csv", index=False)

    v = verdict(env_tbl, h1[h1["metric"] == "dead_step_rate"] if not h1.empty else h1, h3_dead, regret, incumbent)
    v["instrument"] = instrument_checks(env_tbl, incumbent)
    v["cells"] = prov_rows
    (out_dir / "incl3196_summary.json").write_text(json.dumps(v, indent=2))
    write_report(
        out_dir,
        v,
        {"env": env_tbl, "h1": h1, "h4": gaps if not gaps.empty else pd.DataFrame([{"note": "no rows"}])},
        {"cells": prov_rows},
    )
    common.log(f"wrote {out_dir / 'incl3196_summary.json'} and {out_dir / 'REPORT_incl3196.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
