"""Does a different Autopilot **opening** mine better positives?  (issue #3267)

The run sweeps the opening: how many clicks to spend before the first learned
sort, and how deep in the seed sort to spend them.  ``launch_good_mining.sh``
defines the arms; ``vtscore/eval/startup_schedule.py`` defines the grammar.

The question is not only "which arm scores best".  An opening can win for two
very different reasons - it found *more* positives, or it found *harder* ones -
and those imply different follow-ups, so this analyzer reports the mining
behaviour separately from the outcome and never collapses the two.

Four things it refuses to do quietly:

1. **Report an arm without checking its opening actually moved.**  Every arm's
   realized ``startup_cut_percentile`` is compared against the control's
   sampling depth.  An inclusion (``k``) arm is the likely offender: how far a
   given ``k`` moves the pick is a property of the fitted mixture, and on a
   steep sort the whole usable range can land inside a couple of rank percent.
   Such an arm **measured nothing**, which is a different finding from "the
   lever does nothing".
2. **Compare a 16-click opening against a 7-click one and call the difference
   depth.**  Every banded arm is reported against ``flat_mid`` (the
   length-matched control) as well as against ``prod``, and the opening's own
   click count is a reported column.
3. **Read the falsification arm as decoration.**  ``deep_first`` opens *below*
   the good mass and must mine fewer positives.  If it does not, depth is not
   the mechanism and the verdict is withheld.
4. **Hide the cells that produced nothing.**  An opening that never finds both
   classes trains no detector and emits no main row; that is a *result* about
   that arm, not a missing file, so those cells are counted per arm and
   reported.  Paired tests lose them, which is exactly why the count matters.

Writes ``agg/*.csv``, ``startup_summary.json``, ``figures/*.png`` and
``REPORT_startup.md`` under ``$CALIB_EXP/analysis``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import common

common.setup_env()

import curves  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_spikes as sp  # noqa: E402  (reuse the #2847 loader)

try:
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # noqa: BLE001
    _wilcoxon = None

#: Arms, in the order the report reads them.  Keep in step with
#: ``launch_good_mining.sh``; an arm listed here with no cells is reported as
#: absent rather than silently skipped.
ARMS: tuple[str, ...] = (
    "prod",
    "top_long",
    "easy_med_hard",
    "band_wide",
    "incl_k",
    "incl_k_wide",
    "flat_mid",
    "deep_first",
)
CONTROL = "prod"
#: The length-matched control: same opening budget as the banded arms, none of
#: it spent mining.  A banded arm that beats ``prod`` but not this one won on
#: *clicks*, not on *depth*.
LENGTH_CONTROL = "flat_mid"
FALSIFIER = "deep_first"

ARM_SCHEDULE: dict[str, str] = {
    "prod": "(app default: g3@top,b4@mid)",
    "top_long": "g8@top,b4@mid",
    "easy_med_hard": "n5@q0.02,n5@q0.10,n6@mid",
    "band_wide": "n5@q0.05,n5@q0.25,n6@mid",
    "incl_k": "n5@k-6,n5@k-2,n6@k0",
    "incl_k_wide": "n5@k-10,n5@k-4,n6@k0",
    "flat_mid": "n16@mid",
    "deep_first": "n10@q0.35,n6@mid",
}

#: An arm's opening counts as having moved if its median sampling depth differs
#: from the control's by at least this many rank percent.  Below it the arm is
#: reported as having measured nothing.
DEPTH_EPS = float(os.environ.get("GM_DEPTH_EPS", "0.01"))
#: Ship rule, mirroring the acquisition-inclusion study's: positives must rise
#: and cost must not regress by more than this at the 95% upper bound.
COST_REGRESSION_TOLERANCE = float(os.environ.get("GM_COST_TOL", "0.01"))
ALPHA = 0.05
#: Click marks the mining curve is read at.  The axis a user spends is clicks,
#: which is not the axis the method converges on - so both are reported.
CLICK_MARKS = tuple(int(c) for c in os.environ.get("GM_CLICK_MARKS", "10,20,50,100,200").split(",") if c.strip())

OUT = Path(os.environ.get("GM_OUT", str(common.EXP / "analysis")))
KEYS = ("dataset", "embedder", "category", "seed")

#: ``text_baseline.py``'s CSV — the ZERO-CLICK anchor every quality curve is
#: drawn against.  Typing the query and reading the ranked haystack is free, so
#: it is what a clicked detector has to beat, and without it the curves start at
#: the first trainable click with nothing to compare the far right against.
#: Optional: absent, the curves simply have no ``t=0`` point.
TEXT_BASELINE = os.environ.get("GM_TEXT_BASELINE", "")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_picks(arm_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Concatenate one arm's ``__picks.csv`` cells, with provenance.

    Separate from :func:`analyze_spikes.load_arm` because the pick log is the
    frame that records the *opening*, and the opening emits no main row.  An
    unreadable or zero-byte cell is counted, never dropped in silence.
    """
    cells = arm_dir / "cells"
    files = sorted(cells.glob("task_*__picks.csv")) if cells.is_dir() else []
    frames, bad, empty, headless = [], [], [], []
    for f in files:
        if f.stat().st_size == 0:
            empty.append(f.name)
            continue
        try:
            fr = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            bad.append(f.name)
            continue
        if fr.empty:
            headless.append(f.name)
            continue
        frames.append(fr)
    prov = {"n_files": len(files), "n_read": len(frames), "unreadable": bad, "zero_byte": empty, "empty": headless}
    if not frames:
        return pd.DataFrame(), prov
    df = pd.concat(frames, ignore_index=True)
    prov["n_rows"] = int(len(df))
    return df, prov


def load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(main, picks, provenance)`` over every arm, each tagged with ``arm``."""
    mains, picks, prov = [], [], {}
    for arm in ARMS:
        m, pm = sp.load_arm(root / arm)
        p, pp = load_picks(root / arm)
        prov[arm] = {"main": pm, "picks": pp}
        for df, sink in ((m, mains), (p, picks)):
            if not df.empty:
                df = df.copy()
                df["arm"] = arm
                sink.append(df)
    main = pd.concat(mains, ignore_index=True) if mains else pd.DataFrame()
    pick = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    return main, pick, prov


def _keys(df: pd.DataFrame) -> list[str]:
    return [k for k in KEYS if k in df.columns]


#: Restrict every table to cells that exist in **all** arms.
#:
#: The paired contrasts already drop unmatched cells, but the per-arm columns -
#: open yield, starvation rate, sampling depth - do not, and those are read
#: side by side as though they described the same grid.  A run stopped on a
#: wall clock, or one arm losing cells to a node failure, then shifts an arm's
#: unpaired number for a reason that has nothing to do with its opening.
#:
#: On by default: a balanced grid is what every table here claims to describe.
#: The count dropped is reported, because silently analysing a subset is how a
#: disk incident becomes a wrong verdict.
BALANCED = os.environ.get("GM_BALANCED", "1") not in ("", "0")


def balance(main: pd.DataFrame, picks: pd.DataFrame, arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Keep only the cells every arm has, so unpaired columns compare like with like."""
    if picks.empty or not BALANCED:
        return main, picks, {"balanced": False}
    keys = _keys(picks)
    if not keys:
        return main, picks, {"balanced": False}
    seen = picks.groupby(keys)["arm"].nunique()
    complete = seen[seen == len(arms)].index
    before = int(picks.groupby(keys).ngroups)
    if len(complete) == 0:
        return main, picks, {"balanced": False, "reason": "no cell is present in every arm"}
    idx = pd.MultiIndex.from_tuples(list(complete), names=keys) if len(keys) > 1 else pd.Index(complete, name=keys[0])
    pk = picks.set_index(keys)
    pk = pk.loc[pk.index.isin(idx)].reset_index()
    mn = main
    if not main.empty and all(k in main.columns for k in keys):
        mi = main.set_index(keys)
        mn = mi.loc[mi.index.isin(idx)].reset_index()
    return (
        mn,
        pk,
        {
            "balanced": True,
            "cells_complete": int(len(complete)),
            "cells_seen": before,
            "cells_dropped": int(before - len(complete)),
            "arms_required": len(arms),
        },
    )


# ---------------------------------------------------------------------------
# Mining: what the opening actually did
# ---------------------------------------------------------------------------


def opening_stats(picks: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(arm, cell)`` describing that trajectory's **opening**.

    The opening is every click spent before the first learned sort, i.e. every
    click carrying a schedule round (``startup_round >= 0``) - or, on the
    control, every click in the ``good`` / ``bad`` phases, which is the same
    thing under a different name.
    """
    if picks.empty:
        return pd.DataFrame()
    keys = ["arm", *_keys(picks)]
    out = []
    for key, g in picks.groupby(keys, dropna=False):
        g = g.sort_values("t")
        in_open = (
            g["startup_round"] >= 0 if (g["startup_round"] >= 0).any() else g["phase"].astype(str).isin(("good", "bad"))
        )
        op = g[in_open]
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=False))
        rec.update(
            n_clicks=int(len(g)),
            open_clicks=int(len(op)),
            open_positives=int(op["picked_label"].sum()) if len(op) else 0,
            open_yield=float(op["picked_label"].mean()) if len(op) else np.nan,
            # Where the opening sampled, and how deep it actually had to reach
            # for the positives it found.  The first is the arm's *intent*, the
            # second its *outcome*; an arm can move the first without moving the
            # second, and that is the case worth catching.
            open_cut_depth=float(op["startup_cut_percentile"].median()) if len(op) else np.nan,
            open_pick_depth=float(op["picked_seed_percentile"].median()) if len(op) else np.nan,
            open_pos_depth=(
                float(op.loc[op["picked_label"] == 1, "picked_seed_percentile"].median())
                if len(op) and (op["picked_label"] == 1).any()
                else np.nan
            ),
            # An opening that never produced both classes: the harness kept
            # voting past the schedule to get a trainable pair.  A non-zero
            # count is a finding about that arm's opening, not noise.
            # Clicks the arm spent as WRITTEN, and clicks it was held past the
            # schedule for want of a trainable pair.  Kept apart because only
            # the first is the arm's design: `flat_mid` is the length-matched
            # control and stops being one the moment it overruns.
            open_scheduled_clicks=int(_scheduled(op).sum()) if len(op) else 0,
            open_overrun=int((~_scheduled(op)).sum()) if len(op) else 0,
            open_starved=bool(len(op) and int(op["picked_label"].sum()) == 0),
            # The labelset at the horizon, which is the thing a detector is
            # actually trained on.  Every click labels an item regardless of
            # phase, so a held arm is not idling - it is piling up negatives.
            # Reporting both makes the failure legible as what it is: a
            # one-class labelset, not a shortage of votes.
            n_good_final=int(g["n_good"].iloc[-1]) if "n_good" in g.columns and len(g) else np.nan,
            n_bad_final=int(g["n_bad"].iloc[-1]) if "n_bad" in g.columns and len(g) else np.nan,
            trained_at=int(g.loc[g["phase"].astype(str).isin(("hard", "new", "done")), "t"].min())
            if g["phase"].astype(str).isin(("hard", "new", "done")).any()
            else -1,
        )
        for mark in CLICK_MARKS:
            head = g[g["t"] <= mark]
            rec[f"positives_{mark}"] = int(head["picked_label"].sum())
        out.append(rec)
    return pd.DataFrame(out)


def _scheduled(op: pd.DataFrame) -> pd.Series:
    """Per click: was it spent as the schedule was WRITTEN (not held past it)?

    Read from the harness's own ``startup_held`` column.  This was previously
    reconstructed from the round indices, and the reconstruction was a no-op -
    it subtracted ``min(count_of_last_round, len(rounds))``, which is just
    ``count_of_last_round``, so the overrun it computed was identically zero
    and an arm that spent its whole horizon under the seed sort, still waiting
    for a first positive, reported an opening exactly as long as it had asked
    for.  The state was
    never derivable from the round indices; it is now recorded.
    """
    if "startup_held" not in op.columns:
        # A pick log from before the column existed.  Say so rather than
        # inventing zeros: silently reporting "no overrun" is the failure this
        # replaced.
        return pd.Series(np.nan, index=op.index, dtype="float64").notna()
    return ~op["startup_held"].fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Outcome: what the detector did afterwards
# ---------------------------------------------------------------------------


def trajectory_stats(main: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(arm, cell)`` carrying the endpoints the ship rule reads."""
    if main.empty:
        return pd.DataFrame()
    keys = ["arm", *_keys(main)]
    out = []
    for key, g in main.groupby(keys, dropna=False):
        g = g.sort_values("t")
        cost = g["cost"].to_numpy(dtype=float)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=False))
        rec.update(
            n_steps=int(len(g)),
            final_cost=float(cost[-1]),
            mean_cost_warm=float(np.nanmean(cost[g["t"].to_numpy(dtype=float) >= sp.WARM_T]))
            if (g["t"] >= sp.WARM_T).any()
            else np.nan,
            final_ap=float(g["average_precision"].iloc[-1]) if "average_precision" in g.columns else np.nan,
            final_n_good=int(g["n_good"].iloc[-1]),
        )
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def paired(traj: pd.DataFrame, metric: str, arm: str, control: str) -> dict:
    """Wilcoxon + bootstrap CI on the paired ``(cell)`` delta ``arm - control``.

    Paired on the identical (dataset, embedder, category, seed) so the arms are
    compared on the same data and the same splits; a cell missing from either
    side drops out of the pair and is counted in ``n_pairs``.
    """
    keys = _keys(traj)
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


def lever_moved(opening: pd.DataFrame, arm: str, control: str = CONTROL) -> dict:
    """Did *arm*'s opening sample somewhere else than the control's?

    The check that stops "no effect" being reported for an arm whose cut never
    left the control's rank position.  Measured on where the opening's picks
    *landed*, not on the cut's nominal value: two very different inclusions can
    name the same rank on a steep sort, and rank is what the pick reads.
    """
    if opening.empty:
        return {"moved": False, "reason": "no pick rows"}
    a = opening[opening["arm"] == arm]["open_pick_depth"].median()
    c = opening[opening["arm"] == control]["open_pick_depth"].median()
    if not np.isfinite(a) or not np.isfinite(c):
        return {"moved": False, "reason": "no opening depth recorded"}
    return {
        "moved": bool(abs(a - c) >= DEPTH_EPS),
        "arm_depth": float(a),
        "control_depth": float(c),
        "delta": float(a - c),
        "eps": DEPTH_EPS,
    }


def verdict(summary: dict) -> str:
    """The pre-registered read, or a withheld one and why.

    Withheld beats wrong: the falsifier failing to falsify means depth is not
    the mechanism, and no arm-vs-control number in the run is interpretable
    until that is understood.
    """
    fals_arm = summary["arms"].get(FALSIFIER, {})
    fals_pos = fals_arm.get("positives_100", {})
    if not fals_pos.get("n_pairs"):
        return f"WITHHELD: the falsification arm ({FALSIFIER}) produced no pairs."
    if fals_pos.get("median_delta", 0.0) >= 0:
        return (
            f"WITHHELD: {FALSIFIER} opens below the good mass and should mine FEWER positives, "
            f"but its paired delta is {fals_pos['median_delta']:+.3g}. Depth is not behaving as "
            "the mechanism assumes; nothing else here is interpretable until that is resolved."
        )
    winners = []
    for arm, rec in summary["arms"].items():
        if arm in (CONTROL, FALSIFIER):
            continue
        if not rec.get("lever", {}).get("moved"):
            continue
        pos = rec.get("positives_100", {})
        cost = rec.get("final_cost", {})
        if not pos.get("n_pairs") or not cost.get("n_pairs"):
            continue
        beats_len = rec.get("vs_length_control", {}).get("positives_100", {}).get("median_delta", 0.0)
        if pos.get("median_delta", 0.0) > 0 and cost.get("ci95_hi", 1.0) <= COST_REGRESSION_TOLERANCE:
            winners.append((arm, pos["median_delta"], cost["ci95_hi"], beats_len))
    if not winners:
        return "NO ARM SHIPS: no opening both mined more positives and held cost within tolerance."
    winners.sort(key=lambda w: -w[1])
    arm, dpos, chi, dlen = winners[0]
    tail = (
        f" It also beats the length-matched control ({LENGTH_CONTROL}) by {dlen:+.3g} positives, "
        "so the gain is depth rather than budget."
        if dlen > 0
        else (
            f" But it does NOT beat the length-matched control ({LENGTH_CONTROL}) ({dlen:+.3g}), "
            "so the gain may be the extra opening clicks rather than where they were spent."
        )
    )
    return (
        f"CANDIDATE: {arm} ({ARM_SCHEDULE.get(arm, '?')}) mines {dpos:+.3g} more positives per 100 "
        f"clicks with cost regression bounded at {chi:+.3g}.{tail}"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def prevalence_table(root: Path) -> dict[tuple[str, str], float]:
    """``(dataset, category) -> prevalence``, from the prepare stage's own counts.

    Prevalence is the axis this study's mechanism runs on: an opening that mines
    better should matter most where positives are scarce, and an average taken
    across a 50x prevalence range is precisely the number that hides that.
    """
    info_p = root / "prepare_info.json"
    if not info_p.exists():
        return {}
    info = json.loads(info_p.read_text())
    out: dict[tuple[str, str], float] = {}
    for ds, embs in info.get("datasets", {}).items():
        for _emb, d in embs.items():
            n = int(d.get("n_medias") or 0)
            counts = d.get("category_counts") or {}
            for cat in d.get("selected_categories") or []:
                if n:
                    out[(ds, cat)] = float(counts.get(cat, 0)) / n
    return out


def make_figures(
    picks: pd.DataFrame,
    opening: pd.DataFrame,
    outdir: Path,
    prevalence: dict[tuple[str, str], float] | None = None,
    traj: pd.DataFrame | None = None,
    main: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
) -> list[str]:
    """Cost curves, mining curve (mean and per-run), opening depth, prevalence, starvation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 0. THE HEADLINE, and the standard every simulated-user report owes: how
    #    good the user's detector is as they keep clicking - averaged per
    #    dataset, and again with every single run drawn.  Delegated to
    #    ``curves.py`` rather than written here, so the figure a reader learns
    #    to read in one report is literally the same figure in the next one.
    #
    #    ``opening`` is the denominator on purpose: it has a row for every cell
    #    the run ATTEMPTED, including the ones that never trained a detector and
    #    so appear nowhere in ``main``.  Without it the coverage strip would be
    #    computed from the cells that produced rows and read 100% for an arm
    #    that starved on a third of its grid.
    if main is not None and not main.empty:
        for metric, lower_better in (("cost", True), ("average_precision", False)):
            if metric not in main.columns:
                continue
            written.extend(
                curves.quality_vs_clicks(
                    main,
                    outdir,
                    arms=ARMS,
                    metric=metric,
                    denominator=opening if not opening.empty else None,
                    baseline=baseline,
                    prevalence=prevalence,
                    lower_is_better=lower_better,
                )
            )

    if picks.empty:
        return written

    # 1. Positives found vs clicks spent - the figure that answers "what do I
    #    get after 20 clicks?".  Mean over cells, with a spread band.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for arm in ARMS:
        g = picks[picks["arm"] == arm]
        if g.empty:
            continue
        cum = g.sort_values("t").groupby([*_keys(g), "t"])["picked_label"].sum().groupby(level=_keys(g)).cumsum()
        wide = cum.reset_index().pivot_table(index="t", columns=_keys(g), values="picked_label")
        ax.plot(wide.index, wide.mean(axis=1), label=arm, lw=1.6)
        ax.fill_between(wide.index, wide.quantile(0.25, axis=1), wide.quantile(0.75, axis=1), alpha=0.10)
    ax.set_xlabel("clicks spent")
    ax.set_ylabel("positives found (cumulative)")
    ax.set_title("Good mining: positives per click, by opening")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = outdir / "mining_curve.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # 2. The same, one line per run.  A mean hides that some runs never leave
    #    the floor, and that spread is usually the real finding.
    arms_present = [a for a in ARMS if not picks[picks["arm"] == a].empty]
    fig, axes = plt.subplots(1, len(arms_present), figsize=(2.5 * len(arms_present), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms_present, strict=False):
        g = picks[picks["arm"] == arm]
        for _, run in g.groupby(_keys(g)):
            run = run.sort_values("t")
            ax.plot(run["t"], run["picked_label"].cumsum(), lw=0.6, alpha=0.45, color="#2b6cb0")
        ax.set_title(arm, fontsize=8)
        ax.set_xlabel("clicks")
    axes[0].set_ylabel("positives found")
    fig.suptitle("Per-run mining curves (a mean hides the runs that never start)", fontsize=9)
    fig.tight_layout()
    p = outdir / "mining_per_run.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    # 3. Where each opening actually sampled, against where it was aimed.  An
    #    arm whose two bars coincide with the control's measured nothing.
    if not opening.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        arms_present = [a for a in ARMS if (opening["arm"] == a).any()]
        x = np.arange(len(arms_present))
        aimed = [opening.loc[opening["arm"] == a, "open_cut_depth"].median() for a in arms_present]
        landed = [opening.loc[opening["arm"] == a, "open_pick_depth"].median() for a in arms_present]
        ax.bar(x - 0.2, aimed, 0.4, label="cut depth (aimed)")
        ax.bar(x + 0.2, landed, 0.4, label="pick depth (landed)")
        ax.set_xticks(x)
        ax.set_xticklabels(arms_present, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("depth in the seed sort (0 = top)")
        ax.set_title("Did the opening move? aimed vs landed")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "opening_depth.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    # 4. THE AXIS THE MECHANISM RUNS ON.  "Mine more Goods" should matter most
    #    where Goods are scarce, and these environments span ~50x in prevalence,
    #    so one pooled number per arm is an average across the very crossover
    #    the study is looking for.
    #
    #    Two panels, because they answer different halves and one of them lies
    #    on its own.  LEFT bands prevalence into three and shows the mean paired
    #    contrast per band with its standard error - the readable answer.  RIGHT
    #    keeps every category as a point so the band means cannot hide a
    #    category that disagrees with its own band.
    #
    #    Points are NOT joined.  A line between two categories implies a series
    #    and there is none: the x axis is a property each category happens to
    #    have, not an axis anything moves along.  Joining them made a noisy
    #    scatter read as a trend.
    if prevalence and not picks.empty:
        keys = _keys(picks)
        totals = picks.groupby(["arm", *keys])["picked_label"].sum().rename("positives").reset_index()
        ctrl = totals[totals["arm"] == CONTROL].drop(columns="arm").rename(columns={"positives": "ctrl"})
        merged = totals.merge(ctrl, on=keys, how="inner")
        merged["delta"] = merged["positives"] - merged["ctrl"]
        merged["prevalence"] = [prevalence.get((d, c), np.nan) for d, c in zip(merged["dataset"], merged["category"])]
        merged = merged.dropna(subset=["prevalence"])
        arms_here = [a for a in ARMS if a != CONTROL and (merged["arm"] == a).any()]
        if not merged.empty and arms_here:
            cats = merged[["dataset", "category", "prevalence"]].drop_duplicates().sort_values("prevalence")
            # Terciles of the CATEGORIES, not of the cells: every category
            # carries the same number of seeds, so an equal-count split of
            # categories is an equal-weight split of the evidence.
            n = len(cats)
            edges = [
                cats["prevalence"].iloc[0],
                cats["prevalence"].iloc[max(0, n // 3 - 1)],
                cats["prevalence"].iloc[max(0, 2 * n // 3 - 1)],
                cats["prevalence"].iloc[-1],
            ]
            names = [
                f"scarce\n(<{edges[1] * 100:.1f}%)",
                f"mid\n({edges[1] * 100:.1f}-{edges[2] * 100:.1f}%)",
                f"common\n(>{edges[2] * 100:.1f}%)",
            ]

            def _band(v: float) -> int:
                return 0 if v <= edges[1] else (1 if v <= edges[2] else 2)

            merged["band"] = [_band(v) for v in merged["prevalence"]]

            fig, (axb, axs) = plt.subplots(1, 2, figsize=(12.5, 4.8), width_ratios=[1.0, 1.25])
            # One colour per ARM, fixed across both panels.  Letting matplotlib
            # cycle per call gave the two panels different colours for the same
            # arm, which is worse than no colour: a reader matches the legend on
            # the left to a cloud on the right and reads the wrong arm.
            palette = {a: f"C{i}" for i, a in enumerate(arms_here)}
            width = 0.8 / len(arms_here)
            for i, arm in enumerate(arms_here):
                g = merged[merged["arm"] == arm]
                means = [g.loc[g["band"] == b, "delta"].mean() for b in range(3)]
                std_errs = [g.loc[g["band"] == b, "delta"].sem() for b in range(3)]
                axb.bar(
                    np.arange(3) + i * width - 0.4 + width / 2,
                    means,
                    width,
                    yerr=np.nan_to_num(std_errs),
                    capsize=2,
                    label=arm,
                    color=palette[arm],
                )
            axb.axhline(0.0, color="#444", lw=1.0)
            axb.set_xticks(np.arange(3))
            axb.set_xticklabels(names, fontsize=8)
            axb.set_ylabel(f"positives at the horizon, minus {CONTROL}")
            axb.set_title("Banded by prevalence (mean ± SE)")
            axb.legend(fontsize=7, ncol=2)

            markers = {"coco_val": "o", "visual_genome_m": "^"}
            for arm in arms_here:
                g = merged[merged["arm"] == arm]
                per_cat = g.groupby(["dataset", "category", "prevalence"])["delta"].mean().reset_index()
                first = True
                for ds, sub in per_cat.groupby("dataset"):
                    axs.scatter(
                        sub["prevalence"],
                        sub["delta"],
                        s=22,
                        alpha=0.75,
                        color=palette[arm],
                        marker=markers.get(ds, "o"),
                        label=arm if first else None,
                    )
                    first = False
            axs.axhline(0.0, color="#444", lw=1.0, ls="--")
            axs.set_xscale("log")
            axs.set_xlabel("category prevalence in the pool (log) — o coco_val, ^ visual_genome_m")
            axs.set_title("Every category as its own point (not a series - points are not joined)")
            axs.legend(fontsize=7, ncol=2)
            fig.suptitle("Does a better opening matter more where Goods are scarce?", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            p = outdir / "mining_by_prevalence.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            written.append(p.name)

    # 5. THE BINDING CONSTRAINT, and this study's headline failure mode: an
    #    opening that finds no positive at all.  The harness then holds the
    #    trajectory on the schedule's last round rather than hand a one-class
    #    labelset to a learned sort.
    #
    #    Those clicks are NOT wasted votes: every click labels an item and goes
    #    into the training data whatever phase the autopilot thinks it is in -
    #    the phase decides only which item is shown next, never whether the
    #    answer counts.  A held arm is accumulating negatives at full rate.
    #    What it does not have is a POSITIVE, and one class cannot be fitted,
    #    so no detector exists and no metric row is emitted.  The cost is that
    #    the clicks buy labels the model cannot yet use, and are spent under the
    #    seed sort rather than under a learned one.
    #    Two bars because they are different facts: how often an arm starves,
    #    and how much of the horizon it loses when it does.
    if not opening.empty and "open_starved" in opening.columns:
        arms_present = [a for a in ARMS if (opening["arm"] == a).any()]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
        x = np.arange(len(arms_present))
        starved = [100.0 * opening.loc[opening["arm"] == a, "open_starved"].mean() for a in arms_present]
        ax1.bar(x, starved, 0.6, color="#b91c1c")
        ax1.set_xticks(x)
        ax1.set_xticklabels(arms_present, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("% of cells whose opening found NO positive")
        ax1.set_title("How often an opening starves")
        if "open_overrun" in opening.columns:
            data = [opening.loc[opening["arm"] == a, "open_overrun"].dropna().to_numpy() for a in arms_present]
            data = [d if len(d) else np.array([0.0]) for d in data]
            ax2.boxplot(data, tick_labels=arms_present, showfliers=False)
            ax2.set_xticklabels(arms_present, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("clicks held past the written schedule")
            ax2.set_title("What it costs when it does")
        fig.tight_layout()
        p = outdir / "starvation.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p.name)

    # 6. THE ISSUE'S PREMISE, tested directly.  #3267 opens by asserting it:
    #    "getting enough Goods is important to VTSearch runs doing well.
    #    (Certainly being Good-starved seems related to failing.)"  Everything
    #    else here assumes that and asks which opening mines best; this asks
    #    whether the assumption holds, on this run's own data.
    #
    #    Pooled over ALL arms on purpose - the relationship is a claim about
    #    trajectories, not about openings - with the arms coloured so a reader
    #    can see whether it is one arm's cloud doing the work.  Binned medians
    #    over the top, because a scatter of thousands of cells shows a shape
    #    only by accident.
    if traj is not None and not traj.empty and not opening.empty:
        keys = [k for k in ["arm", *_keys(opening)] if k in traj.columns and k in opening.columns]
        merged = opening.merge(traj, on=keys, how="inner", suffixes=("", "_traj"))
        if "final_cost" in merged.columns and "open_positives" in merged.columns:
            m = merged.dropna(subset=["final_cost", "open_positives"])
            if len(m) > 10:
                fig, ax = plt.subplots(figsize=(7.5, 4.6))
                for arm in ARMS:
                    g = m[m["arm"] == arm]
                    if g.empty:
                        continue
                    ax.scatter(g["open_positives"], g["final_cost"], s=7, alpha=0.30, label=arm)
                bins = np.arange(-0.5, float(m["open_positives"].max()) + 1.5, 1.0)
                m = m.assign(_b=pd.cut(m["open_positives"], bins))
                med = m.groupby("_b", observed=True)["final_cost"].agg(["median", "size"])
                centres = [iv.mid for iv in med.index]
                ax.plot(centres, med["median"], color="#111", lw=2.0, marker="o", ms=4, label="median (all arms)")
                ax.set_xlabel("positives found in the opening")
                ax.set_ylabel("final cost (lower is better)")
                ax.set_title("Is a Good-starved opening really a failing run?")
                ax.legend(fontsize=7, ncol=2)
                fig.tight_layout()
                p = outdir / "premise_starvation_vs_cost.png"
                fig.savefig(p, dpi=130)
                plt.close(fig)
                written.append(p.name)
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(d: dict, key: str, digits: int = 2) -> str:
    v = d.get(key)
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}g}"


#: Captions for the figures whose *reading* is not obvious from the axes - what
#: the line means, and what it does NOT license.  A figure without one still
#: renders; a figure whose dashed segments go unexplained is worse than absent.
FIGURE_CAPTIONS: dict[str, str] = {
    "cost_vs_clicks.png": (
        "**The headline: how good the user's detector is as they keep clicking.** One panel per "
        "dataset, one line per arm, mean over every seed and category on that dataset, with an "
        "inter-quartile band. **Click 0 is the free text sort** - what the typed query got for "
        "nothing - carried across the panel as a dotted line, so the far left is what typing was "
        "worth, the far right is what clicking was worth, and where a curve crosses the dotted "
        "line is the click at which the detector started earning its keep. The lower strip is the "
        "denominator: what fraction of that arm's cells are measured at that click (all of them at "
        "click 0, which has a text sort; from click 1 only the ones with a detector). The mean is "
        "**dashed** wherever that is below "
        f"{curves.SOLID_COVERAGE:.0%} - there it is a level over the subset of cells that trained, "
        "not over the grid, and an arm that starves looks better than it is on exactly those "
        "clicks. Read a level only off a solid segment. Averaged across a wide prevalence range, "
        "so it says which arm, not how well any one category does."
    ),
    "average_precision_vs_clicks.png": (
        "The same figure for **average precision** - the ranking, with the threshold taken out of "
        "it. Higher is better here. An arm that improves cost but not AP moved the *threshold*; "
        "one that improves both moved the *ranking*. Same dashed-means-subset rule."
    ),
}


def _runs_caption(metric: str, dataset: str) -> str:
    better = "lower is better" if metric == "cost" else "higher is better"
    return (
        f"**The individuals, on `{dataset}`: every seed of every arm as its own line** ({metric}, "
        f"{better}), coloured by the category's prevalence in the pool. A mean cannot show that two "
        "arms with the same level are 'every run is mediocre' and 'half the runs are excellent and "
        "half never start', and on this axis that is usually the finding. A run that never trained "
        "a detector draws **no line at all** - the panel title counts those, because an absent "
        "curve and a missing seed look identical otherwise. The black line is the median over the "
        "runs present at that click, dashed where that is a median over a subset. Each line starts "
        "at **that cell's own text-sort quality at click 0**, so the leftmost point is what the "
        "typed query was worth on that exact cell; a never-trained run is the lone `x` at click 0 "
        "with nothing to its right."
    )


def write_report(summary: dict, figures: list[str], outdir: Path) -> Path:
    lines = [
        "# Good Mining: does a different Autopilot opening find better positives?",
        "",
        f"Issue #3267.  Arms: `{'`, `'.join(ARMS)}`.  Control: `{CONTROL}`; ",
        f"length-matched control: `{LENGTH_CONTROL}`; falsifier: `{FALSIFIER}`.",
        "",
        f"**Verdict.** {summary['verdict']}",
        "",
        "## Coverage",
        "",
        "| arm | schedule | cells (main) | cells (picks) | no detector trained | unreadable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        pv = summary["provenance"].get(arm, {})
        m, p = pv.get("main", {}), pv.get("picks", {})
        lines.append(
            f"| `{arm}` | `{ARM_SCHEDULE.get(arm, '?')}` | {m.get('n_read', 0)} | {p.get('n_read', 0)} | "
            f"{len(m.get('no_positive_found', []))} | {len(m.get('unreadable', [])) + len(p.get('unreadable', []))} |"
        )
    bal = summary.get("balance") or {}
    if bal.get("balanced"):
        lines += [
            "",
            f"Analysed on the **balanced** grid: {bal['cells_complete']} cells present in all "
            f"{bal['arms_required']} arms, of {bal['cells_seen']} seen "
            f"({bal['cells_dropped']} dropped). The paired contrasts would drop the unmatched cells",
            "anyway; the per-arm columns would not, and they are read side by side as though they",
            "described the same grid. Set `GM_BALANCED=0` to analyse every cell that exists.",
        ]
    elif bal.get("reason"):
        lines += ["", f"**Not balanced**: {bal['reason']}. Per-arm columns below may not describe the same cells."]
    lines += [
        "",
        "A cell under **no detector trained** is a result, not a missing file: that opening never",
        "found both vote classes inside the horizon, so it emitted no main row.  Those cells drop",
        "out of every paired test below, which is why the count belongs here.",
        "",
        "## Did each opening move?",
        "",
        "| arm | aimed depth | landed depth | vs control | moved? |",
        "|---|---:|---:|---:|:--:|",
    ]
    for arm in ARMS:
        rec = summary["arms"].get(arm, {})
        lv = rec.get("lever", {})
        aim = rec.get("open_cut_depth")
        lines.append(
            f"| `{arm}` | {'-' if aim is None else f'{aim:.2g}'} | {_fmt(lv, 'arm_depth')} | "
            f"{_fmt(lv, 'delta')} | {'yes' if lv.get('moved') else '**no**'} |"
        )
    lines += [
        "",
        "Depth is a rank position in the seed sort (0 = the top).  An arm marked **no** sampled",
        f"within {DEPTH_EPS:.2g} of the control and therefore measured nothing - do not read its",
        "outcome columns as evidence that the opening does not matter.",
        "",
        "## Mining and outcome, paired against the control",
        "",
        "| arm | open clicks (written) | held past it | starved | labelset @200 (good/bad) | "
        "open yield | positives@100 Δ | [95% CI] | median | final cost Δ | [95% CI] | median | AP Δ |",
        "|---|---:|---:|---:|:--:|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for arm in ARMS:
        if arm == CONTROL:
            continue
        rec = summary["arms"].get(arm, {})
        pos, cost, ap = rec.get("positives_100", {}), rec.get("final_cost", {}), rec.get("final_ap", {})
        lines.append(
            f"| `{arm}` | {_fmt(rec, 'open_scheduled_clicks')} | {_fmt(rec, 'open_overrun_median')} | "
            f"{_fmt(rec, 'open_starved_pct')}% | "
            f"{_fmt(rec, 'n_good_final')}/{_fmt(rec, 'n_bad_final')} | {_fmt(rec, 'open_yield')} | "
            f"{_fmt(pos, 'mean_delta')} | [{_fmt(pos, 'ci95_lo')}, {_fmt(pos, 'ci95_hi')}] | "
            f"{_fmt(pos, 'median_delta')} | "
            f"{_fmt(cost, 'mean_delta')} | [{_fmt(cost, 'ci95_lo')}, {_fmt(cost, 'ci95_hi')}] | "
            f"{_fmt(cost, 'median_delta')} | "
            f"{_fmt(ap, 'mean_delta')} |"
        )
    lines += [
        "",
        "**open clicks (written)** is the opening the arm's schedule asked for; **held past it** is",
        "the clicks it was then held on the last round for, because one vote class was still empty",
        "and handing a learned sort a one-class labelset would leave the selector picking at random.",
        "**starved** is the share of cells whose opening found no positive at all - the extreme of",
        "the regime this study is about, and the reason the two click columns cannot be added.",
        "",
        "A held click is **not** an idle one: every click labels an item and enters the training",
        "data whatever phase the autopilot is in - the phase chooses which item is shown next, never",
        "whether the answer counts. A held arm is piling up negatives at full rate. What it lacks is",
        "a *positive*, and one class cannot be fitted, so no detector exists and no metric row is",
        "emitted. `labelset @200` below reports what the model was actually handed.",
        "",
        "The interval is a bootstrap of the **mean** paired delta, so the mean is what sits beside",
        "it; the median is given too because these distributions are skewed - a third of some arms'",
        "cells find no positive at all, and a mean and a median say different true things about",
        "that. Reading a median against a mean's interval, as an earlier draft of this table did,",
        "produces the nonsense of a point estimate outside its own interval.",
        "",
        "Every delta is paired on the identical (dataset, embedder, category, seed).  A difference",
        "smaller than twice its standard error is not resolvable here, and saying so is a finding.",
        "",
        "## Against the length-matched control",
        "",
        f"Every banded arm spends more opening clicks than `{CONTROL}`, so a win against it could be",
        f"budget rather than depth.  `{LENGTH_CONTROL}` spends the same budget with no mining round.",
        "",
        "| arm | positives@100 Δ vs length control | [95% CI] | final cost Δ | [95% CI] |",
        "|---|---:|---|---:|---|",
    ]
    for arm in ARMS:
        if arm in (CONTROL, LENGTH_CONTROL):
            continue
        vs = summary["arms"].get(arm, {}).get("vs_length_control", {})
        pos, cost = vs.get("positives_100", {}), vs.get("final_cost", {})
        lines.append(
            f"| `{arm}` | {_fmt(pos, 'mean_delta')} | [{_fmt(pos, 'ci95_lo')}, {_fmt(pos, 'ci95_hi')}] | "
            f"{_fmt(cost, 'mean_delta')} | [{_fmt(cost, 'ci95_lo')}, {_fmt(cost, 'ci95_hi')}] |"
        )
    x = (summary.get("crossover") or {}).get("cost") or []
    if x:
        lines += [
            "",
            "## Is clicking worth it at all? — against the zero-click text sort",
            "",
            "Typing the query and reading the ranked haystack is **free**, so it is the thing a",
            "clicked detector has to beat. `baseline` is that zero-click cost, `final` is the cost",
            "at the horizon, and `crossover` is the first click at which the arm's mean is better",
            "than the baseline. An arm that never crosses is reported as `never`, which is a",
            "finding about that arm and not a missing number.",
            "",
            "| arm | dataset | text sort (0 clicks) | final | clicks to beat it |",
            "|---|---|---:|---:|---:|",
        ]
        for row in sorted(x, key=lambda r: (ARMS.index(r["arm"]) if r["arm"] in ARMS else 99, r["dataset"])):
            ct = row.get("crossover_t")
            crossed = "never" if ct is None or ct != ct else f"{int(ct)}"
            lines.append(
                f"| `{row['arm']}` | {row['dataset']} | {row['baseline']:.3g} | {row['final']:.3g} | {crossed} |"
            )
    sheets = sorted(f for f in figures if f.startswith("opening_") and f.endswith(".jpg"))
    figures = [f for f in figures if f not in sheets]
    # The cost-over-clicks pair leads: it is the figure that answers the
    # question the user actually has ("is what I am building getting better as
    # I keep clicking?"), and every other figure here explains it.
    curve_figs = [f for f in figures if "_vs_clicks" in f]
    figures = curve_figs + [f for f in figures if f not in curve_figs]
    if figures:
        lines += ["", "## Figures", ""]
        for name in figures:
            lines.append(f"![{name}](figures/{name})")
            if name in FIGURE_CAPTIONS:
                lines += ["", f"*{FIGURE_CAPTIONS[name]}*"]
            elif name.startswith(("cost_vs_clicks_runs__", "average_precision_vs_clicks_runs__")):
                metric, ds = name[: -len(".png")].split("_vs_clicks_runs__")
                lines += ["", f"*{_runs_caption(metric, ds)}*"]
            lines.append("")
    lines += [
        "## The openings themselves",
        "",
        "The tables say *whether* an opening mined better.  The issue also asks **why**, and that",
        "is a question about the items, so here are the items: every click of each arm's opening",
        "on one cell, in the order it was made, captioned with its round, its rank in the seed",
        "sort, and whether it turned out to be a positive, with the dataset's ground-truth box",
        "drawn where it has one.  Read two arms side by side and the mechanism is visible rather",
        "than inferred.  Rendered by `make_startup_sheets.py`; a starved arm shows its written",
        "opening in full plus a sample of the clicks it was held for, and the caption says how",
        "many are not shown.",
        "",
    ]
    if sheets:
        by_cell: dict[str, list[str]] = {}
        for f in sheets:
            by_cell.setdefault(f.rsplit("_", 1)[0], []).append(f)

        def _arm_rank(x: str) -> int:
            arm = x.rsplit("_", 1)[1][:-4]
            return ARMS.index(arm) if arm in ARMS else 99

        for cell, files in sorted(by_cell.items()):
            lines += [
                "",
                f"### `{cell[len(chr(111) + chr(112) + chr(101) + chr(110) + chr(105) + chr(110) + chr(103) + chr(95)) :]}`",
                "",
            ]
            for f in sorted(files, key=_arm_rank):
                lines.append(f"![{f}](figures/{f})")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "REPORT_startup.md"
    path.write_text("\n".join(lines))
    return path


def analyze(root: Path, outdir: Path) -> dict:
    main, picks, prov = load_all(root)
    arms_present = [a for a in ARMS if not picks[picks["arm"] == a].empty] if not picks.empty else []
    main, picks, bal = balance(main, picks, arms_present)
    opening = opening_stats(picks)
    traj = trajectory_stats(main)

    summary: dict = {
        "provenance": prov,
        "balance": bal,
        "arms": {},
        "n_main_rows": int(len(main)),
        "n_pick_rows": int(len(picks)),
    }
    for arm in ARMS:
        rec: dict = {"schedule": ARM_SCHEDULE.get(arm, "?")}
        if not opening.empty and (opening["arm"] == arm).any():
            g = opening[opening["arm"] == arm]
            rec["open_clicks"] = float(g["open_clicks"].median())
            # As WRITTEN, and the overrun separately.  Reporting only the total
            # is what made `flat_mid` look like a 200-click opening on a starved
            # cell instead of a 16-click one that could not finish.
            rec["open_scheduled_clicks"] = float(g["open_scheduled_clicks"].median())
            rec["open_yield"] = float(g["open_yield"].median())
            rec["open_cut_depth"] = float(g["open_cut_depth"].median())
            rec["open_pos_depth"] = float(g["open_pos_depth"].median())
            rec["open_overrun_cells"] = int((g["open_overrun"] > 0).sum())
            rec["open_overrun_median"] = float(g["open_overrun"].median())
            rec["open_starved_cells"] = int(g["open_starved"].sum())
            rec["open_starved_pct"] = float(100.0 * g["open_starved"].mean())
            rec["n_good_final"] = float(g["n_good_final"].median())
            rec["n_bad_final"] = float(g["n_bad_final"].median())
            rec["n_cells"] = int(len(g))
        rec["lever"] = lever_moved(opening, arm)
        if arm != CONTROL:
            if not opening.empty:
                for mark in CLICK_MARKS:
                    rec[f"positives_{mark}"] = paired(opening, f"positives_{mark}", arm, CONTROL)
            if not traj.empty:
                for metric in ("final_cost", "final_ap"):
                    rec[metric] = paired(traj, metric, arm, CONTROL)
            if arm != LENGTH_CONTROL:
                vs: dict = {}
                if not opening.empty:
                    vs["positives_100"] = paired(opening, "positives_100", arm, LENGTH_CONTROL)
                if not traj.empty:
                    vs["final_cost"] = paired(traj, "final_cost", arm, LENGTH_CONTROL)
                rec["vs_length_control"] = vs
        summary["arms"][arm] = rec
    summary["verdict"] = verdict(summary)

    agg = outdir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    if not opening.empty:
        opening.to_csv(agg / "opening_stats.csv", index=False)
    if not traj.empty:
        traj.to_csv(agg / "trajectory_stats.csv", index=False)
    baseline = None
    if TEXT_BASELINE and Path(TEXT_BASELINE).exists():
        baseline = curves.text_sort_baseline(TEXT_BASELINE)
    elif TEXT_BASELINE:
        print(f"WARNING: GM_TEXT_BASELINE={TEXT_BASELINE!r} does not exist - curves will have no zero-click anchor")
    figures = make_figures(picks, opening, outdir / "figures", prevalence_table(root), traj, main, baseline)
    # How many clicks each arm needs before it is worth more than the free text
    # sort.  Read off the same curves the figure plots, so the number in the
    # table and the crossing in the picture cannot disagree.
    summary["crossover"] = {}
    for metric in ("cost", "average_precision"):
        curve_csv = outdir / "figures" / f"{metric}_vs_clicks.csv"
        if not curve_csv.exists():
            continue
        x = curves.crossover(pd.read_csv(curve_csv), lower_is_better=metric == "cost")
        if not x.empty:
            summary["crossover"][metric] = x.to_dict(orient="records")
    (outdir / "startup_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(summary, figures, outdir)
    return summary


def main() -> int:
    root = Path(os.environ.get("GM_RESULTS", str(common.EXP / "results")))
    summary = analyze(root, OUT)
    print(summary["verdict"])
    print(f"report -> {OUT / 'REPORT_startup.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
