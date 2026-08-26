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
CLICK_MARKS = (10, 20, 50, 100)

OUT = Path(os.environ.get("GM_OUT", str(common.EXP / "analysis")))
KEYS = ("dataset", "embedder", "category", "seed")


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
            open_overrun=int(max(0, len(op) - _declared_clicks(g))),
            trained_at=int(g.loc[g["phase"].astype(str).isin(("hard", "new", "done")), "t"].min())
            if g["phase"].astype(str).isin(("hard", "new", "done")).any()
            else -1,
        )
        for mark in CLICK_MARKS:
            head = g[g["t"] <= mark]
            rec[f"positives_{mark}"] = int(head["picked_label"].sum())
        out.append(rec)
    return pd.DataFrame(out)


def _declared_clicks(g: pd.DataFrame) -> int:
    """How many opening clicks the arm's schedule *asked* for.

    Read off the pick log rather than re-parsing the spec, so an arm whose
    rounds were cut short by a small pool is measured as it ran.
    """
    rounds = g[g["startup_round"] >= 0]
    if rounds.empty:
        return int(len(g[g["phase"].astype(str).isin(("good", "bad"))]))
    # Every click in a round the trajectory genuinely entered, minus the ones
    # spent held on the last round waiting for the missing vote class.
    last = int(rounds["startup_round"].max())
    return int((rounds["startup_round"] < last).sum() + min((rounds["startup_round"] == last).sum(), len(rounds)))


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
    fals = summary["arms"].get(FALSIFIER, {})
    fals_pos = fals.get("positives_100", {})
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


def make_figures(picks: pd.DataFrame, opening: pd.DataFrame, outdir: Path) -> list[str]:
    """Mining curve (mean and per-run), opening depth, and yield by depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
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
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(d: dict, key: str, digits: int = 2) -> str:
    v = d.get(key)
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}g}"


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
        "| arm | open clicks | open yield | positives@100 Δ | [95% CI] | final cost Δ | [95% CI] | AP Δ |",
        "|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for arm in ARMS:
        if arm == CONTROL:
            continue
        rec = summary["arms"].get(arm, {})
        pos, cost, ap = rec.get("positives_100", {}), rec.get("final_cost", {}), rec.get("final_ap", {})
        lines.append(
            f"| `{arm}` | {_fmt(rec, 'open_clicks')} | {_fmt(rec, 'open_yield')} | "
            f"{_fmt(pos, 'median_delta')} | [{_fmt(pos, 'ci95_lo')}, {_fmt(pos, 'ci95_hi')}] | "
            f"{_fmt(cost, 'median_delta')} | [{_fmt(cost, 'ci95_lo')}, {_fmt(cost, 'ci95_hi')}] | "
            f"{_fmt(ap, 'median_delta')} |"
        )
    lines += [
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
            f"| `{arm}` | {_fmt(pos, 'median_delta')} | [{_fmt(pos, 'ci95_lo')}, {_fmt(pos, 'ci95_hi')}] | "
            f"{_fmt(cost, 'median_delta')} | [{_fmt(cost, 'ci95_lo')}, {_fmt(cost, 'ci95_hi')}] |"
        )
    if figures:
        lines += ["", "## Figures", ""]
        for name in figures:
            lines.append(f"![{name}](figures/{name})")
            lines.append("")
    lines += [
        "## What is still owed",
        "",
        "This analyzer reports *whether* an opening mined better.  The issue also asks **why**,",
        "and that is a question about the items themselves: dump the opening's picks with",
        "`VTS_DUMP_TEST_SCORES` and render them with `make_error_sheets.py`, so a winning arm's",
        "extra positives can be looked at rather than counted.  On image data, show the images.",
        "",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "REPORT_startup.md"
    path.write_text("\n".join(lines))
    return path


def analyze(root: Path, outdir: Path) -> dict:
    main, picks, prov = load_all(root)
    opening = opening_stats(picks)
    traj = trajectory_stats(main)

    summary: dict = {"provenance": prov, "arms": {}, "n_main_rows": int(len(main)), "n_pick_rows": int(len(picks))}
    for arm in ARMS:
        rec: dict = {"schedule": ARM_SCHEDULE.get(arm, "?")}
        if not opening.empty and (opening["arm"] == arm).any():
            g = opening[opening["arm"] == arm]
            rec["open_clicks"] = float(g["open_clicks"].median())
            rec["open_yield"] = float(g["open_yield"].median())
            rec["open_cut_depth"] = float(g["open_cut_depth"].median())
            rec["open_pos_depth"] = float(g["open_pos_depth"].median())
            rec["open_overrun_cells"] = int((g["open_overrun"] > 0).sum())
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
    figures = make_figures(picks, opening, outdir / "figures")
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
