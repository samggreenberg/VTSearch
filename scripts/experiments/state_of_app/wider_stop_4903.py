"""Autopilot's stop past the eval's horizon, and a click cap to back it (#4903).

Every stop study before this one (#3560, #4878) ran sessions to 150 clicks, so a session no rule stopped simply
left at 150.  The app has no horizon: a user Autopilot never tells to stop either quits when they like or clicks
on through the collection.  ``wider_stop_4903.sh`` runs today's app (Done on the lights or the labels line's
level, #4878) to 400 clicks; this reads three things off those sessions:

* **Where each rule stops** at 150, 200, 250, 300 and 400 clicks: the app's own Done (``phase == "done"``, the
  shipped ``lights|level``), and, read offline off the same sessions, the lights alone (the first click all three
  read green) and the level alone (the first click the class model's ``line_efbeta`` reads at least
  :data:`~vtscore.training.thresholds.labels_line.STOP_LEVEL`).  The offline two are the click their condition
  first holds; the app's Done comes a check later (#4842's Spot Check, a few clicks), and past the first stop the
  session is the app's, not the one the rule alone would have run.
* **The sessions no rule stops**: which band, what F-beta they hold, and whether it is still climbing.
* **A click cap as the backstop**: the rule, or N clicks, whichever comes first, priced the way
  ``stop_signal_4878.py`` prices a rule, at the 400-click horizon: the F-beta a session leaves with, the gain it
  leaves behind (to 400), its edge over a stop at a fixed click at the same mean leave, and over the same leave
  clicks dealt at random.  Plus what the cap costs the sessions it cuts: the F-beta they would have had at the
  rule's stop (or at 400), less the F-beta at N.

    python wider_stop_4903.py read --cells 0.25=<b025>/results/cells --cells 1=<b1>/... --cells 4=<b4>/... \\
        --work /expscratch/$USER/stop-4903/work
    python wider_stop_4903.py report --work /expscratch/$USER/stop-4903/work \\
        --out docs/experiments/<date>-wider-stop-4903 [--ab-work /expscratch/$USER/stop-4878/ab]

``--ab-work`` checks the first 150 clicks against #4878's ``or`` arm, the same rule at the 150-click horizon.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import _cells_io  # noqa: E402
import objective  # noqa: E402
import stop_signal_4878 as S  # noqa: E402
import stopping  # noqa: E402

from vtscore.eval.autopilot_flow import STOPPING_PHASE  # noqa: E402
from vtscore.training.thresholds import STOP_LEVEL  # noqa: E402

#: How far the sessions run.
HORIZON = 400
CLICKS = tuple(range(1, HORIZON + 1))
#: The eval's horizon, where every earlier study left the sessions no rule stopped.
EVAL_HORIZON = S.HORIZON
#: Where the survival table reads each rule.
MARKS = (150, 200, 250, 300, 400)
#: The caps priced.  ``None`` is no cap: a session the rule never stops leaves at :data:`HORIZON`.
CAPS = (100, 150, 200, 250, 300, 350, None)
#: The rules: the app's Done, and the two halves of it read offline.
RULES = ("done", "lights", "level")
RULE_LABEL = {"done": "the app's Done (lights or level)", "lights": "lights alone", "level": "level alone"}
PRESET = S.PRESET
#: The windows that say whether a session's F-beta is still climbing at the horizon.
TAIL = 100
#: A session gaining less than this over the last :data:`TAIL` clicks has gone flat.
FLAT = 0.01
#: Autopilot's opening: no rule reads the lights or the level here, and a cap past it never ends a session that
#: never leaves it (#4216's starved hunts, still walking the typed query's sort for its first Goods).
OPENING = frozenset({"good", "bad", "more", "example", ""})


def session_record(g: pd.DataFrame) -> dict:
    """One session: its withheld F-beta at every click, and the click each rule first holds."""
    g = g.sort_values("t")
    t = g["t"].to_numpy(dtype=float)
    fb = S._per_click(t, pd.to_numeric(g[objective.OBJECTIVE], errors="coerce").to_numpy(dtype=float), HORIZON)
    phase = g["phase"].fillna("").astype(str).to_numpy()
    done = np.flatnonzero(phase == STOPPING_PHASE)
    green = np.ones(len(g), dtype=bool)
    for light in ("smart", "stable", "span"):
        green &= g[light].fillna("").astype(str).to_numpy() == "green"
    lights = np.flatnonzero(green)
    hc = pd.to_numeric(g["line_head_count"], errors="coerce").to_numpy(dtype=float)
    ef = pd.to_numeric(g["line_efbeta"], errors="coerce").to_numpy(dtype=float)
    # The app reads no level while the head's count line serves (stop_level_reading).
    level = np.flatnonzero((hc == 0) & (ef >= STOP_LEVEL))
    trained = np.flatnonzero(~np.isin(phase, list(OPENING)))
    n_good = pd.to_numeric(g["n_good"], errors="coerce").to_numpy(dtype=float)
    rec: dict = {
        "last_click": int(t[-1]),
        "trained_from": int(t[trained[0]]) if trained.size else np.nan,
        "stop[done]": int(t[done[0]]) if done.size else np.nan,
        "stop[lights]": int(t[lights[0]]) if lights.size else np.nan,
        "stop[level]": int(t[level[0]]) if level.size else np.nan,
        "done_rows": int(done.size),
    }
    # What holds a session at the horizon: each light's share of clicks past the eval's horizon it read green,
    # and the level reading at marks.
    past = t > EVAL_HORIZON
    for light in ("smart", "stable", "span"):
        col = g[light].fillna("").astype(str).to_numpy()
        rec[f"{light}_green_past"] = float((col[past] == "green").mean()) if past.any() else np.nan
    ef_click = S._per_click(t, np.where(hc == 0, ef, np.nan), HORIZON)
    good_click = S._per_click(t, n_good, HORIZON)
    for c in MARKS:
        rec[f"efbeta@{c}"] = float(ef_click[c - 1])
        rec[f"n_good@{c}"] = float(good_click[c - 1])
    rec.update({f"fbeta@{c}": float(fb[c - 1]) for c in CLICKS})
    return rec


def _read_cell(path: str, embedder: str) -> list[dict]:
    df = _cells_io._base_rows(pd.read_csv(path, low_memory=False))
    df = df[~_cells_io.check_rows(df)]
    if embedder:
        df = df[df["embedder"] == embedder]
    rows = []
    for (cat, seed), g in df.groupby(["category", "seed"], sort=False):
        rows.append({"category": cat, "seed": int(seed), **session_record(g)})
    return rows


def read(cells: list[str], embedder: str, work: Path, nproc: int) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for spec in cells:
        beta, path = spec.split("=", 1)
        files = [str(p) for p in _cells_io.main_frame_files(path)]
        with Pool(nproc) as pool:
            parts = pool.map(partial(_read_cell, embedder=embedder), files, chunksize=8)
        frame = pd.DataFrame([r for p in parts for r in p])
        frame.to_csv(work / f"sessions-{beta}.csv.gz", index=False, float_format="%.5g")
        print(f"beta {beta}: {len(files)} cells, {len(frame)} sessions", flush=True)


def load(work: Path) -> pd.DataFrame:
    frames = []
    for beta, name in PRESET.items():
        p = work / f"sessions-{beta:g}.csv.gz"
        if p.exists():
            d = pd.read_csv(p).assign(preset=name, beta=beta)
            frames.append(d.assign(band=d["category"].astype(str).str.split("@").str[1].fillna("")))
    return pd.concat(frames, ignore_index=True)


def _fb(g: pd.DataFrame) -> np.ndarray:
    # A click before a session's first scored row is the empty set's: F-beta 0 (curves.EMPTY_SET).
    return np.nan_to_num(g[[f"fbeta@{c}" for c in CLICKS]].to_numpy(dtype=float), nan=0.0)


def _stops(g: pd.DataFrame, rule: str) -> np.ndarray:
    return pd.to_numeric(g[f"stop[{rule}]"], errors="coerce").to_numpy(dtype=float)


def survival(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per preset and rule: the share of sessions the rule has stopped by each mark, and its KM median stop."""
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        for rule in RULES:
            s = _stops(g, rule)
            fired = np.isfinite(s)
            row = {"preset": preset, "beta": beta, "rule": rule, "sessions": len(g)}
            row.update({f"stopped_by_{c}": float((fired & (s <= c)).mean()) for c in MARKS})
            row["km_median"] = stopping.km_median(s[fired], np.full(int((~fired).sum()), HORIZON))
            if rule == "done":
                row.update(check_split(g))
            row["km_median_at_150"] = stopping.km_median(
                np.where(fired & (s <= EVAL_HORIZON), s, np.nan),
                np.full(int((~(fired & (s <= EVAL_HORIZON))).sum()), EVAL_HORIZON),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def score_leave(fb: np.ndarray, leave: np.ndarray, category: pd.Series) -> dict:
    """``stop_signal_4878.score`` at this study's horizon, on the leave clicks given.

    *fb* is every session's F-beta at clicks 1..:data:`HORIZON`; *leave* the click each leaves at.
    """
    n = len(leave)
    leave = leave.astype(int)
    at = fb[np.arange(n), leave - 1]
    final = fb[:, HORIZON - 1]
    mean_leave = float(leave.mean())
    lo = int(np.floor(mean_leave))
    hi = min(lo + 1, HORIZON)
    frac = mean_leave - lo
    fixed_at = (1 - frac) * fb[:, max(lo, 1) - 1] + frac * fb[:, hi - 1]
    cat = category.reset_index(drop=True)
    edge = pd.Series(at - fixed_at)
    shuffle_edge = pd.Series(at - fb[:, leave - 1].mean(axis=1))
    gain = pd.Series(final - at)
    return {
        "mean_leave": mean_leave,
        "median_leave": float(np.median(leave)),
        "fbeta_at_leave": float(at.mean()),
        "gain_after": float(gain.mean()),
        "se_gain_after": S._se(gain, cat),
        "edge": float(edge.mean()),
        "se_edge": S._se(edge, cat),
        "shuffle_edge": float(shuffle_edge.mean()),
        "se_shuffle_edge": S._se(shuffle_edge, cat),
    }


def check_clicks(g: pd.DataFrame) -> float:
    """The clicks the app's check before Done takes: its Done, less the first click a light or the level said stop.

    The mean over the sessions the app's Done stopped.  It is two populations: about half skip the check (a
    weak-separation check ended in the last 25 votes, ``done_check_due``) and take 0, the rest take 20-30, so
    the median says nothing; a capped session's check is priced at the mean.
    """
    done = _stops(g, "done")
    first = np.fmin(_stops(g, "lights"), _stops(g, "level"))
    ok = np.isfinite(done) & np.isfinite(first) & (first <= done)
    return float(np.mean(done[ok] - first[ok])) if ok.any() else 0.0


def check_split(g: pd.DataFrame) -> dict:
    """The check before Done's two populations: the share that skip it, and the clicks the rest take."""
    done = _stops(g, "done")
    first = np.fmin(_stops(g, "lights"), _stops(g, "level"))
    ok = np.isfinite(done) & np.isfinite(first) & (first <= done)
    took = (done - first)[ok]
    ran = took[took > 0]
    return {
        "check_skipped": float((took == 0).mean()) if took.size else np.nan,
        "check_clicks_when_run": float(np.median(ran)) if ran.size else np.nan,
        "check_clicks_mean": float(took.mean()) if took.size else np.nan,
    }


#: Where a cap counts from.  ``votes``: every vote, the opening's too, so a session still walking for its first
#: Goods stops at N as well (no check: there is no line to check).  ``trained``: the cap is read where the lights
#: and the level are, past the opening, so a session that leaves the opening after N stops as soon as it does,
#: and one that never leaves it is never capped.  ``since``: N votes from the click the session leaves the
#: opening, so a session that starts training late gets the same budget as one that starts early (one that never
#: leaves it is never capped; a stop past :data:`HORIZON` reads as leaving there).  A trained session's Done
#: comes a check after the cap.
CAP_KINDS = ("votes", "trained", "since")


def cap_leave(g: pd.DataFrame, rule: str, cap: int | None, kind: str, check: float) -> tuple[np.ndarray, np.ndarray]:
    """Each session's leave click under *rule* or *cap*, and where the rule alone would have left it."""
    s = _stops(g, rule)
    natural = np.where(np.isfinite(s), s, HORIZON)
    if cap is None:
        return natural, natural
    trained = pd.to_numeric(g["trained_from"], errors="coerce").to_numpy(dtype=float)
    in_opening = ~np.isfinite(trained) | (trained > cap)
    if kind == "votes":
        point = np.where(in_opening, cap, cap + check)
    elif kind == "trained":
        point = np.where(np.isfinite(trained), np.maximum(cap, trained) + check, np.inf)
    else:
        point = np.where(np.isfinite(trained), trained + cap + check, np.inf)
    leave = np.clip(np.minimum(natural, np.round(point)), 1, HORIZON)
    return leave, natural


def caps(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per preset, rule, cap kind (:data:`CAP_KINDS`) and cap: the rule or N clicks, whichever comes first.

    ``capped`` is the share of sessions the cap ends; ``forgone`` the mean F-beta those sessions give up, from the
    cap to where the rule alone would have left them (its stop, or :data:`HORIZON`), with ``forgone_clicks`` the
    clicks they save; ``forgone_per_100`` the rate, per 100 clicks, at which they would have bought it.
    """
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        g = g.reset_index(drop=True)
        fb = _fb(g)
        idx = np.arange(len(g))
        check = check_clicks(g)
        for rule in RULES:
            for kind in CAP_KINDS:
                for cap in CAPS:
                    if cap is None and kind != CAP_KINDS[0]:
                        continue
                    leave, natural = cap_leave(g, rule, cap, kind, check)
                    cut = leave < natural
                    gave_up = (fb[idx, natural.astype(int) - 1] - fb[idx, leave.astype(int) - 1])[cut]
                    clicks = (natural - leave)[cut]
                    rows.append(
                        {
                            "preset": preset,
                            "beta": beta,
                            "rule": rule,
                            "kind": "none" if cap is None else kind,
                            "cap": "none" if cap is None else str(cap),
                            "check_clicks": check,
                            "capped": float(cut.mean()),
                            **score_leave(fb, leave, g["category"]),
                            "forgone": float(gave_up.mean()) if cut.any() else np.nan,
                            "forgone_clicks": float(clicks.mean()) if cut.any() else np.nan,
                            "forgone_per_100": float(100 * gave_up.sum() / clicks.sum()) if cut.any() else np.nan,
                            "never_stopped": float((leave >= HORIZON).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def groups(g: pd.DataFrame) -> pd.Series:
    """Each session's group by the app's Done: by the eval's horizon, past it, or never by 400.

    A session never Done splits by whether it left the opening: ``opening`` never did (no rule ever read it),
    ``never`` trained a detector the rules never stopped.
    """
    s = _stops(g, "done")
    trained = np.isfinite(pd.to_numeric(g["trained_from"], errors="coerce").to_numpy(dtype=float))
    out = np.where(
        np.isfinite(s),
        np.where(s <= EVAL_HORIZON, "by 150", "151-400"),
        np.where(trained, "never", "opening"),
    )
    return pd.Series(out, index=g.index)


GROUPS = ("by 150", "151-400", "never", "opening")
GROUP_LABEL = {
    "by 150": "Done by 150",
    "151-400": "Done at 151-400",
    "never": "trained, never Done",
    "opening": "never left the opening",
}


def never(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per preset and band: how many sessions the app's Done leaves going, and what they hold.

    ``climbing`` is the share of never-stopped sessions still gaining at least :data:`FLAT` over the last
    :data:`TAIL` clicks; ``near_zero`` the share holding an F-beta under 0.1 at 400.
    """
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        for scope, h in [("all", g), *((str(b), x) for b, x in g.groupby("band", sort=True))]:
            grp = groups(h)
            nv = h[grp == "never"]
            row = {
                "preset": preset,
                "beta": beta,
                "band": scope,
                "sessions": len(h),
                **{f"share_{k}": float((grp == k).mean()) for k in GROUPS},
            }
            if len(nv):
                tail = nv[f"fbeta@{HORIZON}"] - nv[f"fbeta@{HORIZON - TAIL}"]
                row.update(
                    {
                        "never_fbeta@150": float(nv["fbeta@150"].mean()),
                        "never_fbeta@200": float(nv["fbeta@200"].mean()),
                        "never_fbeta@400": float(nv[f"fbeta@{HORIZON}"].mean()),
                        "never_gain_150_400": float((nv[f"fbeta@{HORIZON}"] - nv["fbeta@150"]).mean()),
                        "never_gain_last100": float(tail.mean()),
                        "never_climbing": float((tail >= FLAT).mean()),
                        "never_near_zero": float((nv[f"fbeta@{HORIZON}"] < 0.1).mean()),
                        "never_efbeta@400": float(nv[f"efbeta@{HORIZON}"].mean()),
                        "never_no_class_model": float(nv[f"efbeta@{HORIZON}"].isna().mean()),
                        "never_smart_green_past": float(nv["smart_green_past"].mean()),
                        "never_stable_green_past": float(nv["stable_green_past"].mean()),
                        "never_span_green_past": float(nv["span_green_past"].mean()),
                        "never_trained_from": float(nv["trained_from"].mean()),
                    }
                )
            op = h[grp == "opening"]
            if len(op):
                row.update(
                    {
                        "opening_fbeta@400": float(op[f"fbeta@{HORIZON}"].mean()),
                        "opening_n_good@150": float(op["n_good@150"].mean()),
                        "opening_n_good@400": float(op[f"n_good@{HORIZON}"].mean()),
                        "opening_no_good@400": float((op[f"n_good@{HORIZON}"].fillna(0) == 0).mean()),
                    }
                )
            # The sessions that leave the opening late: no rule reads a session before it does.
            tf = pd.to_numeric(h["trained_from"], errors="coerce")
            for c in (EVAL_HORIZON, 200):
                late = h[~(tf <= c)]
                row[f"share_opening_at_{c}"] = len(late) / len(h)
                if len(late):
                    row[f"opening_at_{c}_fbeta@{c}"] = float(late[f"fbeta@{c}"].mean())
                    row[f"opening_at_{c}_fbeta@400"] = float(late[f"fbeta@{HORIZON}"].mean())
                    row[f"opening_at_{c}_n_good@{c}"] = float(late[f"n_good@{c}"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def over_clicks(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per preset and group: mean withheld F-beta at every click."""
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        grp = groups(g)
        for k in ("all", *GROUPS):
            h = g if k == "all" else g[grp == k]
            if h.empty:
                continue
            for c, m in zip(CLICKS, _fb(h).mean(axis=0), strict=True):
                rows.append({"preset": preset, "beta": beta, "group": k, "sessions": len(h), "click": c, "fbeta": m})
    return pd.DataFrame(rows)


def against_150(sessions: pd.DataFrame, ab_work: Path) -> pd.DataFrame:
    """The first 150 clicks beside #4878's ``or`` arm: the same rule, run to the eval's horizon.

    A session's first 150 clicks do not read the step budget, so they should be the same session.
    """
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        p = ab_work / f"sessions-{beta:g}-or.csv.gz"
        if not p.exists():
            continue
        o = pd.read_csv(p).set_index(["category", "seed"])
        a = g.set_index(["category", "seed"])
        both = a.index.intersection(o.index)
        if both.empty:
            continue
        cols = [f"fbeta@{c}" for c in range(1, EVAL_HORIZON + 1)]
        fa = np.nan_to_num(a.loc[both, cols].to_numpy(dtype=float), nan=0.0)
        f_or = np.nan_to_num(o.loc[both, cols].to_numpy(dtype=float), nan=0.0)
        da = pd.to_numeric(a.loc[both, "stop[done]"], errors="coerce")
        da = da.where(da <= EVAL_HORIZON)
        do = pd.to_numeric(o.loc[both, "done"], errors="coerce")
        same_done = (da.fillna(-1).to_numpy() == do.fillna(-1).to_numpy()).mean()
        rows.append(
            {
                "preset": preset,
                "beta": beta,
                "sessions": len(both),
                "identical_fbeta_1_150": float(np.all(np.abs(fa - f_or) < 1e-4, axis=1).mean()),
                "session_mean_diff": float((fa.mean(axis=1) - f_or.mean(axis=1)).mean()),
                "same_done_by_150": float(same_done),
            }
        )
    return pd.DataFrame(rows)


GROUP_COLORS = {"all": S.INK, "by 150": "#2a78d6", "151-400": "#eb6834", "never": "#1baf7a", "opening": "#898781"}
RULE_COLORS = {"done": S.INK, "lights": "#2a78d6", "level": "#eb6834"}


def _plt():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def survival_figure(sessions: pd.DataFrame, out: Path) -> None:
    """The share of sessions each rule has not yet stopped, over clicks to 400, one panel a preset."""
    plt = _plt()
    presets = list(dict.fromkeys(sessions["preset"]))
    fig, axes = plt.subplots(1, len(presets), figsize=(3.8 * len(presets), 3.6), sharey=True, facecolor=S.SURFACE)
    for ax, preset in zip(np.atleast_1d(axes), presets, strict=True):
        S._axes(ax)
        g = sessions[sessions["preset"] == preset]
        for rule in RULES:
            s = _stops(g, rule)
            going = [float((~(np.isfinite(s) & (s <= c))).mean()) for c in CLICKS]
            ax.plot(CLICKS, going, color=RULE_COLORS[rule], linewidth=1.8, label=RULE_LABEL[rule])
        ax.axvline(EVAL_HORIZON, color=S.MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(preset, color=S.INK, fontsize=10)
        ax.set_xlabel("clicks", color=S.INK)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("share of sessions not yet stopped", color=S.INK)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=130, facecolor=S.SURFACE)
    plt.close(fig)


def groups_figure(curves: pd.DataFrame, out: Path) -> None:
    """Mean withheld F-beta over clicks to 400, by when the app's Done first fired, one panel a preset."""
    plt = _plt()
    presets = list(dict.fromkeys(curves["preset"]))
    fig, axes = plt.subplots(1, len(presets), figsize=(3.8 * len(presets), 3.6), sharey=True, facecolor=S.SURFACE)
    for ax, preset in zip(np.atleast_1d(axes), presets, strict=True):
        S._axes(ax)
        ends = []
        for k in GROUPS:
            g = curves[(curves["preset"] == preset) & (curves["group"] == k)]
            if g.empty:
                continue
            n = int(g["sessions"].iloc[0])
            ax.plot(g["click"], g["fbeta"], color=GROUP_COLORS[k], linewidth=1.8, label=GROUP_LABEL[k])
            ends.append([float(g["fbeta"].iloc[-1]), f"n={n}"])
        # Each line's count at its end, spread so that lines ending close together keep readable labels.
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
        for y, text in ends:
            ax.annotate(
                text, (HORIZON, y), xytext=(3, 0), textcoords="offset points", fontsize=7, color=S.INK, va="center"
            )
        ax.axvline(EVAL_HORIZON, color=S.MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(preset, color=S.INK, fontsize=10)
        ax.set_xlabel("clicks", color=S.INK)
        ax.set_xlim(0, HORIZON + 40)
    np.atleast_1d(axes)[0].set_ylabel("withheld F-beta", color=S.INK)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=130, facecolor=S.SURFACE)
    plt.close(fig)


KIND_LABEL = {"votes": "every vote", "trained": "every vote, past the opening", "since": "votes since the opening"}
KIND_COLORS = {"votes": "#2a78d6", "trained": "#eb6834", "since": "#1baf7a"}


def caps_figure(priced: pd.DataFrame, out: Path) -> None:
    """The app's Done or a cap at N, per cap kind: the F-beta it leaves, and the clicks it takes, against no cap."""
    plt = _plt()
    d = priced[priced["rule"] == "done"]
    presets = list(dict.fromkeys(d["preset"]))
    fig, axes = plt.subplots(
        2,
        len(presets),
        figsize=(3.8 * len(presets), 6.2),
        sharex=True,
        sharey="row",
        facecolor=S.SURFACE,
        squeeze=False,
    )
    for col, preset in enumerate(presets):
        g = d[d["preset"] == preset]
        base = g[g["kind"] == "none"].iloc[0]
        for row, (metric, label) in enumerate(
            (
                ("fbeta_at_leave", "F-beta at the stop,\nless no cap's"),
                ("mean_leave", "clicks to the stop,\nless no cap's"),
            )
        ):
            ax = axes[row][col]
            S._axes(ax)
            ax.axhline(0, color=S.MUTED, linewidth=0.8)
            for kind in CAP_KINDS:
                k = g[g["kind"] == kind].assign(n=lambda x: x["cap"].astype(int)).sort_values("n")
                ax.plot(
                    k["n"],
                    k[metric] - base[metric],
                    color=KIND_COLORS[kind],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    label=f"cap counts {KIND_LABEL[kind]}",
                )
            if row == 0:
                ax.set_title(preset, color=S.INK, fontsize=10)
            else:
                ax.set_xlabel("cap N (clicks)", color=S.INK)
            if col == 0:
                ax.set_ylabel(label, color=S.INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=130, facecolor=S.SURFACE)
    plt.close(fig)


def report(work: Path, out: Path, ab_work: Path | None) -> None:
    sessions = load(work)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    survival(sessions).to_csv(out / "survival.csv", index=False, float_format="%.4f")
    priced = caps(sessions)
    priced.to_csv(out / "caps.csv", index=False, float_format="%.4f")
    never(sessions).to_csv(out / "never.csv", index=False, float_format="%.4f")
    curves = over_clicks(sessions)
    curves.to_csv(out / "over_clicks.csv", index=False, float_format="%.4f")
    if ab_work is not None:
        against_150(sessions, ab_work).to_csv(out / "against_150.csv", index=False, float_format="%.4f")
    survival_figure(sessions, out / "figures" / "survival.png")
    groups_figure(curves, out / "figures" / "by_group.png")
    caps_figure(priced, out / "figures" / "caps.png")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--cells", action="append", required=True, help="BETA=<run dir>/results/cells")
    r.add_argument("--embedder", default="siglip")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--nproc", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ab-work", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "read":
        read(args.cells, args.embedder, args.work, args.nproc)
    else:
        report(args.work, args.out, args.ab_work)


if __name__ == "__main__":
    main()
