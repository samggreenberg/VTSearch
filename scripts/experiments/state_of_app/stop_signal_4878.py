"""Price stop rules in the objective's units against Autopilot's Done (#4878).

Autopilot's Done fires mid-climb (#3560): the light that times it, Smart, reads FPR + FNR at a cut no user
sees, and the withheld F-beta rises as fast after Done as before it.  The labels line computes its own
expected F-beta on every retrain; ``stop_signal_4878.sh`` runs today's app with it recorded
(``line_efbeta``, ``line_efbeta_fresh``, ``line_separation``, ``line_head_count``).  This prices, offline on
those sessions, rules that stop when a signal stops rising.

**A rule changes what the app tells the user, not what Autopilot picks**, so its stop is read off the same
session.  A session's *leave* is the rule's stop, or click 150 when it never fires (the user clicks on to the
eval's horizon).  Every rule is scored the same two ways:

* **Where it stops and what is left:** fire rate, mean leave click, the withheld F-beta at leave, and the gain
  from leave to 150 (mean ± SE over class means), as `stopping.py` reads Done.
* **Against a fixed click count.**  A later stop always gains F-beta, so a rule is worth something only if it
  spends its clicks where they pay: stopping the sessions that have stopped improving and keeping the ones that
  have not.  The yardstick is the user who stops at a fixed click N: ``edge`` is the rule's mean F-beta at leave
  minus that of a fixed stop at the rule's own mean leave click (interpolated between whole clicks), paired by
  session, with its SE over class means.  A rule with no edge is a click counter; Done is scored the same way.

Signals, each read on Train at the session's beta (``voting_iterations._line_reading``):

* ``efbeta``: the expected F-beta at the line's cut over the corpus's unvoted items (what ``corpus_cut``
  maximises).  It falls as the session labels positives away, model unchanged.
* ``fresh``: the class model's expected F-beta on a fresh corpus like this one.  Counts no item.
* ``sep``: the labels' d'.

A rule ``{signal}:w{W}:e{E}`` fires at the first click t, from the first click the class model cuts the line
(``line_head_count == 0``; the head's count line before it reads a model too thin to trust), at which the
signal's running best has risen by at most E over the last W clicks.  ``done+`` variants wait for Done too.

    python stop_signal_4878.py --cells 0.25=<b025>/results/cells --cells 1=<b1>/... --cells 4=<b4>/... \\
        --work /expscratch/$USER/stop-4878/work --out docs/experiments/<date>-stop-signal-4878
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import _cells_io  # noqa: E402
import objective  # noqa: E402

from vtscore.eval.autopilot_flow import STOPPING_PHASE  # noqa: E402

#: The eval's horizon: a session whose rule never fires leaves here.
HORIZON = 150
#: The signals a rule can watch, by the column each is read from.
SIGNALS = {"efbeta": "line_efbeta", "fresh": "line_efbeta_fresh", "sep": "line_separation"}
#: The rule grid: windows in clicks, and the rise in the running best that still counts as "stopped rising".
WINDOWS = (10, 20, 30)
EPSILONS = {"efbeta": (0.0, 0.01, 0.02), "fresh": (0.0, 0.01, 0.02), "sep": (0.0, 0.05, 0.1)}
#: Fixed stops for the yardstick.
FIXED = tuple(range(1, HORIZON + 1))
PRESET = {0.25: "β 1/4", 1.0: "β 1", 4.0: "β 4"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def rule_names() -> list[str]:
    names = []
    for sig in SIGNALS:
        for w in WINDOWS:
            for e in EPSILONS[sig]:
                names.append(f"{sig}:w{w}:e{e:g}")
    return names


def _per_click(t: np.ndarray, v: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """The value at each click 1..horizon: the last row at or before it; NaN before the first row."""
    clicks = np.arange(1, horizon + 1)
    idx = np.searchsorted(t, clicks, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
    return out


def _plateau_stop(sig: np.ndarray, start: int, w: int, eps: float) -> int | None:
    """The first click (1-based) from *start* at which the running best rose by at most *eps* over the last *w*."""
    best = np.fmax.accumulate(np.where(np.isnan(sig), -np.inf, sig))
    for c in range(max(start, w + 1), sig.size + 1):
        then, now = best[c - 1 - w], best[c - 1]
        if np.isfinite(then) and now - then <= eps:
            return c
    return None


def session_rows(g: pd.DataFrame) -> dict:
    """One session: its F-beta at every click, Done's stop, and every rule's."""
    g = g.sort_values("t")
    t = g["t"].to_numpy(dtype=float)
    fb = _per_click(t, pd.to_numeric(g[objective.OBJECTIVE], errors="coerce").to_numpy(dtype=float))
    phase = g["phase"].fillna("").astype(str).to_numpy()
    done = np.flatnonzero(phase == STOPPING_PHASE)
    out: dict = {"fbeta": fb, "done": int(t[done[0]]) if done.size else None}
    hc = pd.to_numeric(g["line_head_count"], errors="coerce").to_numpy(dtype=float)
    cut_rows = np.flatnonzero(hc == 0)
    start = int(t[cut_rows[0]]) if cut_rows.size else None
    out["class_model_from"] = start
    for sig, col in SIGNALS.items():
        v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        # Only the class model's reading counts: the head's count line serves while it is too thin.
        v = np.where(hc == 0, v, np.nan)
        out[f"sig_{sig}"] = _per_click(t, v)
    rules: dict[str, int | None] = {}
    for name in rule_names():
        sig, w, e = name.split(":")
        series, win, eps = out[f"sig_{sig}"], int(w[1:]), float(e[1:])
        rules[name] = None if start is None else _plateau_stop(series, start, win, eps)
        # Done first, then the plateau: the first click from Done at which the signal has stopped rising.
        late = None if start is None or out["done"] is None else max(start, out["done"])
        rules[f"done+{name}"] = None if late is None else _plateau_stop(series, late, win, eps)
    out["rules"] = rules
    return out


def _read_cell(path: str, embedder: str) -> list[dict]:
    df = _cells_io._base_rows(pd.read_csv(path, low_memory=False))
    df = df[~_cells_io.check_rows(df)]
    if embedder:
        df = df[df["embedder"] == embedder]
    rows = []
    for (cat, seed), g in df.groupby(["category", "seed"], sort=False):
        s = session_rows(g)
        rec = {"category": cat, "seed": int(seed), "done": s["done"], "class_model_from": s["class_model_from"]}
        rec.update({f"stop[{k}]": v for k, v in s["rules"].items()})
        rec.update({f"fbeta@{c}": float(s["fbeta"][c - 1]) for c in FIXED})
        for sig in SIGNALS:
            rec.update({f"{sig}@{c}": float(s[f"sig_{sig}"][c - 1]) for c in range(5, HORIZON + 1, 5)})
        rows.append(rec)
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


def _cls(category: pd.Series) -> pd.Series:
    return category.astype(str).str.split("@").str[0]


def _se(diff: pd.Series, category: pd.Series) -> float:
    by = diff.groupby(_cls(category)).mean()
    return float(by.std() / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def score(sessions: pd.DataFrame, stop_col: str | None, fixed: int | None = None) -> dict:
    """What a rule leaves each session with, and its edge over a fixed stop at the same mean click."""
    fb = sessions[[f"fbeta@{c}" for c in FIXED]].to_numpy(dtype=float)
    # A session's F-beta before its first scored click is the empty set's: 0 (curves.EMPTY_SET).
    fb = np.nan_to_num(fb, nan=0.0)
    n = len(sessions)
    if fixed is not None:
        leave = np.full(n, fixed)
        fired = np.ones(n, dtype=bool)
    else:
        stops = pd.to_numeric(sessions[stop_col], errors="coerce").to_numpy(dtype=float)
        fired = np.isfinite(stops)
        leave = np.where(fired, stops, HORIZON).astype(int)
    at = fb[np.arange(n), leave - 1]
    final = fb[:, HORIZON - 1]
    mean_leave = float(leave.mean())
    # The yardstick: every session stopped at the same click, the rule's mean, interpolated.
    lo = int(np.floor(mean_leave))
    hi = min(lo + 1, HORIZON)
    frac = mean_leave - lo
    fixed_at = (1 - frac) * fb[:, max(lo, 1) - 1] + frac * fb[:, hi - 1]
    edge = pd.Series(at - fixed_at)
    gain = pd.Series(final - at)
    cat = sessions["category"].reset_index(drop=True)
    return {
        "fire_rate": float(fired.mean()),
        "mean_leave": mean_leave,
        "median_leave": float(np.median(leave)),
        "fbeta_at_leave": float(at.mean()),
        "gain_after": float(gain.mean()),
        "se_gain_after": _se(gain, cat),
        "fixed_at_mean_leave": float(fixed_at.mean()),
        "edge": float(edge.mean()),
        "se_edge": _se(edge, cat),
    }


def load(work: Path) -> pd.DataFrame:
    frames = []
    for beta, name in PRESET.items():
        p = work / f"sessions-{beta:g}.csv.gz"
        if p.exists():
            frames.append(pd.read_csv(p).assign(arm=name, beta=beta))
    return pd.concat(frames, ignore_index=True)


def price(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [c[5:-1] for c in sessions.columns if c.startswith("stop[")]
    for (arm, beta), g in sessions.groupby(["arm", "beta"], sort=False):
        g = g.reset_index(drop=True)
        d = g.assign(**{"stop[done]": g["done"]})
        for rule in ["done", *rules]:
            rows.append({"arm": arm, "beta": beta, "rule": rule, **score(d, f"stop[{rule}]")})
    return pd.DataFrame(rows)


def frontier(sessions: pd.DataFrame) -> pd.DataFrame:
    """The fixed-click yardstick: mean F-beta at every click, per preset."""
    rows = []
    for (arm, beta), g in sessions.groupby(["arm", "beta"], sort=False):
        fb = np.nan_to_num(g[[f"fbeta@{c}" for c in FIXED]].to_numpy(dtype=float), nan=0.0)
        for c, m in zip(FIXED, fb.mean(axis=0), strict=True):
            rows.append({"arm": arm, "beta": beta, "click": c, "fbeta": float(m)})
    return pd.DataFrame(rows)


def calibration_rows(sessions: pd.DataFrame) -> pd.DataFrame:
    """Each signal's mean over clicks beside the withheld F-beta it estimates, per preset."""
    rows = []
    for (arm, beta), g in sessions.groupby(["arm", "beta"], sort=False):
        for c in range(5, HORIZON + 1, 5):
            row = {"arm": arm, "beta": beta, "click": c, "fbeta": float(g[f"fbeta@{c}"].mean())}
            for sig in SIGNALS:
                row[sig] = float(g[f"{sig}@{c}"].mean())
                row[f"{sig}_n"] = int(g[f"{sig}@{c}"].notna().sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def frontier_figure(priced: pd.DataFrame, front: pd.DataFrame, out: Path) -> None:
    """Mean leave click against mean F-beta at leave: the fixed-click curve, Done, and every rule."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arms = list(dict.fromkeys(priced["arm"]))
    fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 3.8), facecolor=SURFACE)
    colors = {"efbeta": "#2a78d6", "fresh": "#eb6834", "sep": "#1baf7a"}
    for ax, arm in zip(np.atleast_1d(axes), arms, strict=True):
        _axes(ax)
        f = front[front["arm"] == arm]
        ax.plot(f["click"], f["fbeta"], color=MUTED, linewidth=1.5, label="stop at a fixed click")
        p = priced[priced["arm"] == arm]
        for sig, color in colors.items():
            q = p[p["rule"].str.startswith(f"{sig}:")]
            ax.scatter(q["mean_leave"], q["fbeta_at_leave"], s=14, color=color, label=f"{sig} plateau", zorder=3)
        d = p[p["rule"] == "done"].iloc[0]
        ax.scatter([d["mean_leave"]], [d["fbeta_at_leave"]], s=60, marker="D", color=INK, label="Done", zorder=4)
        ax.set_title(arm, color=INK, fontsize=10)
        ax.set_xlabel("mean click the session leaves at", color=INK)
    np.atleast_1d(axes)[0].set_ylabel("withheld F-beta at leave (mean)", color=INK)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def calibration_figure(cal: pd.DataFrame, out: Path) -> None:
    """Each signal's mean over clicks beside the withheld F-beta, per preset."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arms = list(dict.fromkeys(cal["arm"]))
    fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 3.6), sharey=True, facecolor=SURFACE)
    for ax, arm in zip(np.atleast_1d(axes), arms, strict=True):
        _axes(ax)
        g = cal[cal["arm"] == arm]
        ax.plot(g["click"], g["fbeta"], color=INK, linewidth=2, label="withheld F-beta")
        ax.plot(g["click"], g["efbeta"], color="#2a78d6", linewidth=1.5, label="efbeta (unvoted corpus)")
        ax.plot(g["click"], g["fresh"], color="#eb6834", linewidth=1.5, label="fresh (fresh corpus)")
        ax.set_title(arm, color=INK, fontsize=10)
        ax.set_xlabel("clicks", color=INK)
    np.atleast_1d(axes)[0].set_ylabel("F-beta (mean over sessions with a class model)", color=INK)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def report(work: Path, out: Path) -> None:
    sessions = load(work)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    priced = price(sessions)
    priced.to_csv(out / "rules.csv", index=False, float_format="%.4f")
    front = frontier(sessions)
    front.to_csv(out / "fixed_stop.csv", index=False, float_format="%.4f")
    cal = calibration_rows(sessions)
    cal.to_csv(out / "signals_over_clicks.csv", index=False, float_format="%.4f")
    frontier_figure(priced, front, out / "figures" / "frontier.png")
    calibration_figure(cal, out / "figures" / "signals.png")
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
    args = ap.parse_args()
    if args.cmd == "read":
        read(args.cells, args.embedder, args.work, args.nproc)
    else:
        report(args.work, args.out)


if __name__ == "__main__":
    main()
