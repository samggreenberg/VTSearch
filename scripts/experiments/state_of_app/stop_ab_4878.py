"""The A/B of #4878's stop rules: each arm against today's app, session for session.

``stop_signal_4878.sh`` runs the control (``CALIB_STOP_RULE`` unset: Done once every light is green) and three
arms on the same cells, seeds and commit lineage: ``level`` (Done on the labels line's expected F-beta alone),
``and`` (the lights and the level), ``or`` (either), each at a level of 0.5.  A stop rule decides what the app
tells the user, and through the phase also what Autopilot picks (the atlas walk can end sooner, the check
before Done runs at the stop), so the A/B reads both:

* **The session** - the owner's one number (``eval-horizon-not-user-horizon``): the withheld F-beta averaged
  over clicks 1-150, paired with the control's same session.  A stop rule that changes Autopilot's picks
  shows up here; one that only moves the announcement does not.
* **The stop** - where each arm says Done, the F-beta there, the gain from there to 150, and the two edges
  ``stop_signal_4878.py`` judges a rule by: over a stop at a fixed click, and over the arm's own stops dealt at
  random.

    python stop_ab_4878.py --run 0.25:control=<dir>/results/cells --run 0.25:level=<dir>/results/cells ... \\
        --work /expscratch/$USER/stop-4878/ab --out docs/experiments/<date>-stop-ab-4878
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

HORIZON = S.HORIZON
ARMS = ("control", "level", "and", "or")
ARM_LABEL = {"control": "today (lights)", "level": "level", "and": "lights + level", "or": "lights or level"}
PRESET = S.PRESET


def _read_cell(path: str, embedder: str) -> list[dict]:
    df = _cells_io._base_rows(pd.read_csv(path, low_memory=False))
    df = df[~_cells_io.check_rows(df)]
    if embedder:
        df = df[df["embedder"] == embedder]
    rows = []
    for (cat, seed), g in df.groupby(["category", "seed"], sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy(dtype=float)
        fb = S._per_click(t, pd.to_numeric(g[objective.OBJECTIVE], errors="coerce").to_numpy(dtype=float))
        done = np.flatnonzero(g["phase"].fillna("").astype(str).to_numpy() == STOPPING_PHASE)
        rec = {"category": cat, "seed": int(seed), "done": int(t[done[0]]) if done.size else np.nan}
        rec.update({f"fbeta@{c}": float(fb[c - 1]) for c in S.FIXED})
        rows.append(rec)
    return rows


def read(runs: list[str], embedder: str, work: Path, nproc: int) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for spec in runs:
        key, path = spec.split("=", 1)
        beta, arm = key.split(":")
        files = [str(p) for p in _cells_io.main_frame_files(path)]
        with Pool(nproc) as pool:
            parts = pool.map(partial(_read_cell, embedder=embedder), files, chunksize=8)
        frame = pd.DataFrame([r for p in parts for r in p])
        frame.to_csv(work / f"sessions-{beta}-{arm}.csv.gz", index=False, float_format="%.5g")
        print(f"beta {beta} {arm}: {len(files)} cells, {len(frame)} sessions", flush=True)


def load(work: Path) -> pd.DataFrame:
    frames = []
    for beta, name in PRESET.items():
        for arm in ARMS:
            p = work / f"sessions-{beta:g}-{arm}.csv.gz"
            if p.exists():
                frames.append(pd.read_csv(p).assign(preset=name, beta=beta, arm=arm))
    return pd.concat(frames, ignore_index=True)


def _fb(g: pd.DataFrame) -> np.ndarray:
    # A click before a session's first scored row is the empty set's: F-beta 0 (curves.EMPTY_SET).
    return np.nan_to_num(g[[f"fbeta@{c}" for c in S.FIXED]].to_numpy(dtype=float), nan=0.0)


def ab_rows(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per preset and arm: the session's mean F-beta paired with the control's, and where the arm says Done."""
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        ctl = g[g["arm"] == "control"].set_index(["category", "seed"])
        for arm in ARMS:
            a = g[g["arm"] == arm].set_index(["category", "seed"])
            both = a.index.intersection(ctl.index)
            if both.empty:
                continue
            a, c = a.loc[both], ctl.loc[both]
            mean_a = _fb(a).mean(axis=1)
            mean_c = _fb(c).mean(axis=1)
            diff = pd.Series(mean_a - mean_c, index=both)
            cats = pd.Series(both.get_level_values("category"), index=both)
            stop = score_stop(a.reset_index())
            rows.append(
                {
                    "preset": preset,
                    "beta": beta,
                    "arm": arm,
                    "sessions": len(both),
                    "session_mean": float(mean_a.mean()),
                    "session_mean_vs_control": float(diff.mean()),
                    "se_session_mean_vs_control": S._se(diff.reset_index(drop=True), cats.reset_index(drop=True)),
                    **stop,
                }
            )
    return pd.DataFrame(rows)


def score_stop(sessions: pd.DataFrame) -> dict:
    """``stop_signal_4878.score`` on the arm's own Done, plus its Kaplan-Meier median stop."""
    d = sessions.assign(**{"stop[done]": sessions["done"]})
    out = S.score(d.reset_index(drop=True), "stop[done]")
    fired = pd.to_numeric(sessions["done"], errors="coerce")
    out["km_stop"] = stopping.km_median(fired.dropna().to_numpy(), np.full(int(fired.isna().sum()), HORIZON))
    return out


def over_clicks(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per preset, arm and click: the mean withheld F-beta, and the mean paired difference from the control."""
    rows = []
    for (preset, beta), g in sessions.groupby(["preset", "beta"], sort=False):
        ctl = g[g["arm"] == "control"].set_index(["category", "seed"])
        for arm in ARMS:
            a = g[g["arm"] == arm].set_index(["category", "seed"])
            both = a.index.intersection(ctl.index)
            if both.empty:
                continue
            fa, fc = _fb(a.loc[both]), _fb(ctl.loc[both])
            for c, m, d in zip(S.FIXED, fa.mean(axis=0), (fa - fc).mean(axis=0), strict=True):
                rows.append({"preset": preset, "beta": beta, "arm": arm, "click": c, "fbeta": m, "vs_control": d})
    return pd.DataFrame(rows)


def figure(curves: pd.DataFrame, ab: pd.DataFrame, out: Path) -> None:
    """Top: each arm's paired difference from the control over clicks, with its Done marked. One panel a preset."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    colors = {"control": S.INK, "level": "#2a78d6", "and": "#eb6834", "or": "#1baf7a"}
    presets = list(dict.fromkeys(curves["preset"]))
    fig, axes = plt.subplots(1, len(presets), figsize=(3.8 * len(presets), 3.6), sharey=True, facecolor=S.SURFACE)
    for ax, preset in zip(np.atleast_1d(axes), presets, strict=True):
        S._axes(ax)
        ax.grid(True, color=S.GRID, linewidth=0.6)
        ax.axhline(0, color=S.MUTED, linewidth=0.8)
        for arm in ARMS[1:]:
            g = curves[(curves["preset"] == preset) & (curves["arm"] == arm)]
            if g.empty:
                continue
            ax.plot(g["click"], g["vs_control"], color=colors[arm], linewidth=1.8, label=ARM_LABEL[arm])
            r = ab[(ab["preset"] == preset) & (ab["arm"] == arm)]
            if len(r):
                ax.axvline(float(r["mean_leave"].iloc[0]), color=colors[arm], linewidth=0.8, linestyle=":")
        r = ab[(ab["preset"] == preset) & (ab["arm"] == "control")]
        if len(r):
            ax.axvline(float(r["mean_leave"].iloc[0]), color=S.INK, linewidth=0.8, linestyle=":")
        ax.set_title(preset, color=S.INK, fontsize=10)
        ax.set_xlabel("clicks", color=S.INK)
    np.atleast_1d(axes)[0].set_ylabel("withheld F-beta, arm − today's app", color=S.INK)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=S.INK, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=130, facecolor=S.SURFACE)
    plt.close(fig)


def report(work: Path, out: Path) -> None:
    sessions = load(work)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    ab = ab_rows(sessions)
    ab.to_csv(out / "ab.csv", index=False, float_format="%.4f")
    curves = over_clicks(sessions)
    curves.to_csv(out / "over_clicks.csv", index=False, float_format="%.4f")
    figure(curves, ab, out / "figures" / "vs_control.png")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--run", action="append", required=True, help="BETA:ARM=<run dir>/results/cells")
    r.add_argument("--embedder", default="siglip")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--nproc", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "read":
        read(args.run, args.embedder, args.work, args.nproc)
    else:
        report(args.work, args.out)


if __name__ == "__main__":
    main()
