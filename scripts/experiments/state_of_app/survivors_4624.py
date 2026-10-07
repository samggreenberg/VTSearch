"""Who is in the viewer's mean at each click (#4624): the survivors' line against every run.

The viewer's averaged line was, at each click, the mean over the runs with a
scored row at that click.  A spot check prompted mid-session (#4496) answers
its picks in rounds of about five, so a run inside a check had a row only
every fifth click, and Autopilot prompts the check where the labels separate
weakly, so the runs missing from any mid-session click were the weak
sessions.  On the 2026-10-05 Binary Photo review the AP line read 0.58 at
click 133 over 1,195 runs and 0.53 at click 150 over all 1,403, while the
1,195 present at both clicks rose from 0.584 to 0.589.  ``viewer.py`` now
carries each run's last scored value through those clicks (``curves.fill_gaps``).

This reads a committed page built BEFORE that carry and draws, for one metric,
the line the page drew, the mean over every trained run with its last value
carried, and the number of runs with a row underneath.  A page whose payload
says ``gaps_filled`` has no gaps left to show and is refused.  The committed
figure was drawn from the 2026-10-05 Binary Photo page as ``dev`` held it at
53f9b0a47, before the page was reskinned with ``--fill-gaps``::

    git show 53f9b0a47:docs/experiments/2026-10-05-state-of-the-app-binary-photo/viewer.html > /tmp/page.html
    python survivors_4624.py /tmp/page.html --metric average_precision --mark 133 \\
        --out docs/experiments/2026-10-05-state-of-the-app-binary-photo

Writes ``figures/viewer_survivors.png`` and ``viewer_survivors.csv`` under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import common  # noqa: E402

common.setup_env()

from viewer import _decode  # noqa: E402

FIG = "viewer_survivors.png"
CSV = "viewer_survivors.csv"


def read_page(page: Path) -> dict:
    html = page.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        raise SystemExit(f"{page}: no payload script tag - not a viewer page")
    return json.loads(m.group(1))


def survivors(payload: dict, metric: str) -> pd.DataFrame:
    """Per click: the page's line, the carried mean over every trained run, and the run count."""
    runs = payload.get("runs")
    if not runs or [int(x) for x in runs["t"]] != [int(x) for x in payload["t"]]:
        raise SystemExit("the page's per-seed lines are thinned or absent; the comparison needs every click")
    if payload.get("gaps_filled"):
        raise SystemExit(f"this page was carried ({payload['gaps_filled']}); it has no gaps left to show")
    keys = [m["key"] for m in payload["metrics"]]
    if metric not in keys:
        raise SystemExit(f"{metric!r} is not on this page; it offers {keys}")
    t = np.asarray(payload["t"], dtype=int)
    vals = _decode(runs["values"])[:, keys.index(metric), :]
    present = np.isfinite(vals)
    count = present.sum(axis=0)
    trained = present[:, len(t) - 1]
    line = np.where(count > 0, np.where(present, vals, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)
    carried = pd.DataFrame(vals[trained]).T.ffill().T.to_numpy()
    with np.errstate(invalid="ignore"):
        carried_mean = np.nanmean(carried, axis=0)
    return pd.DataFrame(
        {
            "t": t,
            "page_line": line,
            "runs_with_row": count,
            "carried_mean": carried_mean,
            "trained_runs": int(trained.sum()),
        }
    )


def mark_click(frame: pd.DataFrame, mark: int | None) -> int:
    """The click the figure annotates beside the horizon: *mark*, else the one the fewest runs are present at
    in the second half of the session."""
    t = frame["t"].to_numpy()
    if mark is not None:
        if mark not in t:
            raise SystemExit(f"--mark {mark} is not a click on this page (0..{int(t[-1])})")
        return int(np.flatnonzero(t == mark)[0])
    n = frame["runs_with_row"].to_numpy()
    half = len(n) // 2
    return int(np.argmin(n[half:]) + half)


def draw(frame: pd.DataFrame, metric: str, title: str, out: Path, i_min: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = frame["t"].to_numpy()
    page = frame["page_line"].to_numpy()
    carried = frame["carried_mean"].to_numpy()
    n = frame["runs_with_row"].to_numpy()
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax[0].plot(
        t[1:], page[1:], color="#c0392b", lw=2, label="the page's line: the runs with a scored row at that click"
    )
    ax[0].plot(
        t[1:], carried[1:], color="#2c3e50", lw=2, label="every trained run, last value carried through check rounds"
    )
    last = len(t) - 1
    for x in (t[i_min], t[last]):
        ax[0].axvline(x, color="grey", ls=":", lw=1)
        ax[1].axvline(x, color="grey", ls=":", lw=1)
    for i, dy in ((i_min, 10), (last, -16)):
        ax[0].annotate(f"{page[i]:.3f}", (t[i], page[i]), textcoords="offset points", xytext=(-46, dy), color="#c0392b")
    for i, dy in ((i_min, -16), (last, 10)):
        ax[0].annotate(
            f"{carried[i]:.3f}", (t[i], carried[i]), textcoords="offset points", xytext=(-46, dy), color="#2c3e50"
        )
    ax[0].set_ylabel(f"mean {metric}, all cells")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title(title)
    ax[1].plot(t[1:], n[1:], color="#7f8c8d", lw=2)
    ax[1].axhline(int(frame["trained_runs"].iloc[0]), color="black", lw=0.8, ls="--")
    ax[1].set_ylabel("runs with a row")
    ax[1].set_xlabel("click")
    ax[1].annotate(f"{n[i_min]} at {t[i_min]}", (t[i_min], n[i_min]), textcoords="offset points", xytext=(-74, -14))
    ax[1].annotate(f"{n[last]} at {t[last]}", (t[last], n[last]), textcoords="offset points", xytext=(-74, 6))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("page", help="a viewer.html built before #4624's carry")
    ap.add_argument("--metric", default="average_precision")
    ap.add_argument("--out", required=True, help="the report directory: figures/ and the CSV land under it")
    ap.add_argument("--title", default="The viewer's line is who is in the average, not the detectors (#4624)")
    ap.add_argument(
        "--mark",
        type=int,
        default=None,
        help="the click to annotate beside the horizon (default: where the fewest runs are present, second half)",
    )
    args = ap.parse_args(argv)
    payload = read_page(Path(args.page))
    frame = survivors(payload, args.metric)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / CSV, index=False)
    i_min = mark_click(frame, args.mark)
    draw(frame, args.metric, args.title, out / "figures" / FIG, i_min)
    at = frame.set_index("t")
    lo = int(frame["t"].iloc[i_min])
    hi = int(frame["t"].iloc[-1])
    print(
        f"{args.metric}: the page's line {at.loc[lo, 'page_line']:.3f} at {lo} ({int(at.loc[lo, 'runs_with_row'])} runs) "
        f"-> {at.loc[hi, 'page_line']:.3f} at {hi} ({int(at.loc[hi, 'runs_with_row'])}); "
        f"carried {at.loc[lo, 'carried_mean']:.3f} -> {at.loc[hi, 'carried_mean']:.3f}"
    )
    print(f"survivors_4624 -> {out / 'figures' / FIG}, {out / CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
