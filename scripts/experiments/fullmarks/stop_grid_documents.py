"""Does the dry-run stop hold across size and prevalence? The robustness grid of #4488.

Each cell is a tier (s / m / l: 5k, 50k, 200k pages; the same positives for 27 of 36 classes, while 9
recurring logos gain positives with the tier) and a fraction of positives kept (``sota_documents.py
--thin``), with two replicates of ``--stop-log`` sessions. Per cell, for the dry run (16 clicks in a
row without a Good):

* the share of sessions that stop by click 50, and the stop click (median, range);
* click-half positives left unfound at the stop, positive-weighted and per-session mean; and,
  descriptive only, how many of those clicking on to the last click found (what the stop forgoes)
  and how many it never found;
* AP at the last click minus AP at the stop (median, and the share of stopping sessions losing > 0.05);
* the returned set's F1 at the last click minus at the stop.

Pre-registered on #4488: the dry run holds if **every cell** has positive-weighted unfound <= 5%
and at most 10% of stopping sessions losing more than 0.05 AP.

    python stop_grid_documents.py --cell s,1=<rep1>,<rep2> --cell l,1=<rep1>+<resumed part>,<rep2> ... --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import stop_rules_documents as sr

MAX_UNFOUND = 0.05
BIG_LOSS = 0.05
MAX_BIG_LOSS_SHARE = 0.10


def cell_metrics(sessions: Sequence[sr.Session]) -> dict[str, Any]:
    # (click, left, positives, ap_loss, f1_loss, found after the stop by the last click)
    stops: list[tuple[int, int, int, float, float, int]] = []
    for s in sessions:
        t = sr.signals(s, 0, 0.01)[2]
        if t is None:
            continue
        last = max(s.ap)
        found = sum(s.labels[:t])
        stops.append(
            (
                t,
                s.positives - found,
                s.positives,
                s.ap[last] - s.ap[t],
                s.f1[last] - s.f1.get(t, np.nan),
                sum(s.labels) - found,
            )
        )
    n = len(sessions)
    if not stops:
        return {"sessions": n, "stopped": 0}
    left = sum(x[1] for x in stops)
    pos = sum(x[2] for x in stops)
    losses = np.array([x[3] for x in stops])
    return {
        # Descriptive, not the bar: of the unfound, those clicking on to the last click found
        # (what the stop forgoes inside the horizon), and those still unfound there.
        "forgone_weighted": sum(x[5] for x in stops) / pos if pos else 0.0,
        "unfound_at_last_weighted": (left - sum(x[5] for x in stops)) / pos if pos else 0.0,
        "sessions": n,
        "stopped": len(stops),
        "stop_share": len(stops) / n,
        "median_click": float(np.median([x[0] for x in stops])),
        "min_click": min(x[0] for x in stops),
        "max_click": max(x[0] for x in stops),
        "unfound_weighted": left / pos if pos else 0.0,
        "unfound_mean": float(np.mean([x[1] / x[2] for x in stops if x[2]])),
        "ap_loss_median": float(np.nanmedian(losses)),
        "big_loss_share": float(np.mean(losses > BIG_LOSS)),
        "f1_loss_mean": float(np.nanmean([x[4] for x in stops])),
        "holds": bool(left / pos <= MAX_UNFOUND and np.mean(losses > BIG_LOSS) <= MAX_BIG_LOSS_SHARE) if pos else True,
    }


def load_cell(dirs: str) -> list[sr.Session]:
    """A cell's sessions: replicates separated by ``,``, the parts of one replicate joined by ``+``.

    A replicate resumed after a crash is two run directories with disjoint classes.
    """
    sessions = []
    for i, rep in enumerate(dirs.split(",")):
        for part in rep.split("+"):
            sessions += sr.load(Path(part), f"rep{i + 1}")
    keys = [s.key for s in sessions]
    if len(keys) != len(set(keys)):
        raise ValueError(f"a class appears twice in one replicate: {dirs}")
    return sessions


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cell", action="append", required=True, help="<tier>,<thin>=<rep1 dir>[+<part>],<rep2 dir>[+<part>]"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = []
    cells: list[tuple[str, float, Sequence[sr.Session]]] = []
    for spec in args.cell:
        key, dirs = spec.split("=", 1)
        tier, thin = key.split(",")
        sessions = load_cell(dirs)
        cells.append((tier, float(thin), sessions))
        rows.append({"tier": tier, "thin": float(thin), **cell_metrics(sessions)})
    lines = [
        "| tier | positives kept | sessions | stop by 50 | stop click, median (range) | unfound (weighted / mean) | of which: found by 50 / never | AP lost: median, share > 0.05 | F1 lost (mean) | holds |",
        "|---|---:|---:|---:|---|---|---|---|---:|---|",
    ]
    for r in rows:
        if not r.get("stopped"):
            lines.append(f"| {r['tier']} | {r['thin']:g} | {r['sessions']} | 0% | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['tier']} | {r['thin']:g} | {r['sessions']} | {r['stop_share']:.0%} | "
            f"{r['median_click']:.0f} ({r['min_click']}–{r['max_click']}) | "
            f"{r['unfound_weighted']:.1%} / {r['unfound_mean']:.1%} | "
            f"{r['forgone_weighted']:.1%} / {r['unfound_at_last_weighted']:.1%} | "
            f"{r['ap_loss_median']:+.3f}, {r['big_loss_share']:.0%} | {r['f1_loss_mean']:+.3f} | "
            f"{'yes' if r['holds'] else '**no**'} |"
        )
    lines += ["", *run_length_table(cells)]
    args.out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.out / "grid.md").write_text(text, encoding="utf-8")
    (args.out / "grid.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(text)
    return 0


RUN_LENGTHS = (8, 12, 16, 24, 32)


def dry_stop(labels: Sequence[bool], n: int) -> Optional[int]:
    """The first click ending *n* clicks in a row without a Good."""
    run = 0
    for t, good in enumerate(labels, start=1):
        run = 0 if good else run + 1
        if run >= n:
            return t
    return None


def run_length_table(cells: Sequence[tuple[str, float, Sequence[sr.Session]]]) -> list[str]:
    """Descriptive: other run lengths, read off the same sessions (each clicked on to the last click).

    Per cell and run length: the share of sessions stopping by the last click, their median stop
    click, and the click-half positives (weighted over all sessions) that clicking on to the last
    click found after the stop, i.e. what that run length forgoes inside the horizon. A run longer
    than the clicks left after the last Good cannot stop, so long runs stop fewer sessions.
    """
    head = (
        "| tier | positives kept | " + " | ".join(f"run {n}: stop / median click / forgone" for n in RUN_LENGTHS) + " |"
    )
    out = [head, "|---|---:|" + "---|" * len(RUN_LENGTHS)]
    for tier, thin, sessions in cells:
        pos = sum(s.positives for s in sessions)
        cols = []
        for n in RUN_LENGTHS:
            stops = [(t, sum(s.labels) - sum(s.labels[:t])) for s in sessions if (t := dry_stop(s.labels, n))]
            share = len(stops) / len(sessions) if sessions else 0.0
            med = f"{np.median([t for t, _ in stops]):.0f}" if stops else "—"
            forgone = sum(f for _, f in stops) / pos if pos else 0.0
            cols.append(f"{share:.0%} / {med} / {forgone:.1%}")
        out.append(f"| {tier} | {thin:g} | " + " | ".join(cols) + " |")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
