"""Does the app's dry-run readout stop where the measured rule did? (#4488)

Replays each ``sota_documents.py --stop-log`` session's clicks through the app's
:func:`~vtscore.detectors.labeling_progress.dry_run_status`, one vote at a time,
and compares the first click at which it turns green with the dry run the
robustness grid measured (16 clicks in a row without a Good, as in
``stop_rules_documents.py``). The app's readout also needs one Good in the
labelset, so a session whose first 16 clicks hold no Good is reported apart.

    python dry_run_parity_documents.py <run dir> [<run dir> ...]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from vtscore.detectors.labeling_progress import DRY_RUN_TARGET, dry_run_status


def measured_stop(labels: list[bool]) -> Optional[int]:
    run = 0
    for t, good in enumerate(labels, start=1):
        run = 0 if good else run + 1
        if run >= DRY_RUN_TARGET:
            return t
    return None


def app_stop(page_ids: list[str], labels: list[bool]) -> Optional[int]:
    ids = {p: i for i, p in enumerate(page_ids)}
    history: list[tuple[int, str, float]] = []
    good: dict[int, None] = {}
    bad: dict[int, None] = {}
    for t, (p, g) in enumerate(zip(page_ids, labels), start=1):
        mid = ids[p]
        (good if g else bad)[mid] = None
        history.append((mid, "good" if g else "bad", float(t)))
        if dry_run_status(history, good, bad)["status"] == "green":
            return t
    return None


def main(runs: list[str]) -> int:
    same = differ = no_good = 0
    for run in runs:
        sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
        with (Path(run) / "clicks.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                sessions[r["class_id"]].append(r)
        for cls, rows in sorted(sessions.items()):
            rows.sort(key=lambda r: int(r["click"]))
            labels = [r["label"] == "good" for r in rows]
            m, a = measured_stop(labels), app_stop([r["page_id"] for r in rows], labels)
            if m == a:
                same += 1
            elif not any(labels[:DRY_RUN_TARGET]) and m == DRY_RUN_TARGET:
                no_good += 1
                print(f"  {run}: {cls}: no Good in the first {DRY_RUN_TARGET} clicks; measured {m}, app {a}")
            else:
                differ += 1
                print(f"  {run}: {cls}: measured {m}, app {a}")
    print(f"same stop click: {same}; no Good before the run: {no_good}; other differences: {differ}")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
