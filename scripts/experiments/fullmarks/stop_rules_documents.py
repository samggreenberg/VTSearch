"""When should a document user stop clicking? The lights' ideas, adapted to the structural detector (#4488).

The app's Smart / Stable / Span read a page-VLAD head on document datasets, not the structural
detector. This scores the same three ideas on the structural detector and its line, from
``sota_documents.py --stop-log`` runs (``clicks.csv`` with ``score_before`` / ``line_before``, and
``stop_log.jsonl``), as pre-registered on #4488:

* **Smart** -- has clicking stopped teaching it? Each click is judged by the detector before it:
  correct when (score >= line) equals the label. Green when the last 10 clicks hold <= ``e`` errors.
* **Stable** -- are unlabelled predictions still moving? Churn of the returned set between clicks,
  ``|R_t ^ R_{t-1}| / |R_t | R_{t-1}|`` over pages unlabelled at *t*. Green when the max over the
  last 10 clicks is <= ``c``.
* **Span** -- have the clicks walked past the positives? The dry run: 16 clicks in a row without a
  Good, latched.
* **Stop** -- the first click with all three green.

``(e, c)`` is chosen by class-split CV (#4458's sha256 folds): the earliest median stop with mean
unfound click-half positives <= 5% and mean (AP at 50 - AP at the stop) <= 0.02.

    python stop_rules_documents.py --run <rep1 dir> --run <rep2 dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

WINDOW = 10
DRY_RUN = 16
E_GRID = (0, 1, 2)
C_GRID = (0.01, 0.02, 0.05, 0.10)
MAX_UNFOUND = 0.05
MAX_AP_LOSS = 0.02


def fold(cls: str) -> int:
    return hashlib.sha256(cls.encode()).digest()[0] % 2


class Session(NamedTuple):
    key: str  # class + replicate
    cls: str
    labels: list[bool]  # per click, Good?
    correct: list[bool]  # the detector before the click called it right
    churn: list[float]  # per click t >= 1: returned-set churn from v=t-1 to v=t
    positives: int  # click-half positives at the start
    ap: dict[int, float]
    f1: dict[int, float]


def load(run: Path, tag: str) -> list[Session]:
    clicks: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (run / "clicks.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            clicks[r["class_id"]].append(r)
    above: dict[tuple[str, int], set[int]] = {}
    with (run / "stop_log.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            above[(d["class_id"], int(d["v"]))] = set(d["above"])
    steps: dict[tuple[str, int], dict[str, str]] = {}
    with (run / "steps.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            steps[(r["class_id"], int(r["v"]))] = r
    num = lambda x: float(x) if x not in ("", "nan") else float("nan")  # noqa: E731
    out = []
    for cls, rows in clicks.items():
        rows.sort(key=lambda r: int(r["click"]))
        labels = [r["label"] == "good" for r in rows]
        correct = [(float(r["score_before"]) >= float(r["line_before"])) == g for r, g in zip(rows, labels)]
        churn = []
        for t, r in enumerate(rows, start=1):
            cur, prev = above.get((cls, t)), above.get((cls, t - 1))
            if cur is None or prev is None:
                churn.append(float("nan"))
                continue
            prev = prev - {int(r["page_index"])}  # the page clicked at t is labelled now
            union = cur | prev
            churn.append(len(cur ^ prev) / len(union) if union else 0.0)
        out.append(
            Session(
                key=f"{cls} ({tag})",
                cls=cls,
                labels=labels,
                correct=correct,
                churn=churn,
                positives=int(steps[(cls, 0)]["left"]),
                ap={v: num(r["ap"]) for (c, v), r in steps.items() if c == cls},
                f1={v: num(r["gate_f1"]) for (c, v), r in steps.items() if c == cls},
            )
        )
    return out


def signals(s: Session, e: int, c: float) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """First click at which Smart, Stable, Span (dry) and all three are green."""
    first: list[Optional[int]] = [None, None, None, None]
    run = 0
    dry = False
    for t in range(1, len(s.labels) + 1):
        run = 0 if s.labels[t - 1] else run + 1
        dry = dry or run >= DRY_RUN
        smart = t >= WINDOW and sum(not x for x in s.correct[t - WINDOW : t]) <= e
        window = s.churn[max(0, t - WINDOW) : t]
        stable = t >= WINDOW and all(np.isfinite(window)) and max(window) <= c
        for i, ok in enumerate((smart, stable, dry, smart and stable and dry)):
            if ok and first[i] is None:
                first[i] = t
    return first[0], first[1], first[2], first[3]


def outcome(s: Session, t: Optional[int]) -> dict[str, float]:
    last = max(s.ap)
    if t is None:
        return {
            "stopped": 0.0,
            "click": float("nan"),
            "unfound": float("nan"),
            "ap_loss": float("nan"),
            "f1_loss": float("nan"),
        }
    found = sum(s.labels[:t])
    return {
        "stopped": 1.0,
        "click": float(t),
        "unfound": (s.positives - found) / s.positives if s.positives else 0.0,
        "ap_loss": s.ap[last] - s.ap.get(t, float("nan")),
        "f1_loss": s.f1[last] - s.f1.get(t, float("nan")),
    }


def summarise(sessions: Sequence[Session], pick: Any) -> dict[str, float]:
    outs = [outcome(s, pick(s)) for s in sessions]
    stopped = [o for o in outs if o["stopped"]]
    agg = lambda k: float(np.nanmean([o[k] for o in stopped])) if stopped else float("nan")  # noqa: E731
    return {
        "sessions": float(len(outs)),
        "stopped_share": len(stopped) / len(outs) if outs else float("nan"),
        "median_click": float(np.median([o["click"] for o in stopped])) if stopped else float("nan"),
        "min_click": float(min((o["click"] for o in stopped), default=float("nan"))),
        "max_click": float(max((o["click"] for o in stopped), default=float("nan"))),
        "unfound": agg("unfound"),
        "ap_loss": agg("ap_loss"),
        "f1_loss": agg("f1_loss"),
    }


def choose(sessions: Sequence[Session]) -> tuple[int, float]:
    best: Optional[tuple[float, int, float]] = None
    for e, c in itertools.product(E_GRID, C_GRID):
        m = summarise(sessions, lambda s: signals(s, e, c)[3])
        if not (m["unfound"] <= MAX_UNFOUND and m["ap_loss"] <= MAX_AP_LOSS):
            continue
        key = (m["median_click"], e, c)
        if best is None or key < best:
            best = key
    return (best[1], best[2]) if best else (0, min(C_GRID))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sessions = [s for i, r in enumerate(args.run) for s in load(r, f"rep{i + 1}")]
    chosen = {k: choose([s for s in sessions if fold(s.cls) != k]) for k in (0, 1)}
    held = {s.key: chosen[fold(s.cls)] for s in sessions}
    lines = [f"{len(sessions)} sessions ({len({s.cls for s in sessions})} classes).", ""]
    lines += [f"- chosen on fold 1, applied to fold 0: e={chosen[0][0]}, c={chosen[0][1]}"]
    lines += [f"- chosen on fold 0, applied to fold 1: e={chosen[1][0]}, c={chosen[1][1]}", ""]
    rows = []
    for name, pick in [
        ("**stop (held-out e, c)**", lambda s: signals(s, *held[s.key])[3]),
        ("Smart alone (held-out e)", lambda s: signals(s, *held[s.key])[0]),
        ("Stable alone (held-out c)", lambda s: signals(s, *held[s.key])[1]),
        ("dry run alone (Span)", lambda s: signals(s, *held[s.key])[2]),
    ]:
        m = summarise(sessions, pick)
        rows.append({"rule": name, **m})
    lines += [
        "| rule | stops by 50 | stop click, median (range) | unfound click-half positives | AP at 50 − at stop | returned-set F1 at 50 − at stop |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['rule']} | {r['stopped_share']:.0%} | {r['median_click']:.0f} ({r['min_click']:.0f}–{r['max_click']:.0f}) | "
            f"{r['unfound']:.1%} | {r['ap_loss']:+.3f} | {r['f1_loss']:+.3f} |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.out / "stop_rules.md").write_text(text, encoding="utf-8")
    (args.out / "stop_rules.json").write_text(
        json.dumps({"chosen": {str(k): v for k, v in chosen.items()}, "rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
