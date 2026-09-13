#!/usr/bin/env python3
"""Which correction rows predate the ruling now in force? (#3814)

A row in `corrections.json` records what a reviewer answered. Since #3814 it also
records *which question they were shown* -- the rule name on the detector they
were voting (#3612), and a digest of that rule's full ``test``. This is the other
end of that: the question "which rows were cast under a rule that has since
moved?" asked as a query instead of as an 80-image review.

The 80-image review is not hypothetical. `vase` claimed "flower pots, planters"
until #3784 withdrew the line, and nothing on the 31 `vase` rows or the 49 `bowl`
rows said which wording they were cast under -- so the only way to find out was
to put all 80 back in front of a human (#3778), of which 21 had moved and 59 had
not. Re-asking is right when the answer is unknown; what this ends is the answer
being *unknowable*.

Four states, and the distinction between the last two is the whole point:

``current``      the row's rule is the one in force, wording included
``edited``       the name matches; the ``test`` behind it has been rewritten
                 since (#3756 changed `bench`'s Bad list and kept its name)
``superseded``   the row names a rule that has been ruled away -- **this is the
                 actionable state**, and the set a recheck slate is built from
``unknown``      the row carries no rule at all

**`unknown` is not a defect and cannot be repaired here.** The 872 rows written
before the field have nothing on disk saying what wording they answered, and
back-filling them with today's rule would assert the exact thing #3814 was filed
about. They shrink as passes land -- a recheck that re-confirms a row stamps it,
which is the one honest way out (`apply_recheck.py`) -- and until then the right
reading of the number is "this much of the file is not queryable", not "this much
of the file is wrong".

Usage::

    python rule_drift.py                                  # the table
    python rule_drift.py --state superseded --ids         # the recheck slate
    python rule_drift.py --class vase --state unknown     # one class
    python rule_drift.py --json                           # for another script
    python rule_drift.py --corrections scripts/experiments/pile/human_record/PILE__corrections.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pile_config as pc
from pilebuild.corrections import (
    RULE_CURRENT,
    RULE_EDITED,
    RULE_SUPERSEDED,
    RULE_UNKNOWN,
    rule_states,
)

#: Worst first: the order a reader should spend attention in, and the column
#: order of the table. `superseded` leads because it is the only state anybody
#: can act on today.
STATES = (RULE_SUPERSEDED, RULE_EDITED, RULE_UNKNOWN, RULE_CURRENT)


def log(msg: str) -> None:
    print(f"[rule-drift] {msg}", flush=True)


def report(rows: list[dict]) -> tuple[dict[str, Counter], list[tuple[dict, str]]]:
    """``({class: Counter(state)}, [(row, state)])`` over *rows*."""
    states = rule_states(rows)
    per_class: dict[str, Counter] = defaultdict(Counter)
    for row, state in states:
        per_class[row["class"]][state] += 1
    return per_class, states


def print_table(per_class: dict[str, Counter]) -> None:
    names = {cls: pc.review_name(cls) for cls in per_class}
    wide = max([len(n) for n in names.values()] + [len("rule in force")])
    hdr = f"{'class':<14}" + "".join(f"{s:>13}" for s in STATES) + "   " + "rule in force"
    print(hdr)
    print("-" * (len(hdr) - len("rule in force") + wide))
    totals: Counter = Counter()
    for cls in sorted(per_class):
        counts = per_class[cls]
        totals.update(counts)
        # A zero prints as blank: the eye should land on the states that have
        # rows in them, and every class has zeros in most columns.
        cells = "".join(f"{counts.get(s, 0) or '':>13}" for s in STATES)
        print(f"{cls:<14}{cells}   {names[cls]}")
    print("-" * (len(hdr) - len("rule in force") + wide))
    print(f"{'ALL':<14}" + "".join(f"{totals.get(s, 0):>13}" for s in STATES))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrections", default=os.environ.get("VTS_CORRECTIONS", str(pc.PILE / "corrections.json")))
    ap.add_argument("--class", dest="klass", default="", help="one class (default: all)")
    ap.add_argument("--state", default="", help=f"one of {', '.join(STATES)} (default: all)")
    ap.add_argument("--ids", action="store_true", help="print `class image_id` lines instead of the table")
    ap.add_argument("--json", action="store_true", help="print the rows and their states as JSON")
    args = ap.parse_args()

    if args.state and args.state not in STATES:
        raise SystemExit(f"unknown --state {args.state!r}; known: {', '.join(STATES)}")

    path = Path(args.corrections)
    if not path.exists():
        raise SystemExit(
            f"no corrections file at {path} (pass --corrections, or the committed copy under human_record)"
        )
    rows = json.loads(path.read_text())
    if args.klass:
        rows = [r for r in rows if r["class"] == args.klass]
    per_class, states = report(rows)
    selected = [(r, s) for r, s in states if not args.state or s == args.state]

    if args.json:
        print(json.dumps([{**r, "rule_state": s} for r, s in selected], indent=1))
        return 0
    if args.ids:
        for r, _ in sorted(selected, key=lambda rs: (rs[0]["class"], int(rs[0]["image_id"]))):
            print(f"{r['class']} {r['image_id']}")
        return 0

    log(f"{len(rows)} rows in {path}")
    print()
    print_table(per_class)

    # Named individually, because the recheck they imply is a per-rule pass: the
    # question to re-ask is not "is this row stale" but "does this photo hold one
    # under the NEW wording", which is asked of every row a given ruling moved
    # at once (`make_class_recheck.py`).
    superseded: dict[tuple[str, str], int] = Counter()
    for row, state in states:
        if state == RULE_SUPERSEDED:
            superseded[(row["class"], row["rule"])] += 1
    if superseded:
        print(f"\n=== {sum(superseded.values())} rows answer a rule that has been ruled away ===")
        print("   Each line is a recheck slate: re-ask the class question of those images")
        print("   under the wording now in force, then `apply_recheck.py`.\n")
        print("   %-14s %-7s %-34s %s" % ("class", "rows", "voted under", "in force"))
        for (cls, was), n in sorted(superseded.items()):
            print("   %-14s %-7d %-34s %s" % (cls, n, was, pc.review_name(cls)))
        print(f"\n   ids: python {Path(__file__).name} --state superseded --ids")

    unknown = sum(c.get(RULE_UNKNOWN, 0) for c in per_class.values())
    if unknown:
        print(
            f"\n{unknown} rows record no rule. Not repairable and not a defect: the wording they were"
            f"\ncast under was never written down (#3814). They leave this column only by being"
            f"\nre-answered -- a recheck that re-confirms a row stamps it -- never by back-fill."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
