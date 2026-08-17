#!/usr/bin/env python3
"""PreToolUse gate: no Claude-filed GitHub issue lands without its labels.

CLAUDE.md ("Label every issue you file") requires `claude` on every issue
Claude creates, and `experiment` on any issue that cannot be closed without a
measurement (a GRID/SLURM sweep, an eval arm, a calibration run).

That rule used to be prose only, which made it unenforceable in two distinct
ways -- and issue #3127 was filed unlabeled by hitting the first one:

1. **Staleness.** The rule reached `dev` at 2026-08-12 19:29 UTC; #3127 was
   filed at 17:17 UTC by a session whose checkout predated it by two hours.
   Prose in CLAUDE.md can only bind a session that checked it out.
2. **Attention.** CLAUDE.md is long, and a rule near the bottom competes with
   everything else in the window on a session that has been running for hours.

A hook is immune to both: it runs at tool-call time, from the checkout as of
session start, and it does not need to be remembered.

The hook has a second job at the other end of an issue's life: the `dev` label
(docs/RELEASE.md step 6) means "fixed on `dev`, NOT yet on `main`", so it must
come off in the write that closes the issue. See `_close_problems`.

Contract: read the PreToolUse payload on stdin, exit 2 to block (stderr is fed
back to Claude as the reason), exit 0 to allow. Anything unexpected -- a
payload we cannot parse, a tool we do not police -- allows, because a hook that
fails closed would wedge every unrelated GitHub call.

Escape hatch for the `experiment` heuristic (see `_looks_like_an_experiment`):
put `<!-- not-an-experiment: <reason> -->` in the issue body. It renders as
nothing on GitHub, greps cleanly, and forces the reason to be stated rather
than letting the check be dodged by rewording the body.
"""

from __future__ import annotations

import json
import os
import re
import sys

TOOL_SUFFIX = "issue_write"

DEV_LABEL = "dev"

OPT_OUT = re.compile(r"<!--\s*not-an-experiment\s*:", re.IGNORECASE)

# One of these alone means the issue cannot be closed without machine time.
# They are repo-specific enough that a false positive is a real surprise.
STRONG_SIGNALS = [
    r"vtscore\.eval",
    r"scripts/experiments",
    r"\bsbatch\b",
    r"\bSLURM\b",
    r"\bGRID\b",
    r"\bCALIB_EXP\b",
    r"\bmAP\b",
    r"\bnDCG\b",
    r"\beval arm\b",
    r"\bsweeps?\b",
    r"\bre-?measure",
    r"\bre-?run the\b",
]

# Individually weak -- "measure" shows up in plenty of pure code changes -- so
# two are required before the hook will block on them.
WEAK_SIGNALS = [
    r"\bcalibrat\w*",
    r"\bmeasur\w*",
    r"\bstud(?:y|ies)\b",
    r"\bbaselines?\b",
    r"\bbenchmarks?\b",
    r"\bablations?\b",
    r"\bexperiments?\b",
    r"\barms?\b",
    r"\brecall@",
    r"\bprecision@",
]


def _read_payload() -> dict:
    """Parse the PreToolUse payload, preferring stdin and falling back to env."""
    raw = ""
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
    if not raw:
        raw = os.environ.get("TOOL_INPUT", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_input(payload: dict) -> dict | None:
    """Return the issue_write arguments, or None if this is not our tool.

    Handles both the nested envelope (`{"tool_name", "tool_input"}`) and a
    bare arguments dict, since the two shapes have both been observed.
    """
    name = payload.get("tool_name") or payload.get("toolName") or ""
    args = payload.get("tool_input") or payload.get("toolInput")
    if isinstance(args, dict):
        return args if name.endswith(TOOL_SUFFIX) else None
    # Bare arguments: identify it by its own shape rather than by a tool name.
    if "method" in payload and "repo" in payload:
        return payload
    return None


def _looks_like_an_experiment(text: str) -> bool:
    """Heuristic: does closing this issue require a run, not just an edit?"""
    if any(re.search(p, text, re.IGNORECASE) for p in STRONG_SIGNALS):
        return True
    hits = sum(1 for p in WEAK_SIGNALS if re.search(p, text, re.IGNORECASE))
    return hits >= 2


def _close_problems(args: dict) -> list[str]:
    """Guard the `dev` label on a completing close (the docs/RELEASE.md step-6 sweep).

    `dev` means "fixed on `dev`, NOT yet on `main`", so it must not survive the
    close that ships the fix -- a closed issue still carrying it asserts
    something false, and the awaiting-release view (`is:open label:dev`) is only
    trustworthy if the strip is reliable.

    The hook sees the call's arguments, never the issue's current state, so it
    cannot check whether an issue *has* the label when `labels` is omitted.
    What it can require is that a completing close states the label set
    explicitly, which is the only form of the call that provably strips `dev`.
    """
    if str(args.get("state") or "").strip().lower() != "closed":
        return []

    raw = args.get("labels")
    labels = {str(item).strip().lower() for item in (raw or [])}

    if DEV_LABEL in labels:
        return [
            f'KEEPS `{DEV_LABEL}` ON A CLOSED ISSUE: `{DEV_LABEL}` means "on `dev`, NOT on `main`". '
            "Closing this issue is the act of shipping it to `main`, so the label is now false. "
            "Drop it from the `labels` array."
        ]

    if raw is None and str(args.get("state_reason") or "").strip().lower() == "completed":
        return [
            f"CLOSE DOES NOT STRIP `{DEV_LABEL}`: a `completed` close ships the fix to `main`, so "
            f'`{DEV_LABEL}` ("on `dev`, NOT on `main`") must come off in the same write.\n'
            "  Pass `labels` explicitly. NOTE: `labels` REPLACES the whole set, so list every label "
            f"the issue should keep (`claude`, `experiment`, ...) and simply omit `{DEV_LABEL}`. "
            "Read the issue first if you do not already know its labels -- passing `[]` would wipe them.\n"
            f"  If the issue never had `{DEV_LABEL}`, passing its existing labels unchanged satisfies this."
        ]

    return []


def _create_problems(args: dict) -> list[str]:
    labels = {str(item).strip().lower() for item in (args.get("labels") or [])}
    body = str(args.get("body") or "")
    found = []

    if "claude" not in labels:
        found.append(
            "MISSING `claude`: you are filing this issue, so it is Claude-authored. "
            "Claude and humans file through the same GitHub account, so this label is "
            "the ONLY thing that keeps your issues out of the human-issue view "
            "(`is:issue is:open -label:claude`). It is never optional."
        )

    if "experiment" not in labels and not OPT_OUT.search(body):
        if _looks_like_an_experiment(f"{args.get('title') or ''}\n{body}"):
            found.append(
                "MISSING `experiment`: this reads like an issue nobody can close from a "
                "laptop with the test suite -- it needs measured results first. Add the "
                "label so it lands in the queue of work that needs machine time booked.\n"
                "  If closing it genuinely needs no run, say so explicitly instead of "
                "rewording the body: add `<!-- not-an-experiment: <reason> -->`."
            )

    return found


CREATE_FOOTER = (
    "\nDecide BOTH labels now and re-issue the call, so this takes one round-trip:\n"
    "  `claude`     -> always, on every issue you file.\n"
    "  `experiment` -> only if it cannot be closed without a run (sweep, eval arm, calibration)."
)

CLOSE_FOOTER = (
    "\nSee docs/RELEASE.md step 6. `dev` is a transient status, not a historical fact:\n"
    "  it goes ON when the fix merges to `dev`, and comes OFF in the write that closes the issue."
)


def main() -> int:
    args = _tool_input(_read_payload())
    if args is None:
        return 0

    method = str(args.get("method") or "").strip().lower()
    if method == "create":
        found, footer = _create_problems(args), CREATE_FOOTER
        headline = "BLOCKED: this issue is missing a required label (CLAUDE.md, 'Label every issue you file')."
    elif method == "update":
        found, footer = _close_problems(args), CLOSE_FOOTER
        headline = f"BLOCKED: this close mishandles the `{DEV_LABEL}` label."
    else:
        return 0

    if not found:
        return 0

    print(headline, file=sys.stderr)
    for problem in found:
        print(f"\n  - {problem}", file=sys.stderr)
    print(footer, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
