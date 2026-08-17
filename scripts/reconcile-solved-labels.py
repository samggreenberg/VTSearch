#!/usr/bin/env python3
"""Decide which issues should carry the `solved` label, and which should lose it.

`solved` means **the development on this issue is done** — the problem has been
solved, a fix PR carries it, and nothing remains but git merges (into `dev`,
and then into `main` at the next release). Its job is to keep a solved issue
out of the queue a human picks from:

    is:issue is:open -label:solved    # what someone should pick up next
    is:issue is:open label:solved     # solved; waiting only on merges

## Why the label goes on at fix-PR time, not at merge time

The label used to be called `dev` and to mean the narrower "merged into the
`dev` branch, not yet on `main`", which made its only honest trigger a merge —
an event no session observes, so in practice nothing ever applied it. The
wider meaning is both truer to what the label is *for* and self-triggering:
from a planning perspective
"solved in an open PR" and "merged to `dev`" are the same state, because
neither has any problem-solving left in it. So the fix session applies the
label when it opens the PR, in the same motion as its `Addressed in #M`
comment (see CLAUDE.md), and this script is the backstop that catches what a
session forgot and reconciles the other direction.

Widening the front edge adds a removal case the narrow meaning could not
have: a fix PR that is **closed without merging** un-solves its issue, and the
label has to come off so the issue returns to the human queue. That is why
`abandoned_prs` is part of the input.

## Why this is a script and not a runbook paragraph

The rule reuses docs/RELEASE.md step 6's own resolution logic, which is fiddly
in exactly the ways prose hides: closing keywords must be told apart from
`Refs`/`Part of`, `Partially addressed in #M` must not read as `Addressed in
#M`, a comment posted *after* a fix pointer may or may not dispute it, and a
pointer naming a commit instead of a PR cannot be resolved here at all.
Getting any of those subtly wrong silently corrupts both views above.
Encoding it here makes it testable; see tests/core/test_reconcile_solved_labels.py.

## Why it takes input instead of fetching

The GitHub REST API is not reachable from a Claude session — `GITHUB_TOKEN` is
present but unauthorized (403), because the session's GitHub access is
intermediated by the MCP server. So this script is a pure function from data
to plan, and whoever *does* hold credentials supplies the data:

    # in a Claude session: gather via the github MCP tools, then
    python scripts/reconcile-solved-labels.py --input plan-input.json

    # on a machine with the gh CLI: see docs/RELEASE.md for the recipe
    gh ... | python scripts/reconcile-solved-labels.py

Input schema (unknown keys are ignored, so richer API payloads pipe in as-is):

    {
      "release_prs":   [ {"number": 3128, "body": "... Closes #3077 ..."} ],
      "open_prs":      [ {"number": 3160, "body": "... Closes #3081 ..."} ],
      "abandoned_prs": [ {"number": 3155, "body": "... Closes #3090 ..."} ],
      "issues": [
        {"number": 3077, "state": "open", "labels": ["claude"],
         "comments": [ {"body": "Addressed in #3128"} ]}
      ]
    }

`release_prs` are the PRs merged into `dev` since the last release — the same
`origin/main..origin/dev` window step 6 uses. `open_prs` are the PRs currently
open against `dev`; `abandoned_prs` are those closed without merging. Only
`issues` and at least one PR list need be present. `comments` must be in
chronological order (oldest first), which is what the GitHub API returns by
default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

SOLVED_LABEL = "solved"

# Where each PR list sits on the live/dead axis. A "live" claim means a fix
# exists and is still on its way in; a "dead" one means it fell through.
PR_SOURCES = (("release_prs", "merged"), ("open_prs", "open"), ("abandoned_prs", "abandoned"))
DEAD_KIND = "abandoned"

# `Closes #12, closes #15` is the documented multi-issue form, so the keyword
# is matched per-reference rather than once per line.
CLOSING_REF = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[:\s]+#(\d+)", re.IGNORECASE)

# A comment pointing at the fix PR, per CLAUDE.md's "Addressed in #M" rule.
# `(?<!ly )` keeps "Partially addressed in #M" — the documented marker for
# work that is NOT finished — from reading as a resolution.
COMMENT_POINTER = re.compile(
    r"(?<!ly )\b(?:addressed|fixed|resolved|shipped|handled)\s+(?:in|by)\s+#(\d+)",
    re.IGNORECASE,
)

# The same claim, but naming a commit instead of a PR ("Fixed on `dev` by
# `de9ae81ac`"). A SHA cannot be mapped to a PR from this input, so such a
# comment is surfaced for review rather than resolved -- see `_sha_pointer`.
# The gap between the verb and the SHA absorbs interjections like "on `dev`",
# and excludes `#` so a well-formed `#M` pointer can never land here instead.
SHA_POINTER = re.compile(
    r"(?<!ly )\b(?:addressed|fixed|resolved|shipped|handled)\b[^.\n#]{0,40}?"
    r"\b(?:in|by)\s+(?:commit\s+)?`?([0-9a-f]{7,40})`?\b",
    re.IGNORECASE,
)


def _refs(pattern: re.Pattern[str], text: str) -> set[int]:
    return {int(m) for m in pattern.findall(text or "")}


def pull_requests(data: dict) -> list[tuple[int, str, str]]:
    """Flatten the PR lists into (number, body, kind) triples."""
    return [(pr.get("number"), pr.get("body") or "", kind) for key, kind in PR_SOURCES for pr in (data.get(key) or [])]


def closing_targets(prs: list[tuple[int, str, str]]) -> dict[int, tuple[int, str]]:
    """Map issue number -> (PR number, kind), for issues a PR claims to close.

    Only closing keywords count. `Refs #N` / `Part of #N` / a bare `#N` mean
    work is still owed on that issue, so its development is not done.
    """
    targets: dict[int, tuple[int, str]] = {}
    for number, body, kind in prs:
        for issue in _refs(CLOSING_REF, body):
            targets.setdefault(issue, (number, kind))
    return targets


def _pointer_verdict(issue: dict, live_numbers: set[int]) -> tuple[str, str] | None:
    """Classify an issue's comments as a fix pointer, a disputed one, or neither.

    Returns (verdict, detail) where verdict is "resolved" or "review", or None
    when no comment points at a live PR at all.

    The newest comment wins. A pointer that is *not* the newest comment is
    ambiguous by construction: the later comment might be a maintainer saying
    "thanks", or it might be the reporter saying the fix does not work. Guessing
    either way is wrong -- silently tagging buries a dispute (and hides an issue
    that still needs solving from the human queue), while silently skipping
    leaves solved work in that queue -- so the ambiguous case is surfaced for a
    human instead.
    """
    comments = issue.get("comments") or []
    hits = [
        (i, pr)
        for i, comment in enumerate(comments)
        for pr in _refs(COMMENT_POINTER, comment.get("body") or "")
        if pr in live_numbers
    ]
    if not hits:
        return None

    last_index, pr_number = hits[-1]
    trailing = comments[last_index + 1 :]
    if not trailing:
        return "resolved", f"newest comment points at #{pr_number}"

    return "review", (
        f"pointer at #{pr_number} is followed by {len(trailing)} later comment(s); check whether they dispute the fix"
    )


def _dead_pointer(issue: dict, dead_numbers: set[int]) -> int | None:
    """Return a PR that a comment claims fixed this issue but which never merged."""
    for comment in issue.get("comments") or []:
        for pr in _refs(COMMENT_POINTER, comment.get("body") or ""):
            if pr in dead_numbers:
                return pr
    return None


def _sha_pointer(issue: dict) -> str | None:
    """Return the commit named by the newest SHA-form fix claim, if any.

    CLAUDE.md prescribes `Addressed in #M` -- a *PR number* -- precisely because
    a bare commit SHA cannot be mapped back to a PR from this input. But a
    comment saying "Fixed on `dev` by `de9ae81ac`" plainly claims a fix, and
    reporting it as unsolved is the script guessing, in the one direction it is
    meant never to guess: silently. #2911 sat open through a release exactly
    this way -- its fix was on `main` while no view showed it anywhere. So the
    claim is surfaced and a human maps the commit to its PR.
    """
    for comment in reversed(issue.get("comments") or []):
        found = SHA_POINTER.findall(comment.get("body") or "")
        if found:
            return found[-1]
    return None


def _describe(pr_number: int, kind: str, issue_number: int) -> str:
    article = "open PR" if kind == "open" else "merged PR"
    return f"{article} #{pr_number} claims `Closes #{issue_number}`"


def _classify(
    issue: dict,
    closers: dict[int, tuple[int, str]],
    live_numbers: set[int],
    dead_numbers: set[int],
) -> tuple[str, str]:
    """Return (action, reason) for one issue. Action is add/remove/review/none."""
    number = issue.get("number")
    labelled = SOLVED_LABEL in {str(item).strip().lower() for item in (issue.get("labels") or [])}
    is_open = str(issue.get("state") or "open").lower() == "open"

    if not is_open:
        if labelled:
            return "remove", "issue is closed but still carries `solved` — the closing write did not strip it"
        return "none", "closed, no label"

    claim = closers.get(number)
    if claim and claim[1] != DEAD_KIND:
        detail = _describe(claim[0], claim[1], number)
        return ("none", f"already labelled; {detail}") if labelled else ("add", detail)

    verdict = _pointer_verdict(issue, live_numbers)
    if verdict is None:
        # Nothing live claims this issue. Did something dead claim it?
        dead_pr = claim[0] if claim else _dead_pointer(issue, dead_numbers)
        if dead_pr is not None:
            if labelled:
                return "remove", (
                    f"its fix PR #{dead_pr} was closed without merging — the development is owed again, "
                    "so the issue belongs back in the human queue"
                )
            return "none", f"claimed only by #{dead_pr}, which was closed without merging"

        sha = _sha_pointer(issue)
        if sha:
            return "review", (
                f"a comment claims a fix by commit `{sha}` rather than `Addressed in #M`; "
                "map the commit to its PR by hand, then re-run"
            )
        if labelled:
            return (
                "review",
                "carries `solved` but no PR or comment resolves it — stale, or fixed in an earlier release",
            )
        return "none", "no fix PR claims it"

    kind, detail = verdict
    if kind == "review":
        return "review", detail
    return ("none", f"already labelled; {detail}") if labelled else ("add", detail)


def reconcile(data: dict) -> dict[str, list[tuple[int, str]]]:
    """Group every issue into add / remove / review / none, with a reason each."""
    prs = pull_requests(data)
    closers = closing_targets(prs)
    live_numbers = {number for number, _, kind in prs if kind != DEAD_KIND}
    dead_numbers = {number for number, _, kind in prs if kind == DEAD_KIND}

    plan: dict[str, list[tuple[int, str]]] = {"add": [], "remove": [], "review": [], "none": []}
    for issue in data.get("issues") or []:
        action, reason = _classify(issue, closers, live_numbers, dead_numbers)
        plan[action].append((issue.get("number"), reason))
    for bucket in plan.values():
        bucket.sort()
    return plan


def render(plan: dict[str, list[tuple[int, str]]]) -> str:
    headings = [
        ("add", f"ADD `{SOLVED_LABEL}`"),
        ("remove", f"REMOVE `{SOLVED_LABEL}`"),
        ("review", "NEEDS REVIEW (ambiguous — do not guess)"),
    ]
    lines = ["", "solved-label reconciliation", "=" * 40]
    for key, heading in headings:
        entries = plan[key]
        lines.append(f"\n{heading} ({len(entries)})")
        if not entries:
            lines.append("  (none)")
        for number, reason in entries:
            lines.append(f"  #{number}: {reason}")
    lines.append(f"\nno change: {len(plan['none'])} issue(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="JSON file to read (default: stdin)")
    parser.add_argument("--json", action="store_true", help="emit the plan as JSON instead of text")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any issue needs a label change or review (for use as a gate)",
    )
    args = parser.parse_args(argv)

    raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: input is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("error: input must be a JSON object with PR list(s) and an `issues` key", file=sys.stderr)
        return 2

    plan = reconcile(data)
    if args.json:
        print(json.dumps({k: [{"number": n, "reason": r} for n, r in v] for k, v in plan.items()}, indent=2))
    else:
        print(render(plan))

    if args.check and (plan["add"] or plan["remove"] or plan["review"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
