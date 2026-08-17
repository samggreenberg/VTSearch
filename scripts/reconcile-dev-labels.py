#!/usr/bin/env python3
"""Decide which issues should carry the `dev` label, and which should lose it.

`dev` means **fixed on `dev`, not yet on `main`** — the window between a fix
merging and the release that ships it. Because a fix PR targets `dev`, GitHub
never auto-closes its issue (that only happens on the default branch), so
without this label there is no way to tell "waiting for release" apart from
"nobody has started it". The awaiting-release view is:

    is:issue is:open label:dev

The label is **transient**: `docs/RELEASE.md` step 6 strips it in the same
write that closes the issue. So it is a precise status at all times rather
than a historical fact — a reopened issue is automatically clean, and a closed
issue is not cluttered with a label that is true of every shipped fix.

## Why this is a script and not a runbook paragraph

The rule reuses step 6's own resolution logic, which is fiddly in exactly the
ways prose hides: closing keywords must be told apart from `Refs`/`Part of`,
`Partially addressed in #M` must not read as `Addressed in #M`, and a comment
posted *after* a fix pointer may or may not dispute it. Getting any of those
subtly wrong silently corrupts the awaiting-release view. Encoding it here
makes it testable; see tests/core/test_reconcile_dev_labels.py.

## Why it takes input instead of fetching

The GitHub REST API is not reachable from a Claude session — `GITHUB_TOKEN` is
present but unauthorized (403), because the session's GitHub access is
intermediated by the MCP server. So this script is a pure function from data
to plan, and whoever *does* hold credentials supplies the data:

    # in a Claude session: gather via the github MCP tools, then
    python scripts/reconcile-dev-labels.py --input plan-input.json

    # on a machine with the gh CLI: see docs/RELEASE.md for the recipe
    gh ... | python scripts/reconcile-dev-labels.py

Input schema (unknown keys are ignored, so richer API payloads pipe in as-is):

    {
      "release_prs": [ {"number": 3128, "body": "... Closes #3077 ..."} ],
      "issues": [
        {"number": 3077, "state": "open", "labels": ["claude"],
         "comments": [ {"body": "Addressed in #3128"} ]}
      ]
    }

`release_prs` are the PRs merged into `dev` since the last release — the same
`origin/main..origin/dev` window step 6 uses. `comments` must be in chronological
order (oldest first), which is what the GitHub API returns by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

DEV_LABEL = "dev"

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


def _refs(pattern: re.Pattern[str], text: str) -> set[int]:
    return {int(m) for m in pattern.findall(text or "")}


def closing_targets(release_prs: list[dict]) -> dict[int, int]:
    """Map issue number -> PR number, for issues a release PR claims to close.

    Only closing keywords count. `Refs #N` / `Part of #N` / a bare `#N` mean
    work is still owed on that issue, so it is not resolved on `dev`.
    """
    targets: dict[int, int] = {}
    for pr in release_prs:
        for issue in _refs(CLOSING_REF, pr.get("body") or ""):
            targets.setdefault(issue, pr.get("number"))
    return targets


def _pointer_verdict(issue: dict, release_numbers: set[int]) -> tuple[str, str] | None:
    """Classify an issue's comments as a fix pointer, a disputed one, or neither.

    Returns (verdict, detail) where verdict is "resolved" or "review", or None
    when no comment points at a PR in this release at all.

    The newest comment wins. A pointer that is *not* the newest comment is
    ambiguous by construction: the later comment might be a maintainer saying
    "thanks", or it might be the reporter saying the fix does not work. Guessing
    either way is wrong -- silently tagging buries a dispute, silently skipping
    drops the issue out of the awaiting-release view with no signal -- so the
    ambiguous case is surfaced for a human instead.
    """
    comments = issue.get("comments") or []
    hits = [
        (i, pr)
        for i, comment in enumerate(comments)
        for pr in _refs(COMMENT_POINTER, comment.get("body") or "")
        if pr in release_numbers
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


def _classify(issue: dict, closers: dict[int, int], release_numbers: set[int]) -> tuple[str, str]:
    """Return (action, reason) for one issue. Action is add/remove/review/none."""
    number = issue.get("number")
    labelled = DEV_LABEL in {str(item).strip().lower() for item in (issue.get("labels") or [])}
    is_open = str(issue.get("state") or "open").lower() == "open"

    if not is_open:
        if labelled:
            return "remove", "issue is closed but still carries `dev` — the closing write did not strip it"
        return "none", "closed, no label"

    if number in closers:
        if labelled:
            return "none", f"already labelled; closed by #{closers[number]}"
        return "add", f"#{closers[number]} claims `Closes #{number}`"

    verdict = _pointer_verdict(issue, release_numbers)
    if verdict is None:
        if labelled:
            return (
                "review",
                "carries `dev` but no release PR or comment resolves it — stale, or fixed in an earlier release",
            )
        return "none", "not resolved on dev"

    kind, detail = verdict
    if kind == "review":
        return "review", detail
    return ("none", f"already labelled; {detail}") if labelled else ("add", detail)


def reconcile(data: dict) -> dict[str, list[tuple[int, str]]]:
    """Group every issue into add / remove / review / none, with a reason each."""
    release_prs = data.get("release_prs") or []
    closers = closing_targets(release_prs)
    release_numbers = {pr.get("number") for pr in release_prs}

    plan: dict[str, list[tuple[int, str]]] = {"add": [], "remove": [], "review": [], "none": []}
    for issue in data.get("issues") or []:
        action, reason = _classify(issue, closers, release_numbers)
        plan[action].append((issue.get("number"), reason))
    for bucket in plan.values():
        bucket.sort()
    return plan


def render(plan: dict[str, list[tuple[int, str]]]) -> str:
    headings = [
        ("add", f"ADD `{DEV_LABEL}`"),
        ("remove", f"REMOVE `{DEV_LABEL}`"),
        ("review", "NEEDS REVIEW (ambiguous — do not guess)"),
    ]
    lines = ["", "dev-label reconciliation", "=" * 40]
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
        print("error: input must be a JSON object with `release_prs` and `issues` keys", file=sys.stderr)
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
