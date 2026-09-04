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

**It is only immune to them on the call paths it actually watches.** For its
first weeks this hook policed `mcp__github__issue_write` alone, while sessions
here filed issues with the `gh` CLI through `Bash` -- 76 `gh issue create`
commands against no MCP calls at all, of which 29 carried no `--label` and only
9 carried `experiment`. The gate was not bypassed; it was never on the road.
That is the same shape as #3440 (a hook that grepped an empty string for its
whole life): a safety net is worth nothing until something has been observed to
hit it. So the hook now reads `Bash` commands too, and `TestWiring` asserts
both registrations rather than trusting the file's existence.

The hook has a second job at the other end of an issue's life: the `solved` label
(docs/RELEASE.md step 6) means "the development is done; only merges remain",
so it must come off in the write that closes the issue -- that close *is* the
last merge landing. See `_close_problems`.

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

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_payload import read_payload, tool_arguments  # noqa: E402

TOOL_SUFFIX = "issue_write"
BASH_TOOL = "Bash"

SOLVED_LABEL = "solved"

# `gh issue create`, wherever it sits in a compound command -- these arrive as
# `... && gh issue create ...`, inside `ssh grid '...'`, and after a heredoc
# that builds the body. The lookbehind keeps `foo-gh issue create` out.
GH_ISSUE_CREATE = re.compile(r"(?<![\w./-])gh\s+issue\s+create\b")

# ...but only where it is being *run*. A command that merely mentions the
# string -- a commit message, a doc heredoc, the body of an issue about this
# very rule -- must pass straight through; blocking those is a false block on
# ordinary work, not a caught mistake. So the match must sit in command
# position: at the start, after a separator, or after a wrapper that takes a
# command as its argument. The known gap is a quoted one-liner with no
# separator inside it (`ssh grid 'gh issue create ...'`), which reads
# identically to a quoted mention; erring toward the miss there is deliberate,
# since every such form observed in practice chains through `&&` first.
GH_COMMAND_POSITION = re.compile(
    r"(?:^|[\n;|&(){}`]|\$\(|\b(?:timeout\s+[\d.]+[smhd]?|nohup|sudo|env|command|exec|time))\s*$"
)

# `--label x`, `--label=x`, `-l x`; repeated flags and comma-separated values
# both appear in the transcripts. `--add-label`/`--remove-label` deliberately do
# NOT match: those belong to `gh issue edit`, and a create must carry its labels
# at creation time, not acquire them in a follow-up nobody is around to make.
GH_LABEL_FLAG = re.compile(r"""(?<![\w-])(?:--label[=\s]+|-l\s+)(['"]?)([A-Za-z0-9_,\- ]+?)\1(?=\s|$)""")

# A `--body-file` puts the issue body outside the command, out of the
# heuristic's reach, so it is read back when the path resolves.
GH_BODY_FILE_FLAG = re.compile(r"""(?<![\w-])(?:--body-file[=\s]+|-F\s+)(['"]?)([^\s'"]+)\1""")

# Invocations that file nothing: `--help` prints usage (and is exactly what
# someone types *after* being told to add `--label`, so blocking it would make
# the denial message a dead end), and `--web` hands the whole form to a browser
# where the CLI's flags do not apply.
GH_NON_FILING = re.compile(r"(?<![\w-])(?:--help|-h|--web|-w)(?=\s|$)")

MAX_BODY_FILE_BYTES = 256 * 1024

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


def _looks_like_an_experiment(text: str) -> bool:
    """Heuristic: does closing this issue require a run, not just an edit?"""
    if any(re.search(p, text, re.IGNORECASE) for p in STRONG_SIGNALS):
        return True
    hits = sum(1 for p in WEAK_SIGNALS if re.search(p, text, re.IGNORECASE))
    return hits >= 2


def _close_problems(args: dict) -> list[str]:
    """Guard the `solved` label on a completing close (the docs/RELEASE.md step-6 sweep).

    `solved` means "the development is done; only merges remain", so it must not
    survive the close that lands the last of those merges -- a closed issue
    still carrying it asserts something false, and the views it powers
    (`is:open -label:solved`, what a human should work on next) are only
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

    if SOLVED_LABEL in labels:
        return [
            f'KEEPS `{SOLVED_LABEL}` ON A CLOSED ISSUE: `{SOLVED_LABEL}` means "solved, waiting only on merges". '
            "Closing this issue is the act of landing the last merge, so the label is now false. "
            "Drop it from the `labels` array."
        ]

    if raw is None and str(args.get("state_reason") or "").strip().lower() == "completed":
        return [
            f"CLOSE DOES NOT STRIP `{SOLVED_LABEL}`: a `completed` close ships the fix to `main`, so "
            f'`{SOLVED_LABEL}` ("solved, waiting only on merges") must come off in the same write.\n'
            "  Pass `labels` explicitly. NOTE: `labels` REPLACES the whole set, so list every label "
            f"the issue should keep (`claude`, `experiment`, ...) and simply omit `{SOLVED_LABEL}`. "
            "Read the issue first if you do not already know its labels -- passing `[]` would wipe them.\n"
            f"  If the issue never had `{SOLVED_LABEL}`, passing its existing labels unchanged satisfies this."
        ]

    return []


MISSING_CLAUDE = (
    "MISSING `claude`: you are filing this issue, so it is Claude-authored. "
    "Claude and humans file through the same GitHub account, so this label is "
    "the ONLY thing that keeps your issues out of the human-issue view "
    "(`is:issue is:open -label:claude`). It is never optional."
)

MISSING_EXPERIMENT = (
    "MISSING `experiment`: this reads like an issue nobody can close from a "
    "laptop with the test suite -- it needs measured results first. Add the "
    "label so it lands in the queue of work that needs machine time booked.\n"
    "  If closing it genuinely needs no run, say so explicitly instead of "
    "rewording the body: add `<!-- not-an-experiment: <reason> -->`."
)


def _label_problems(labels: set[str], text: str) -> list[str]:
    """The rule itself, shared by both call paths.

    `_create_problems` and `_gh_create_problems` differ only in how they dig a
    label set and some text out of their payload. Keeping the *rule* in one
    place is what stops the two paths from drifting into two rules, which is
    how the `gh` path came to be unpoliced in the first place.
    """
    found = []

    if "claude" not in labels:
        found.append(MISSING_CLAUDE)

    if "experiment" not in labels and not OPT_OUT.search(text) and _looks_like_an_experiment(text):
        found.append(MISSING_EXPERIMENT)

    return found


def _create_problems(args: dict) -> list[str]:
    labels = {str(item).strip().lower() for item in (args.get("labels") or [])}
    body = str(args.get("body") or "")
    return _label_problems(labels, f"{args.get('title') or ''}\n{body}")


def _gh_labels(command: str) -> set[str]:
    """Every label named by a `--label`/`-l` flag anywhere in the command."""
    found = set()
    for _quote, raw in GH_LABEL_FLAG.findall(command):
        for piece in raw.split(","):
            piece = piece.strip().lower()
            if piece:
                found.add(piece)
    return found


def _gh_issue_text(command: str) -> str:
    """The text the `experiment` heuristic reads for a `gh issue create`.

    The whole command is used rather than a parsed `--title`/`--body`, because
    the bodies here are routinely heredocs (`--body "$(cat <<'EOF' ... EOF)"`)
    that no tokeniser survives -- and that heredoc text genuinely *is* part of
    the command. A `--body-file` is the one shape that puts the body out of
    reach, so it is read back when the path resolves; an unresolvable one
    (`$SP/issue.md`, a path on the GRID) simply leaves the heuristic reading
    the title, which still catches the common case.
    """
    parts = [command]
    for _quote, raw_path in GH_BODY_FILE_FLAG.findall(command):
        try:
            path = Path(raw_path)
            if path.is_file() and path.stat().st_size <= MAX_BODY_FILE_BYTES:
                parts.append(path.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _runs_gh_issue_create(command: str) -> bool:
    """Is `gh issue create` actually being invoked here, or merely mentioned?"""
    if GH_NON_FILING.search(command):
        return False
    return any(GH_COMMAND_POSITION.search(command[: match.start()]) for match in GH_ISSUE_CREATE.finditer(command))


def _gh_create_problems(command: str) -> list[str]:
    """Police `gh issue create` -- the path this repo's sessions actually use."""
    if not _runs_gh_issue_create(command):
        return []
    return _label_problems(_gh_labels(command), _gh_issue_text(command))


CREATE_FOOTER = (
    "\nDecide BOTH labels now and re-issue the call, so this takes one round-trip:\n"
    "  `claude`     -> always, on every issue you file.\n"
    "  `experiment` -> only if it cannot be closed without a run (sweep, eval arm, calibration)."
)

GH_CREATE_FOOTER = (
    "\nDecide BOTH labels now and re-issue the command, so this takes one round-trip:\n"
    "  `--label claude`     -> always, on every issue you file.\n"
    "  `--label experiment` -> only if it cannot be closed without a run (sweep, eval arm, calibration).\n"
    "  `--label claude,experiment` sets both in one flag."
)

CLOSE_FOOTER = (
    "\nSee docs/RELEASE.md step 6. `solved` is a transient status, not a historical fact:\n"
    "  it goes ON when the fix PR is opened, and comes OFF in the write that closes the issue."
)


def _deny(headline: str, problems: list[str], footer: str) -> int:
    print(headline, file=sys.stderr)
    for problem in problems:
        print(f"\n  - {problem}", file=sys.stderr)
    print(footer, file=sys.stderr)
    return 2


def main() -> int:
    payload = read_payload()

    bash_args = tool_arguments(payload, BASH_TOOL, bare_keys=("command",))
    if bash_args is not None:
        found = _gh_create_problems(str(bash_args.get("command") or ""))
        if not found:
            return 0
        return _deny(
            "BLOCKED: this `gh issue create` is missing a required label (CLAUDE.md, 'Label every issue you file').",
            found,
            GH_CREATE_FOOTER,
        )

    args = tool_arguments(payload, TOOL_SUFFIX, bare_keys=("method", "repo"))
    if args is None:
        return 0

    method = str(args.get("method") or "").strip().lower()
    if method == "create":
        found, footer = _create_problems(args), CREATE_FOOTER
        headline = "BLOCKED: this issue is missing a required label (CLAUDE.md, 'Label every issue you file')."
    elif method == "update":
        found, footer = _close_problems(args), CLOSE_FOOTER
        headline = f"BLOCKED: this close mishandles the `{SOLVED_LABEL}` label."
    else:
        return 0

    if not found:
        return 0

    return _deny(headline, found, footer)


if __name__ == "__main__":
    sys.exit(main())
