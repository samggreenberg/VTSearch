#!/usr/bin/env python3
"""Decide whether `run-tests.sh` may skip the frontend gates, or must block.

The gates (the production build, `npm audit`, the Vitest suite) all need
`frontend/node_modules`, which a fresh worktree does not have -- 354 MB is not
worth installing for a backend-only change. So the script skipped them and
carried on, and that is the bug (#4019): a frontend PR run from a fresh
worktree printed `RUN PASSED (all gates green)` from a run that never compiled
the change. PR #4018 hit exactly that; copying `node_modules` in made the same
commit exercise the build and 2,446 Vitest specs.

A skip is sound when nothing the gates can see has changed, and that is a claim
about the *diff*, not about intent -- the same reasoning the `slides` and
markdown-only fast paths already apply to themselves. So this asks the diff:

* ``run``    -- `node_modules` is present; nothing to decide, run them.
* ``skip``   -- absent, and the branch changes no frontend path. The gates
                could not have judged anything; skipping loses nothing.
* ``block``  -- absent, and either the branch changes a frontend path, or there
                is no ``origin/dev`` to diff against so the premise cannot be
                established at all. Declining to narrow when the diff is
                unreadable is what the other fast paths do.

"Frontend path" is `frontend/` and nothing else, deliberately. The bundle also
embeds `docs/user/*.md` through a symlink, but the build only *copies* those
and cannot fail on their contents -- which is why the markdown-only path
already skips these gates by design.

Prints the decision on stdout and the reasoning on stderr. Exits 0 on a
readable answer; a crash leaves stdout empty, which the caller treats as
``block``, because a gate that cannot say what it checked should not be read as
having checked everything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: Path prefixes the frontend gates can observe.
FRONTEND_PREFIXES = ("frontend/",)

RUN = "run"
SKIP = "skip"
BLOCK = "block"


def _git(repo: Path, *args: str) -> str | None:
    """Run git in *repo*; None when it fails (no repo, no such ref, …)."""
    try:
        out = subprocess.run(  # noqa: S603  # fixed argv, no shell
            ["git", *args],  # noqa: S607  # git resolves on PATH, as everywhere else in this repo
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def changed_paths(repo: Path) -> list[str] | None:
    """Everything this branch changes against origin/dev, or None if it cannot tell.

    Committed, staged, unstaged and untracked alike -- the same set
    ``_branch_changed_paths`` in run-tests.sh collects, and for the same
    reason: an untracked file would be linted by a full run, so it counts as a
    change here too.
    """
    base = _git(repo, "merge-base", "HEAD", "origin/dev")
    if base is None or not base.strip():
        return None
    ref = base.strip()
    parts: list[str] = []
    for args in (
        ("diff", "--name-only", ref, "HEAD"),
        ("diff", "--name-only", "HEAD"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        out = _git(repo, *args)
        if out is None:
            return None
        parts.extend(line for line in out.splitlines() if line.strip())
    return sorted(set(parts))


def touches_frontend(paths: list[str]) -> list[str]:
    return [p for p in paths if p.startswith(FRONTEND_PREFIXES)]


def decide(repo: Path) -> tuple[str, list[str]]:
    """Return ``(decision, explanation lines)``."""
    if (repo / "frontend" / "node_modules").is_dir():
        return RUN, []

    paths = changed_paths(repo)
    if paths is None:
        return BLOCK, [
            "frontend/node_modules is missing and there is no origin/dev to diff against,",
            "so this run cannot establish that the change is backend-only.",
        ]

    hits = touches_frontend(paths)
    if hits:
        return BLOCK, [
            "frontend/node_modules is missing, but this branch changes frontend files:",
            "",
            *(f"  {p}" for p in hits[:20]),
            *((f"  … and {len(hits) - 20} more",) if len(hits) > 20 else ()),
        ]
    return SKIP, [
        "frontend/node_modules is missing, and this branch changes no frontend/ path,",
        "so the frontend gates had nothing to judge.",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)

    decision, why = decide(args.repo)
    print(decision)
    if why:
        print("\n".join(why), file=sys.stderr)
    if decision == BLOCK:
        print(
            "\nRun the frontend gates here:  cd frontend && npm install\n"
            "or narrow the run to a backend group:  ./run-tests.sh detectors",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
