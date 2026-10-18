#!/usr/bin/env python3
"""Decide whether `run-tests.sh` may skip the frontend gates, or must block.

The gates (the production build, `npm audit`, the Vitest suite) can only judge
a change under `frontend/`, and they all need `frontend/node_modules`, which a
fresh worktree does not have -- 354 MB is not worth installing for a
backend-only change. So the script used to skip them when the directory was
missing and carry on, and that is the bug (#4019): a frontend PR run from a
fresh worktree printed `RUN PASSED (all gates green)` from a run that never
compiled the change. PR #4018 hit exactly that; copying `node_modules` in made
the same commit exercise the build and 2,446 Vitest specs.

A skip is sound when nothing the gates can see has changed, and that is a claim
about the *diff*, not about intent or about what happens to be installed -- the
same reasoning the `slides` and markdown-only fast paths already apply to
themselves. So this asks the diff:

* ``skip``   -- the branch changes no frontend path. The gates could not have
                judged anything, whether or not `node_modules` is present;
                skipping loses nothing. (With `node_modules` present this used
                to run them anyway, which spent the build and the Vitest suite
                on every backend-only run to re-check `origin/dev`'s frontend.)
* ``run``    -- the branch changes a frontend path and `node_modules` is
                present; or there is no ``origin/dev`` to diff against but
                `node_modules` is present, so running costs only time.
* ``block``  -- `node_modules` is absent, and either the branch changes a
                frontend path, or there is no ``origin/dev`` to diff against so
                the premise cannot be established at all. Declining to narrow
                when the diff is unreadable is what the other fast paths do.

"Frontend path" is `frontend/` and nothing else, deliberately. The bundle also
embeds `docs/user/*.md` through a symlink, but the build only *copies* those
and cannot fail on its contents -- which is why the markdown-only path
already skips these gates by design. `frontend/openapi.json` is inside it: a
route change regenerates that snapshot and the generated client compiled from
it, so such a branch counts as a frontend change.

The runner still builds the bundle on a ``skip`` when `static/` has none and
`node_modules` is present, so the tests that serve it do not build inside a
pytest worker; that is a convenience, not a gate, and is decided there.

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
    has_modules = (repo / "frontend" / "node_modules").is_dir()

    paths = changed_paths(repo)
    if paths is None:
        if has_modules:
            return RUN, []
        return BLOCK, [
            "frontend/node_modules is missing and there is no origin/dev to diff against,",
            "so this run cannot establish that the change is backend-only.",
        ]

    hits = touches_frontend(paths)
    if hits:
        if has_modules:
            return RUN, []
        return BLOCK, [
            "frontend/node_modules is missing, but this branch changes frontend files:",
            "",
            *(f"  {p}" for p in hits[:20]),
            *((f"  … and {len(hits) - 20} more",) if len(hits) > 20 else ()),
        ]
    return SKIP, [
        "this branch changes no frontend/ path, so the frontend build, npm audit and",
        "the Vitest suite have nothing to judge"
        + (" (frontend/node_modules is missing too)." if not has_modules else "."),
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
