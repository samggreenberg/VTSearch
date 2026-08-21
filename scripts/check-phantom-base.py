"""Refuse a branch whose tree is consistent with an *ancestor* of its base.

Five times now, a PR has silently reverted the PRs that merged just before it
(#2741, #2793, #2821, #3184, and the three restored in #3206). Every occurrence
had the same shape, and it is not the shape the reports assumed: the branch's
**merge-base was `origin/dev`'s tip**, zero commits behind. What was stale was
the *worktree*. HEAD moved forward onto fresh `dev` while the checkout still
held the old content, and the difference was staged wholesale — so the commit
records a deletion for every file `dev` had gained in the meantime.

That is why nothing caught it. A staleness check on the merge-base reads 0
commits behind and passes. The suite passes too, and cannot do otherwise: a
clobber removes a feature *and its tests* in one motion, so the deletions are
self-consistent. #2821's revert rode `dev` into `main` in the 2026-08-04
release before anyone noticed.

The signal this gate uses instead is the tree itself. Walk back the first-parent
chain from the merge-base; if some ancestor makes the set of deleted paths
**empty**, then the branch's content matches that ancestor rather than the base
it claims — a stale tree. Across 303 merges on `dev` this flagged 7 branches: 4
genuine clobbers (all of them) and 3 deliberate deletions.

Two findings from that sweep are load-bearing here:

- **Size proves nothing.** Two of the real clobbers deleted 2 and 3 files —
  smaller than nothing separates them from the deliberate one-file deletions.
  There is no magnitude threshold to hide behind, so the gate reports every hit.
- **The comparison must reach the worktree.** The clobber exists as uncommitted
  deletions before it is ever committed; `git status` showed 43 of them the
  whole time. Diffing `origin/dev...HEAD` alone misses it when tests run before
  the commit, so every diff below ends at the working tree, not at HEAD.

Rename detection runs at 40%, not git's default 50%: the one false positive
that was a pure rename scored 47% and would otherwise read as a deletion.

The deliberate-deletion escape hatch is `VTSEARCH_ALLOW_DELETIONS=1`. It is
expected to be needed about once per 100 PRs — `CLAUDE.md` tells you to delete
a plan file when it ships, and that is what the remaining false positives were.

Run from the repo root:
    python scripts/check-phantom-base.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# How far back along first-parent to look for the branch's real base. The
# observed clobbers sat 4-37 commits behind; 200 is far past any of them while
# keeping the walk to a few milliseconds.
MAX_WALK = 200

# Git's default rename threshold is 50%. The single pure-rename false positive
# in the historical sweep (scripts/reconcile-dev-labels.py ->
# reconcile-solved-labels.py, #3155) scored 47%, so it read as a deletion.
RENAME_THRESHOLD = "40%"

OVERRIDE_ENV = "VTSEARCH_ALLOW_DELETIONS"


def _git(*args: str) -> str:
    """Run a git command, returning stripped stdout ('' on failure)."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", *args],  # noqa: S607 - git resolved from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _deleted_vs_worktree(ref: str) -> set[str]:
    """Paths present in `ref` but gone from the working tree.

    Compares against the *worktree*, not HEAD, so an uncommitted clobber is
    caught before it is ever recorded. Renames are resolved first, so moving a
    file does not read as deleting it.
    """
    out = _git(
        "diff",
        "--diff-filter=D",
        "--name-only",
        f"--find-renames={RENAME_THRESHOLD}",
        ref,
    )
    return set(out.splitlines()) if out else set()


def _describe(ref: str) -> str:
    return _git("log", "-1", "--format=%h %cs %s", ref) or ref


def find_phantom_base(base_ref: str) -> tuple[str, set[str]] | None:
    """Return (apparent_base, deleted_paths) if the tree looks stale.

    `None` means either the branch deletes nothing, or its deletions are real:
    no ancestor of the merge-base explains them away.
    """
    merge_base = _git("merge-base", "HEAD", base_ref)
    if not merge_base:
        return None

    deleted = _deleted_vs_worktree(merge_base)
    if not deleted:
        return None

    chain = _git("rev-list", "--first-parent", f"--max-count={MAX_WALK}", merge_base).splitlines()
    # chain[0] is the merge-base itself, which we already know has deletions.
    for ancestor in chain[1:]:
        if not _deleted_vs_worktree(ancestor):
            return ancestor, deleted
    return None


def main() -> int:
    base_ref = os.environ.get("VTSEARCH_BASE_REF", "origin/dev")

    if not _git("rev-parse", "--verify", "--quiet", base_ref):
        print(f"No {base_ref} to diff against; skipping the phantom-base check.")
        return 0

    hit = find_phantom_base(base_ref)
    if hit is None:
        print("Phantom-base check: OK")
        return 0

    apparent_base, deleted = hit

    if os.environ.get(OVERRIDE_ENV):
        print(f"Phantom-base check: {len(deleted)} deletion(s) allowed via {OVERRIDE_ENV}.")
        return 0

    merge_base = _git("merge-base", "HEAD", base_ref)
    behind = _git("rev-list", "--count", f"{apparent_base}..{merge_base}")

    print(
        f"This branch deletes {len(deleted)} file(s) that it never created.\n"
        f"\n"
        f"Its tree matches an ancestor of {base_ref}, not {base_ref} itself:\n"
        f"\n"
        f"  claimed base   {_describe(merge_base)}\n"
        f"  actual content {_describe(apparent_base)}   ({behind} commits earlier)\n"
        f"\n"
        f"That is the signature of a stale checkout committed onto a fresh HEAD:\n"
        f"the work that landed on {base_ref} in between is being reverted "
        f"wholesale.\n"
        f"Nothing will fail as a result — a clobber removes a feature and its "
        f"tests\ntogether — which is why this is a gate and not a warning.\n"
        f"\n"
        f"Deleted without ever being touched by this branch:\n"
    )
    for path in sorted(deleted):
        print(f"  {path}")
    print(
        f"\n"
        f"To recover, rebuild the branch on {base_ref} and re-apply your own\n"
        f"change; do not revert the clobbering commit, which also carries work\n"
        f"you want:\n"
        f"\n"
        f"  git stash          # if the clobber is still uncommitted\n"
        f"  git fetch origin --prune && git reset --hard {base_ref}\n"
        f"\n"
        f"If every deletion above is deliberate (deleting a shipped plan file,\n"
        f"say), re-run with:\n"
        f"\n"
        f"  {OVERRIDE_ENV}=1 ./run-tests.sh\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
