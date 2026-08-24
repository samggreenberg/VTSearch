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

---------------------------------------------------------------------------
The second signal: a window of `dev` the branch does not have
---------------------------------------------------------------------------

Deletions are only the visible half of a stale tree. A clobber whose damage
sits in *hunks inside files that survive* deletes nothing, and the check above
is blind to it: #3184 reverted 332 lines of the pile builder under
`scripts/experiments/` and the EXIF handling in `vtscore/media/image/
decode.py` without removing either file, and rode in on the 21 whole-file
deletions that did trip the deletion signal. A stale tree committed during a
window where `dev` gained only in-file edits would have gone through
untouched (#3210).

Two obvious generalizations are already disproven and must not be revived:
a staleness measure on the merge-base reads *zero commits behind* in all five
occurrences (the base was fresh; the worktree was stale), and "does an older
ancestor fit the tree better?" flags 2 of the 3 branches this gate correctly
clears, at ~50x the cost.

What works is the same *exact* shape the deletion signal uses, moved from
paths to blobs. Attribute every path `dev` changed in the window to the newest
commit that changed it, and ask, per commit, what the branch holds for those
paths. A healthy branch inherits the merge-base, so it holds `dev`'s current
content — the path is *kept* — for everything it did not itself edit. A stale
tree holds the pre-merge blob instead. The clobbers show up as a contiguous
run of commits with **no kept path at all**, ending in a clean boundary:

    #3184   j=0  owned 18  kept  0  reverted 17     <- run: none of this
            j=1  owned  4  kept  0  reverted  4        landed in the branch
            j=2  owned 22  kept  0  reverted 22
            j=3  owned  7  kept  7  reverted  0     <- boundary: from here
            j=4  owned 13  kept 13  reverted  0        back, the branch is
                                                       up to date

The gate fires on a run of **two or more consecutive commits** in which the
branch keeps nothing and reverts at least one path to its pre-run blob. Two is
not a tuning knob for size — a run of one is a deliberate revert of a single
merge, which is a thing people do; a run of two is a claim about `dev`'s
history that no deliberate edit makes.

Replaying all 311 two-parent merges on `dev` through the functions below — the
sweep drives this file rather than a copy of it, via the `tip` argument — it
catches 4/4 of the clobbers (#2741, #2793, #2821, #3184) with **zero** false
positives, clearing both branches the deletion signal does flag (#2792, #2878)
and the 47% rename (#3155) that signal had to be tuned for. Dropping the run
to one commit costs 4 false positives and catches nothing more. Two variants
that look tidier both cost a catch: ignoring `dev`'s file *additions* (leaving
those to the deletion signal) loses #2793, whose window was almost all adds,
whether the additions are dropped outright or merely excluded from the run
test.

Its window is anchored anywhere in the first-parent chain rather than at the
tip, which is what reaches #2821: that branch has the merge immediately before
its base, and is missing the two before *that*.

Its escape hatch is `VTSEARCH_ALLOW_REVERTS=1`, for the deliberate revert of
two or more consecutive merges. A genuine clobber trips both signals and needs
both hatches — which is the intended friction, since agreeing that a tree is
missing 100% of a stretch of `dev` should not be a reflex.

What it does *not* see is a half-repaired tree: restore by hand the files a
clobber deleted and the commits that added them now count as kept, which
breaks the run and silences this signal while the reverted hunks remain. That
is the price of an exact test, and the reason to rebuild the branch on
`origin/dev` rather than patch a stale one back into shape.

Run from the repo root:
    python scripts/check-phantom-base.py         # the gate
    python scripts/sweep-phantom-base.py         # re-measure both signals
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

# The second signal's hatch, for a deliberate revert of consecutive merges.
REVERT_OVERRIDE_ENV = "VTSEARCH_ALLOW_REVERTS"

# How many consecutive commits the branch must be missing entirely before the
# revert signal fires. One is a deliberate revert of a single merge; two is a
# statement about `dev`'s history that no deliberate edit makes. Measured: at
# one, the sweep gains 4 false positives and no extra catch.
MIN_RUN = 2


def _git(*args: str) -> str:
    """Run a git command, returning stripped stdout ('' on failure)."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", *args],  # noqa: S607 - git resolved from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _deleted_vs_worktree(ref: str, tip: str = "") -> set[str]:
    """Paths present in `ref` but gone from the branch's content.

    `tip` empty — the only value `main()` uses — compares against the
    *worktree*, not HEAD, so an uncommitted clobber is caught before it is ever
    recorded. Naming a commit instead points the same test at that commit's
    tree, which is what the historical sweep needs to replay 300 merged
    branches through this exact code rather than a copy of it.

    Renames are resolved first, so moving a file does not read as deleting it.
    """
    out = _git(
        "diff",
        "--diff-filter=D",
        "--name-only",
        f"--find-renames={RENAME_THRESHOLD}",
        ref,
        *([tip] if tip else []),
    )
    return set(out.splitlines()) if out else set()


def _changed_vs_worktree(ref: str, tip: str = "") -> set[str]:
    """Paths whose content differs between `ref` and the branch.

    Renames are *not* resolved: this signal is about blob identity, and a
    similarity heuristic can only blur it. A path the branch renamed away reads
    as changed, which is what it is.
    """
    out = _git("diff", "--name-only", "--no-renames", ref, *([tip] if tip else []))
    return set(out.splitlines()) if out else set()


def _describe(ref: str) -> str:
    return _git("log", "-1", "--format=%h %cs %s", ref) or ref


def find_phantom_base(base_ref: str, tip: str = "") -> tuple[str, set[str]] | None:
    """Return (apparent_base, deleted_paths) if the tree looks stale.

    `None` means either the branch deletes nothing, or its deletions are real:
    no ancestor of the merge-base explains them away.
    """
    merge_base = _merge_base(base_ref, tip)
    if not merge_base:
        return None

    deleted = _deleted_vs_worktree(merge_base, tip)
    if not deleted:
        return None

    chain = _git("rev-list", "--first-parent", f"--max-count={MAX_WALK}", merge_base).splitlines()
    # chain[0] is the merge-base itself, which we already know has deletions.
    for ancestor in chain[1:]:
        if not _deleted_vs_worktree(ancestor, tip):
            return ancestor, deleted
    return None


def _merge_base(base_ref: str, tip: str = "") -> str:
    return _git("merge-base", tip or "HEAD", base_ref)


def _window_owners(merge_base: str) -> tuple[list[str], list[list[str]]]:
    """Attribute each path `dev` changed in the window to the commit that last
    changed it.

    Returns (chain, owned) where `chain[j]` is the j-th commit walking back
    first-parent from the merge-base (`chain[0]` *is* the merge-base) and
    `owned[j]` lists the paths whose newest change in the window is `chain[j]`.
    A path changed twice belongs to the newer commit only, so `owned` is a
    partition: comparing the branch against `chain[j + 1]` answers what the
    branch holds for every path in `owned[j]`.

    One `git log` call carries the whole window.
    """
    out = _git(
        "-c",
        "core.quotePath=false",
        "log",
        "--first-parent",
        f"--max-count={MAX_WALK}",
        "--diff-merges=first-parent",
        "--raw",
        "--no-renames",
        "--format=commit %H",
        merge_base,
    )
    chain: list[str] = []
    owned: list[list[str]] = []
    seen: set[str] = set()
    for line in out.splitlines():
        if line.startswith("commit "):
            chain.append(line.split()[1])
            owned.append([])
        elif line.startswith(":") and chain:
            # :mode mode preimage postimage STATUS	path
            _, _, path = line.partition("\t")
            if not path or path in seen:
                continue
            seen.add(path)
            owned[-1].append(path)
    return chain, owned


def find_reverted_window(base_ref: str, tip: str = "") -> tuple[str, str, list[str], int] | None:
    """Return (run_start, ancestor, reverted_paths, span) if the branch is
    missing a contiguous stretch of the base's history.

    `span` is how many first-parent commits that stretch covers.

    The run is a maximal span of consecutive commits — skipping any that
    changed nothing still attributable to them — for which the branch keeps
    *none* of what the base gained. `None` means no run of at least `MIN_RUN`
    such commits reverts a path to its pre-run blob, i.e. every stretch of the
    base's history is represented in this branch somewhere.
    """
    merge_base = _merge_base(base_ref, tip)
    if not merge_base:
        return None

    chain, owned = _window_owners(merge_base)
    changed = _changed_vs_worktree(merge_base, tip)
    if not changed:
        return None

    # Commits the branch has nothing of. A commit whose every path was later
    # rewritten owns nothing and neither joins a run nor breaks one.
    missing = {j for j, paths in enumerate(owned) if paths and all(p in changed for p in paths)}
    carried = {j for j, paths in enumerate(owned) if paths} - missing

    run: list[int] = []
    for j in range(len(chain)):
        if j in carried:
            hit = _check_run(chain, owned, run, tip)
            if hit:
                return hit
            run = []
        elif j in missing or run:
            # `or run`: an ownerless commit inside a run is a gap, not a break.
            run.append(j)
    return _check_run(chain, owned, run, tip)


def _check_run(
    chain: list[str], owned: list[list[str]], run: list[int], tip: str
) -> tuple[str, str, list[str], int] | None:
    """Qualify one run: long enough, and holding pre-run content."""
    run = [j for j in run if owned[j]]
    if len(run) < MIN_RUN:
        return None
    ancestor_idx = run[-1] + 1
    if ancestor_idx >= len(chain):
        # The window ran out before the run did; the walk cannot see the far
        # side of it, so it cannot say what the branch is holding instead.
        return None
    ancestor = chain[ancestor_idx]
    still_differs = _changed_vs_worktree(ancestor, tip)
    reverted = [p for j in run for p in owned[j] if p not in still_differs]
    if not reverted:
        return None
    return chain[run[0]], ancestor, sorted(reverted), ancestor_idx - run[0]


def main() -> int:
    base_ref = os.environ.get("VTSEARCH_BASE_REF", "origin/dev")

    if not _git("rev-parse", "--verify", "--quiet", base_ref):
        print(f"No {base_ref} to diff against; skipping the phantom-base check.")
        return 0

    # Both signals report, rather than the first one stopping the run: a
    # clobber usually trips both, and each names a different part of the
    # damage — which files vanished, and which commits are missing wholesale.
    failed = _report_deletions(base_ref) | _report_reverted_window(base_ref)
    if not failed:
        print("Phantom-base check: OK")
    return 1 if failed else 0


def _report_reverted_window(base_ref: str) -> bool:
    """The hunk-level signal. True if it fires and is not overridden."""
    hit = find_reverted_window(base_ref)
    if hit is None:
        return False

    _run_start, ancestor, reverted, span = hit

    if os.environ.get(REVERT_OVERRIDE_ENV):
        print(f"Phantom-base check: {span} reverted commit(s) allowed via {REVERT_OVERRIDE_ENV}.")
        return False

    merge_base = _merge_base(base_ref)
    print(
        f"This branch carries none of what {base_ref} gained across {span} "
        f"consecutive commit(s).\n"
        f"\n"
        f"  claimed base   {_describe(merge_base)}\n"
        f"  content as of  {_describe(ancestor)}\n"
        f"\n"
        f"If you meant to revert those commits, skip straight to the "
        f"override:\n"
        f"\n"
        f"  {REVERT_OVERRIDE_ENV}=1 ./run-tests.sh\n"
        f"\n"
        f"Otherwise: every path those commits touched is at its pre-merge "
        f"content here,\nand {len(reverted)} of them are byte-identical to "
        f"the version above. That is a\nstale checkout committed onto a "
        f"fresh HEAD, and it does not have to delete\nanything to do it: the "
        f"damage can sit entirely in hunks inside files that\nsurvive, which "
        f"is what the deletion check above is blind to.\n"
        f"\n"
        f"Nothing will fail as a result — a clobber removes a feature and its "
        f"tests\ntogether — which is why this is a gate and not a warning.\n"
        f"\n"
        f"Held at the pre-merge content:\n"
    )
    for path in reverted:
        print(f"  {path}")
    print(
        f"\n"
        f"If it's not deliberate, recover by rebuilding the branch on "
        f"{base_ref} and\nre-applying your own change:\n"
        f"\n"
        f"  git stash          # if the clobber is still uncommitted\n"
        f"  git fetch origin --prune && git reset --hard {base_ref}\n"
    )
    return True


def _report_deletions(base_ref: str) -> bool:
    """The whole-file signal. True if it fires and is not overridden."""
    hit = find_phantom_base(base_ref)
    if hit is None:
        return False

    apparent_base, deleted = hit

    if os.environ.get(OVERRIDE_ENV):
        print(f"Phantom-base check: {len(deleted)} deletion(s) allowed via {OVERRIDE_ENV}.")
        return False

    merge_base = _merge_base(base_ref)
    behind = _git("rev-list", "--count", f"{apparent_base}..{merge_base}")

    print(
        f"This branch deletes {len(deleted)} file(s) that it never created.\n"
        f"\n"
        f"Its tree matches an ancestor of {base_ref}, not {base_ref} itself:\n"
        f"\n"
        f"  claimed base   {_describe(merge_base)}\n"
        f"  actual content {_describe(apparent_base)}   ({behind} commits earlier)\n"
        f"\n"
        f"If every deletion below is deliberate (deleting a shipped plan "
        f"file, say),\nskip straight to the override:\n"
        f"\n"
        f"  {OVERRIDE_ENV}=1 ./run-tests.sh\n"
        f"\n"
        f"Otherwise, this is the signature of a stale checkout committed "
        f"onto a fresh\nHEAD: the work that landed on {base_ref} in between "
        f"is being reverted wholesale.\n"
        f"Nothing will fail as a result — a clobber removes a feature and its "
        f"tests\ntogether — which is why this is a gate and not a warning.\n"
        f"\n"
        f"Deleted without ever being touched by this branch:\n"
    )
    for path in sorted(deleted):
        print(f"  {path}")
    print(
        f"\n"
        f"If it's not deliberate, recover by rebuilding the branch on "
        f"{base_ref} and\nre-applying your own change; do not revert the "
        f"clobbering commit, which also\ncarries work you want:\n"
        f"\n"
        f"  git stash          # if the clobber is still uncommitted\n"
        f"  git fetch origin --prune && git reset --hard {base_ref}\n"
    )
    return True


if __name__ == "__main__":
    sys.exit(main())
