#!/usr/bin/env bash
# Refuse a commit in the two VTSearch checkouts on the GRID that nobody works in.
#
#   /exp/$USER/projects/VTSearch        the shared common checkout: every worktree
#                                        hangs off its .git, and the suite submits
#                                        the pinned scripts/slurm/suite.sbatch from it
#   /expscratch/$USER/projects/VTSearch the deploy clone ($VTS_DIR) the live app runs
#
# Work happens in a worktree, one per task; see "Worktrees on the GRID" in
# .claude/skills/grid-experiments/SKILL.md. The shared checkout had 17 files
# staged and forgotten for five days, and the deploy clone 137, before this
# existed (#4133).
#
# Installed into the shared hooks dir by scripts/grid/install-guard-hook.sh, as
# pre-commit.legacy (which pre-commit's own hook runs first) and pre-merge-commit.
# A fast-forward `git merge --ff-only origin/dev` makes no commit and still works,
# and that is the one way the shared checkout is meant to move.
#
# Override the protected list (colon-separated) with VTS_PROTECTED_CHECKOUTS.
# There is deliberately no bypass variable; `git commit --no-verify` exists, and
# using it here is the thing this hook is asking you not to do.
set -u

top="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
top="$(readlink -f "$top")"
protected="${VTS_PROTECTED_CHECKOUTS:-/exp/${USER:-$(id -un)}/projects/VTSearch:/expscratch/${USER:-$(id -un)}/projects/VTSearch}"

IFS=: read -r -a dirs <<<"$protected"
for d in "${dirs[@]}"; do
    [[ -n "$d" && -d "$d" ]] || continue
    if [[ "$top" == "$(readlink -f "$d")" ]]; then
        cat >&2 <<MSG

REFUSED: $top is not a place to commit.

  It is either the shared checkout that every worktree hangs off (its .git is
  the common dir, and the suite submits the pinned suite.sbatch from it) or the
  deploy clone the live VTSearch app runs from. Changes left here strand work
  for every other session and go stale silently.

  Do the work in a worktree of its own instead:

    srun --ntasks=1 --partition=cpu --mem=2G --time=00:15:00 bash -lc \\
      'cd /exp/\$USER/projects/VTSearch && flock "\$(git rev-parse --git-common-dir)" \\
       git worktree add -b <branch> /expscratch/\$USER/worktrees/vts-<issue> origin/dev'

  If you already staged something here, carry it over rather than lose it:
    git diff --cached --binary > /expscratch/\$USER/keep/<name>.patch

  The rule: "Worktrees on the GRID" in .claude/skills/grid-experiments/SKILL.md

MSG
        exit 1
    fi
done
exit 0
