#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Run VTSearch's ./run-tests.sh on a cpu node.
#
#   sbatch --job-name=<name> --output=<dir>/tests-%j.out \
#       /exp/$USER/projects/VTSearch/scripts/slurm/suite.sbatch <worktree> <ref>
#
# Submit THIS copy from the dev checkout -- the "pinned checkout" decided in
# #3694 -- never the copy inside the worktree under test. Two reasons:
#
#  * policy: a branch must not be able to edit the gate that judges it, so the
#    gate is whatever `dev` says it is;
#  * mechanics: `sbatch` spools the batch script into slurmd's spool directory
#    at SUBMIT time, so the job runs that snapshot even though the checkout
#    below detaches the worktree it came from (verified on the HLTCOE GRID,
#    #3694: the running job's $0 is the spool copy, and rewriting the submitted
#    file mid-run did not change what ran).
#
# The residual risk is the dev checkout being mid-`git pull` at submit time,
# the same risk every other script run from that checkout carries.
#
# The worktree and ref are ARGUMENTS, not baked in: editing a copy of this file
# with sed is how job 507084 ended up sourcing a gridenv.sh that did not exist,
# ran with no venv, and still reported COMPLETED to SLURM.
set -uo pipefail
WT="$1"
REF="$2"

if [[ ! -f "$WT/gridenv.sh" ]]; then
    echo "FATAL: $WT is not a VTSearch worktree (no gridenv.sh)"
    exit 2
fi

cd "$WT" || exit 2
# Every worktree shares one .git, so two of these jobs fetching at once race on
# the same remote refs ("reference already exists" / "incorrect old value").
# Serialise git through a lock on the common git dir.
_gitdir="$(git rev-parse --git-common-dir)"
flock "$_gitdir" git fetch -q origin || exit 2

# Put the worktree back on whatever it was on, however this job ends.
#
# WHY: the checkout below DETACHES, and the worktree is a real one a human keeps
# working in.  Leave it detached and the next commit made there lands on no
# branch -- `git push <branch>` then reports success having pushed nothing, and
# the following suite run silently discards the commit by checking the branch
# out again.  That cost a real fix in #3292 and looked exactly like a passing
# push.  Restoring is one line and removes the whole class.
_orig="$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)"
restore() { flock "$_gitdir" git checkout -q "$_orig" 2>/dev/null; }
trap restore EXIT

# Resolve the ref before checking it out, because `git checkout --detach <name>`
# REFUSES a name that exists only on the remote: git's DWIM wants to create a
# tracking branch, `--detach` forbids that, and the job dies on
# "'--detach' cannot be used with '-b/-B/--orphan'" -- a message about nothing
# to do with the situation. That is every branch pushed from the laptop and not
# yet checked out here.
#
# Falling back to `origin/$REF` is safe in exactly this case and no other: with
# no local ref there is no local commit to discard. When a local branch DOES
# exist the guard below still refuses a mismatch rather than choosing for you.
_target="$REF"
if ! git rev-parse --verify -q "$REF^{commit}" >/dev/null; then
    if git rev-parse --verify -q "origin/$REF^{commit}" >/dev/null; then
        _target="origin/$REF"
    else
        echo "FATAL: '$REF' names no commit here and none on origin either"
        exit 2
    fi
fi
flock "$_gitdir" git checkout -q --detach "$_target" || exit 2

# The checkout takes a REF, which is normally a LOCAL branch name -- and the
# fetch above is not forced, so a branch that was squashed, amended or rebased
# and force-pushed leaves this clone's copy where it was. Both steps still exit
# 0, and the job then tests the PREVIOUS commit while reporting under the
# branch's name. On 2026-09-06 that happened twice in one afternoon and the
# stale commit failed a docs gate for a row the NEW commit adds -- which reads
# exactly like a real finding about the branch under test (#3677). The
# `=== HEAD` line below was the only thing that disclosed it.
#
# Refuse rather than auto-correct. Checking out `origin/$REF` would silently
# test OLDER code whenever this clone is legitimately AHEAD of origin, which is
# the #3292 failure -- a commit made here, never pushed, then discarded by the
# next checkout. Both directions are mistakes; only the human knows which.
_want="$(git rev-parse --verify -q "origin/$REF" || true)"
_refnote="'$REF' has no origin counterpart, so nothing to compare"
if [[ -n "$_want" ]]; then
    _have="$(git rev-parse HEAD)"
    if [[ "$_want" != "$_have" ]]; then
        echo "FATAL: checked out $_have for '$REF', but origin/$REF is $_want"
        if git merge-base --is-ancestor "$_have" "$_want" 2>/dev/null; then
            echo "  The local branch is BEHIND origin -- a force-push, or a fetch that could"
            echo "  not fast-forward it. Fetching harder is not enough: the branch is wherever"
            echo "  the worktree that owns it has its HEAD. Fix it there:"
            echo "    git -C <that worktree> fetch -f origin && git reset --hard origin/$REF"
        elif git merge-base --is-ancestor "$_want" "$_have" 2>/dev/null; then
            echo "  The local branch is AHEAD of origin. Push it first -- otherwise this job"
            echo "  tests code older than the commit you meant to test, and says nothing."
        else
            echo "  The two have DIVERGED. Reconcile them before running the suite."
        fi
        exit 2
    fi
    if [[ "$_target" == "$REF" ]]; then
        _refnote="'$REF' matches origin/$REF"
    else
        _refnote="'$REF' resolved through origin/$REF -- no local ref here"
    fi
fi

# The model cache the suite's model-loading tests read. /exp is a 50 GB mount
# that one model download can fill, so the default is on /expscratch.
export HF_HOME="${SUITE_HF_HOME:-/expscratch/$USER/cache/huggingface}"
export PATH="$HOME/.local/bin:$PATH"
source "$WT/gridenv.sh"

command -v ruff >/dev/null || { echo "FATAL: venv did not activate (no ruff on PATH)"; exit 2; }

echo "=== host $(hostname)  mem $(free -g | awk '/Mem:/{print $2}')G  node $(node --version) ==="
echo "=== worktree $WT"
echo "=== HEAD $(git log --oneline -1)"
echo "=== ref  $_refnote"
echo

./run-tests.sh
status=$?
echo "RUN_TESTS_EXIT=$status"
# Propagate it, so `sacct` State is the answer and not just a formality.
exit $status
