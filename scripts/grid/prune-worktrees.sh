#!/usr/bin/env bash
# Remove VTSearch worktrees on the GRID whose work is finished. DRY RUN unless --apply.
#
#   srun --ntasks=1 --partition=cpu --mem=2G --time=01:00:00 \
#       bash scripts/grid/prune-worktrees.sh [--apply] [--keep NAME_OR_GLOB ...] [--keep-file FILE]
#
# Run it through srun: it walks every worktree (du, find) and `git worktree
# remove` deletes many files, which is login-node load (see "Worktrees on the
# GRID" in .claude/skills/grid-experiments/SKILL.md). --apply refuses on a login node.
#
# A worktree is REMOVED only when ALL of these hold:
#   * it is not the shared checkout ($VTS_SHARED_CHECKOUT) or the deploy clone ($VTS_DIR);
#   * no tracked file is modified or staged, and no untracked, non-ignored file exists;
#   * no file in it (outside .git) was modified in the last $PRUNE_RECENT_HOURS h (24);
#   * it is not locked, not named by --keep/--keep-file, and no script on origin/dev
#     names its path (a launcher's default VTS_REPO, a peer's hard-coded path);
#   * no queued or running job of $USER names it (WorkDir, command, stdout, or a
#     job name carrying its issue number);
#   * AND EITHER it is on a branch whose tip is in origin/dev, whose origin copy (if
#     any) is also in origin/dev, and which has no open PR;
#     OR it is a detached worktree named *-tests (a suite checkout) with no job.
# Everything else is listed as KEEP with the reason. Detached worktrees not named
# *-tests are always kept: nothing says whose they are.
#
# It never deletes a branch (merged or not) and never uses `git worktree remove
# --force`: a worktree git itself considers dirty stays. /expscratch has no
# snapshots, so a delete is final.
set -uo pipefail

APPLY=0
KEEPS=(vts-annq-3720)   # the vg_scale peer's bank_verdicts.py / retire_finished.py hard-code this path
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1 ;;
        --keep) KEEPS+=("$2"); shift ;;
        --keep-file) while IFS= read -r l; do [[ -n "$l" && "$l" != \#* ]] && KEEPS+=("$l"); done <"$2"; shift ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

SHARED="$(readlink -f "${VTS_SHARED_CHECKOUT:-/exp/$USER/projects/VTSearch}")"
DEPLOY="$(readlink -f "${VTS_DIR:-/expscratch/$USER/projects/VTSearch}")"
RECENT_MIN=$(( ${PRUNE_RECENT_HOURS:-24} * 60 ))

if [[ $APPLY == 1 && "$(hostname -s)" == login* ]]; then
    echo "REFUSED: --apply on login node $(hostname -s). Run it through srun --ntasks=1." >&2
    exit 2
fi

GITDIR="$(git -C "$SHARED" rev-parse --path-format=absolute --git-common-dir)" || exit 2
flock "$GITDIR" git -C "$SHARED" fetch -q --prune origin || { echo "fetch failed" >&2; exit 2; }

# Branches with an open PR. If gh fails, refuse to judge any branch as finished.
if ! OPEN_PRS="$(gh pr list -R samggreenberg/VTSearch --state open --limit 300 \
        --json headRefName --jq '.[].headRefName' 2>/dev/null)"; then
    echo "WARNING: gh pr list failed; every branch worktree will be kept" >&2
    OPEN_PRS="__gh_failed__"
fi

# Everything the queue says about $USER's live jobs, one blob to grep.
JOBINFO="$(squeue -u "$USER" -h -t PENDING,RUNNING,CONFIGURING,COMPLETING -o '%i' 2>/dev/null \
    | sed 's/_.*//' | sort -u | while read -r j; do scontrol show job -o "$j" 2>/dev/null; done)"
JOBNAMES="$(squeue -u "$USER" -h -t PENDING,RUNNING,CONFIGURING,COMPLETING -o '%j' 2>/dev/null | sort -u)"

job_mentions() {  # <path> <name>: is this worktree in any live job?
    local path="$1" name="$2" key
    grep -qF -- "$path" <<<"$JOBINFO" && return 0
    # suite jobs are submitted FROM the shared checkout, so their WorkDir says
    # nothing; the job name carries the issue number (suite-3840 <-> vts-3840-tests).
    key="${name#vts-}"; key="${key%-tests}"
    for n in $(grep -oE '[0-9]{3,5}' <<<"$key"); do
        grep -qE "(^|[^0-9])$n([^0-9]|$)" <<<"$JOBNAMES" && return 0
    done
    return 1
}

removed=0; kept=0; freed_kb=0
keep() { printf 'KEEP    %-28s %s\n' "$1" "$2"; kept=$((kept + 1)); }

while IFS=$'\t' read -r wt br locked prunable; do
    name="$(basename "$wt")"
    real="$(readlink -f "$wt" 2>/dev/null || echo "$wt")"
    [[ "$real" == "$SHARED" ]] && { keep "$name" "the shared checkout (common .git)"; continue; }
    [[ "$real" == "$DEPLOY" ]] && { keep "$name" "the deploy clone"; continue; }
    [[ -n "$prunable" ]] && { keep "$name" "directory is gone; 'git worktree prune' drops the record"; continue; }
    [[ -n "$locked" ]] && { keep "$name" "locked"; continue; }
    matched=""
    for k in "${KEEPS[@]}"; do
        # shellcheck disable=SC2053  # glob match is intended
        [[ "$name" == $k ]] && { matched="$k"; break; }
    done
    [[ -n "$matched" ]] && { keep "$name" "--keep $matched"; continue; }
    # A tool whose DEFAULT reads this path breaks silently when it goes (vts-calib
    # is common.py's default VTS_REPO); a comment naming it also keeps it, on purpose.
    if git -C "$SHARED" grep -qE "(projects|worktrees)/$name([^A-Za-z0-9_-]|\$)" origin/dev -- scripts gridenv.sh 2>/dev/null; then
        keep "$name" "a script on origin/dev names its path"; continue
    fi
    if job_mentions "$real" "$name"; then keep "$name" "a queued/running job uses it"; continue; fi

    dirty="$(git -C "$real" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
    [[ "$dirty" -gt 0 ]] && { keep "$name" "DIRTY: $dirty modified/staged tracked file(s)"; continue; }
    untracked="$(git -C "$real" ls-files --others --exclude-standard 2>/dev/null | wc -l)"
    [[ "$untracked" -gt 0 ]] && { keep "$name" "$untracked untracked file(s); move them out first"; continue; }
    if [[ -n "$(find "$real" -path "$real/.git" -prune -o -type f -mmin "-$RECENT_MIN" -print -quit 2>/dev/null)" ]]; then
        keep "$name" "touched in the last $((RECENT_MIN / 60)) h"; continue
    fi

    if [[ -z "$br" ]]; then
        [[ "$name" == *-tests ]] || { keep "$name" "detached and not a *-tests worktree"; continue; }
        why="detached suite worktree, no job"
    else
        b="${br#refs/heads/}"
        grep -qxF -- "$b" <<<"$OPEN_PRS" && { keep "$name" "branch $b has an open PR"; continue; }
        [[ "$OPEN_PRS" == "__gh_failed__" ]] && { keep "$name" "cannot check open PRs"; continue; }
        git -C "$SHARED" merge-base --is-ancestor "$br" origin/dev \
            || { keep "$name" "branch $b is NOT merged into origin/dev"; continue; }
        if git -C "$SHARED" rev-parse -q --verify "refs/remotes/origin/$b" >/dev/null \
            && ! git -C "$SHARED" merge-base --is-ancestor "origin/$b" origin/dev; then
            keep "$name" "origin/$b has commits not in origin/dev"; continue
        fi
        why="branch $b merged into origin/dev"
    fi

    kb="$(du -sk "$real" 2>/dev/null | cut -f1)"; kb="${kb:-0}"
    if [[ $APPLY == 1 ]]; then
        if flock "$GITDIR" git -C "$SHARED" worktree remove "$real"; then
            printf 'REMOVED %-28s %s (%s MB)\n' "$name" "$why" "$((kb / 1024))"
            removed=$((removed + 1)); freed_kb=$((freed_kb + kb))
        else
            keep "$name" "git worktree remove refused"
        fi
    else
        printf 'WOULD   %-28s %s (%s MB)\n' "$name" "$why" "$((kb / 1024))"
        removed=$((removed + 1)); freed_kb=$((freed_kb + kb))
    fi
done < <(git -C "$SHARED" worktree list --porcelain | awk '
    /^worktree /{wt=substr($0,10); br=""; lk=""; pr=""}
    /^branch /{br=$2}
    /^locked/{lk="1"}
    /^prunable/{pr="1"}
    /^$/{print wt "\t" br "\t" lk "\t" pr}
    END{if (wt != "") print wt "\t" br "\t" lk "\t" pr}' | awk -F'\t' '!seen[$1]++')

verb="would remove"; [[ $APPLY == 1 ]] && verb="removed"
echo "--- $verb $removed, kept $kept, $((freed_kb / 1024)) MB $( [[ $APPLY == 1 ]] && echo freed || echo to free )"
[[ $APPLY == 1 ]] && flock "$GITDIR" git -C "$SHARED" worktree prune
exit 0
