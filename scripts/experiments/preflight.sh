#!/usr/bin/env bash
# Preflight for a GRID experiment launch.
#
# Every check here corresponds to a mistake that actually cost hours on a real
# study.  They are checks, not advice, because advice in a document did not stop
# any of them: the document existed and nobody read it.  Run this immediately
# before submitting arms.
#
#   bash scripts/experiments/preflight.sh --exp /exp/$USER/my-study --arms a,b,c
#
# Exits non-zero if anything is wrong.  `--warn-only` downgrades failures to
# warnings for the cases where you genuinely mean it (resuming a partial run).
#
# See scripts/experiments/GRID-PLAYBOOK.md for the SLURM-resource side of this
# (memory sizing, QOS caps, chunking) and LESSONS.md for the incident log.
set -uo pipefail

EXP=""
ARMS=""
NEED_GB="${PREFLIGHT_NEED_GB:-5}"
WARN_ONLY=0
REPO="${VTS_REPO:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXP="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --need-gb) NEED_GB="$2"; shift 2 ;;
    --warn-only) WARN_ONLY=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$EXP" ]] || { echo "usage: preflight.sh --exp DIR [--arms a,b,c] [--need-gb N]" >&2; exit 2; }

FAILED=0
say_fail() {
  if [[ "$WARN_ONLY" == "1" ]]; then echo "  WARN  $*"; else echo "  FAIL  $*"; FAILED=1; fi
}
say_ok() { echo "  ok    $*"; }

echo "preflight: $EXP"

# --- 1. One experiment, one results dir -------------------------------------
# Two grids pointed at the same CALIB_EXP once.  The resume logic read the other
# grid's cells as "this arm is already complete" and aborted a whole overnight
# batch; had it not aborted, two different grids would have been silently mixed
# in one directory and analysed as one.
if [[ -n "$ARMS" ]]; then
  for arm in ${ARMS//,/ }; do
    cells="$EXP/results-ab/$arm/cells"
    if [[ -d "$cells" ]]; then
      n=$(find "$cells" -name 'task_*.csv' ! -name '*sweep*' 2>/dev/null | wc -l)
      if [[ "$n" -gt 0 ]]; then
        say_fail "arm '$arm' already has $n cell files in $cells"
        echo "        -> a fresh study needs its own --exp dir; a resume should pass --warn-only"
      else
        say_ok "arm '$arm' results dir is empty"
      fi
    else
      say_ok "arm '$arm' results dir is new"
    fi
  done
fi

# --- 2. Free space on the REAL mount ----------------------------------------
# `df -h /exp` reported 394G free while the actual home, /exp/$USER, was its own
# 50G mount at 100%.  Cells died mid-write for hours on that misread.
# A fresh study's dir does not exist yet - that is the normal case - so stat the
# nearest existing ancestor, which is on the same filesystem.
STAT_PATH="$EXP"
while [[ ! -e "$STAT_PATH" && "$STAT_PATH" != "/" ]]; do
  STAT_PATH=$(dirname "$STAT_PATH")
done
MOUNT=$(df -P "$STAT_PATH" 2>/dev/null | awk 'NR==2 {print $6}')
AVAIL_GB=$(df -PBG "$STAT_PATH" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [[ -z "$AVAIL_GB" ]]; then
  say_fail "could not stat $STAT_PATH"
else
  if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    say_fail "only ${AVAIL_GB}G free on $MOUNT (want >= ${NEED_GB}G)"
  else
    say_ok "${AVAIL_GB}G free on $MOUNT (the mount that actually holds $EXP)"
    # The mistake this guards against: reading a *parent* mount's free space.
    # /exp showed 394G while /exp/$USER was its own 50G mount at 100%.
    parent_mount=$(df -P "$(dirname "$STAT_PATH")" 2>/dev/null | awk 'NR==2 {print $6}')
    if [[ -n "$parent_mount" && "$parent_mount" != "$MOUNT" ]]; then
      echo "        note: $(dirname "$STAT_PATH") is a DIFFERENT mount ($parent_mount) - its free space is irrelevant"
    fi
  fi
fi

# --- 3. Zero-byte cells from a previous incident ----------------------------
# A cell killed mid-write leaves a 0-byte CSV.  It counts as "present" to the
# resume logic, so it is never re-run, and it crashes or silently shrinks the
# analysis later.
if [[ -d "$EXP/results-ab" ]]; then
  z=$(find "$EXP/results-ab" -name 'task_*.csv' ! -name '*sweep*' -size 0 2>/dev/null | wc -l)
  if [[ "$z" -gt 0 ]]; then
    say_fail "$z zero-byte cell files present - delete them or they will never be re-run"
  else
    say_ok "no zero-byte cell files"
  fi
fi

# --- 4. The worktree the jobs will actually import ---------------------------
# common.setup_env() puts VTS_REPO at the front of sys.path.  Unset, jobs
# silently import a different (stale) checkout.
if [[ -z "$REPO" ]]; then
  say_fail "VTS_REPO is unset - jobs will import whatever checkout the default points at"
elif [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  say_fail "VTS_REPO=$REPO is not a git worktree"
else
  say_ok "VTS_REPO=$REPO"
  if git -C "$REPO" rev-parse --abbrev-ref HEAD >/dev/null 2>&1; then
    branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
    git -C "$REPO" fetch -q origin 2>/dev/null
    local_sha=$(git -C "$REPO" rev-parse HEAD)
    remote_sha=$(git -C "$REPO" rev-parse "origin/$branch" 2>/dev/null || echo "")
    if [[ -n "$remote_sha" && "$local_sha" != "$remote_sha" ]]; then
      say_fail "worktree is not at origin/$branch (local ${local_sha:0:8}, remote ${remote_sha:0:8})"
      echo "        -> the code you committed is not the code that will run"
    else
      say_ok "worktree matches origin/$branch"
    fi
    if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
      say_fail "worktree has uncommitted tracked changes - the run would be unreproducible"
    fi
  fi
fi

echo
if [[ "$FAILED" == "1" ]]; then
  echo "PREFLIGHT FAILED - fix the above before launching (or --warn-only if deliberate)"
  exit 1
fi
echo "preflight OK"
