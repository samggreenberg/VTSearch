#!/usr/bin/env bash
# #3314: does more cross-calibration ever pay, and what does it cost?
#
#   bash launch_folds_3314.sh prepare      # stage 0, ONCE, shared by every stage
#   bash launch_folds_3314.sh size [cell]  # time ONE cell before committing
#   bash launch_folds_3314.sh baseline     # the click-0 text-sort anchor
#   bash launch_folds_3314.sh screen       # STAGE A: the nested fold-count screen
#   bash launch_folds_3314.sh ab K_BEST SCHEDULE   # STAGE B, only if the gate passes
#
# Design + pre-registered decision rules:
#   docs/experiments/calibration-fold-count-3310/PLAN.md
#
# WHY STAGE A IS ONE RUN AND NOT SIX ARMS.  The folds are NESTED:
# `compute_fold_orderings` draws each fold as an independent stratified split off
# one RandomState(42) stream at a per-fold size that does not depend on the
# count, so the K folds a live `calibrate_count=K` run would train are
# byte-for-byte the first K of the Kmax folds trained here.  One run at Kmax=8
# therefore measures every K's threshold AND every K's wall clock, paired within
# the step.  The screen is exact, not an approximation.
#
# WHY THAT IS STILL NOT THE VERDICT.  The shipped threshold DRIVES ACQUISITION
# (#3115 measured 100% `fold_anchored` provenance on every `app_trained` step),
# so a screen that holds the trajectory fixed cannot see the votes a different K
# would have collected.  That is stage B, and it is GATED: it is booked only if
# the screen shows some K clearing MARGIN=0.005 in some band inside the 1.5x
# banded wall-clock ceiling.  A flat screen is a result, and the report ships
# without stage B.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-screen}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-folds-3314}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# One study, one base; one STAGE, one CALIB_EXP.  Stage B's arms live at a
# different `calibrate_count`, so they are different trajectories and must not
# share a results dir with the screen (preflight check 1).
# /expscratch, not /exp: the screen emits ~6 fold counts x ~8 arms per step per
# cell, which runs to a few GB of cells, and /exp/$USER is a 50G mount.
BASE="${FOLDS3314_BASE:-/expscratch/$USER/folds-3314}"

# --- the screen grid ---------------------------------------------------------
# Kmax=8 caps the extra per-step cost at ~4x production's fold fits.  #2897
# already showed K >= 12 is unaffordable on the interactive path (`cal_share`
# 0.885 at 16), and the laptop bench (#3310) puts the anchored rule's saturation
# at K ~ 6-8, so the grid spends its budget below the knee rather than past it.
export CALIB_FOLD_COUNTS="${CALIB_FOLD_COUNTS:-1,2,3,4,6,8}"
# The run LIVES at production's 2, so the trajectory is the one users get and
# every counterfactual K is scored against the same votes.
export CALIB_CALIBRATE_COUNT="${CALIB_CALIBRATE_COUNT:-2}"
# The pre-registered bands (the #3287 set).  `analyze_folds_3314.py` owns them,
# but `analyze_folds_2897.py` reads the same rows from the same env var, so
# pinning it here keeps a hand-run of the older analyzer on the same bands.
export CALIB_FOLD_CHECKPOINTS="${CALIB_FOLD_CHECKPOINTS:-25,60,100,150}"

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path.  The anchored arms need the per-fold haystacks
# that `_safe_threshold_for_step` supplies, and `docs/ML.md` is explicit that
# fusion is not a setting the app has.
export CALIB_SAFE_THRESHOLDS=1
# CALIB_HEAD deliberately UNSET -> `PRODUCTION_HEAD`, the linear SVM a live
# detector has trained since PR #3198.  CALIB_BLEND_SCHEDULE unset -> the app's
# per-mode default (#2841).  CALIB_CALIBRATION_FRACTION unset -> the per-space
# default `production_split_for` resolves (#3287/#3290, PR #3289).  Each of
# those is preflight check 12's business, and each is production here.
export CALIB_ANALYZE="${CALIB_ANALYZE:-analyze_folds_3314.py}"

# --- environment -------------------------------------------------------------
# ONE dataset, `vg_scale_any` (#3156): 12 hand-checked classes, 300 positives
# each against one shared negative pool, so prevalence is IDENTICAL in every
# cell.  A threshold IS a quantile of the calibration set and this study sweeps
# how many such sets to draw, so a grid whose cells differ 60-fold in positives
# would confound the swept axis with itself.
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
# The region half is the PAIR `siglip+dinov3_patch` (#3278/#3276): DINOv3 has no
# text tower, so a bare `dinov3_patch` arm would open on three random
# known-goods while every SigLIP arm opens on a typed query - a SEEDING contrast
# hidden inside the voting-mode contrast.  This study reads the COLD START,
# where the opening decides the whole calibration set, so that would be fatal.
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
# BOTH styles on the patch embedder, so voting mode is separable from embedder
# (#3115's confound-breaking corner, #3258's confound, fixed as in #3287):
#
#   siglip/whole  vs  dinov3/whole      -> the EMBEDDER, at fixed voting mode
#   dinov3/whole  vs  dinov3/max_patch  -> the VOTING MODE, at fixed embedder
#
# The bench (#3310) predicts a NULL on the single-vector geometries and puts the
# live question on region voting, where per-fold variance is much larger.  The
# middle corner is what stops "region voting is different" from being "DINOv3 is
# different" wearing a disguise.
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""

# --- sizing ------------------------------------------------------------------
# 150 steps because the benefit is predicted to DECAY inside the horizon: the
# folds that help most (few votes) are also the cheapest to fit, and the decay
# is the thing the adaptive schedule exists to exploit.  A pooled winner over a
# decay is precisely the number that hides it.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# Seed-major, not the `category` default: a truncated array then loses its last
# SEEDS uniformly across every environment (a power problem a report can state)
# rather than its last CATEGORIES entirely (a design failure).  #3287's argument.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
# 4 seeds is the minimum that makes `sd(threshold)` - the variance-reduction
# mechanism, observed directly - computable at all: it is taken ACROSS seeds at
# a fixed step.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

# No GPU work: the cells train a small head on cached pile embeddings.
# `launch_cells.sh` defaults to `--partition=gpu --gres=gpu:v100:1`, where the
# `4gpu_tier` QOS caps the array at 2 concurrent tasks.  BOTH must be named -
# the flag is dropped rather than passed as `--gres=none`, which this cluster's
# submit filter rewrites and then rejects (#2897 lost both A/B arms to exactly
# that, with an empty job id as the only symptom).
export CALIB_PARTITION=cpu
export CALIB_GRES=none

# MEASURED on THIS grid by `size` before `screen` went in; see the PLAN's sizing
# note.  No prior grid's seconds transfer: the screen trains 8 folds per step
# (~4x production's fold fits) and re-cuts them under ~8 arms per K, which is a
# different cell from #3287's (no fold grid at all) and from #3115's (K<=16 and
# the anchored EM sweep, so strictly more).  Fill these in from `size`.
export CALIB_MEM="${CALIB_MEM:-12G}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
# ONE array of 96 cells, so the whole study's footprint is this number.
# 1074G is the per-user allowance under QOS `cpu_limit`; %72 x 12G = 864G is
# 80% of it, under preflight check 8's 90% line, and leaves room for the
# analyze step.  Above 96 the extra width buys nothing - there is no 97th cell.
export CALIB_CONC="${CALIB_CONC:-72}"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-64G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-3:00:00}"

# Pin the BLAS pools to one thread each: 72 concurrent cells each spawning a
# node-sized pool oversubscribes whatever node they land on.  Exported HERE, not
# inside a mode, so `size` and `screen` measure and run under the SAME
# environment - a cell timed with different threading than the array will use is
# a guess with a unit attached
# (`lessons/2026-08-24-a-login-node-timing-nearly-cut-an-arm.md`).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

# Stage 0 is shared by every stage and must be, not merely may be.  `prepare`
# selects the categories and builds the exemplar crops, and neither depends on
# the fold count - but `run_cells.py --print-cells` ENUMERATES from
# `prepare_info.json`, so independent prepares would be independent chances for
# array index 37 to mean a different cell in different stages.
PREP="$BASE/prepare/results"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

link_prepare() {
  local rd="$1"
  mkdir -p "$rd/cells"
  [[ -e "$rd/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$rd/prepare_info.json"
  [[ -e "$rd/crops" ]] || ln -s "$PREP/crops" "$rd/crops"
}

# Preflight's checks are mostly PYTHON: it imports vtscore to assert the
# region-voting premise and to compare every pinned knob against its shipped
# constant.  A non-interactive login shell has no venv, where the system python
# is old enough that `X | None` raises at import time - so those checks come
# back FAIL for a reason unrelated to the run.  Loud, and therefore survivable;
# the dangerous version is preflight reporting `ok` without having looked
# (#2905).  Activate the venv first.
activate_venv() {
  # shellcheck disable=SC1091
  source "$WT/gridenv.sh" >/dev/null 2>&1 || {
    echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
  }
}

run_preflight() {
  # $1 = job name, $2... = extra flags
  local job="$1"; shift
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
    --require-min-positives 100 \
    --reuse-prepare "$PREP" \
    --require-region-voting vg_scale_any:siglip+dinov3_patch \
    --contrasts-voting-modes --patch \
    "$@" \
    --job-name "$job" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED for $job" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

case "$MODE" in
  prepare)
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=fc3314-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    echo "cells enumerate from $PREP/prepare_info.json"
    ;;

  size)
    # Time ONE cell before committing to a 96-cell array.  A cell runs EVERY
    # style of its embedder in one task (`array_cells`: they share the loaded
    # pickle), so the grid is 12 categories x 2 embedders x 4 seeds = 96, and
    # the `siglip+dinov3_patch` cell carries TWO of the three geometries -
    # `whole_image` and `max_patch` - which is what makes it the critical path.
    # Under `CALIB_CELL_ORDER=seed` the first seed's 24 environments come first:
    # cells 0-11 are `siglip` (binary, cheap), cells 12-23 the pair.  Size from
    # BOTH - sizing off the binary one alone picks a limit the pair cells cannot
    # run in, and #3115 measured a 13x memory gap between them.
    IDX="${2:-0}"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/cell-$IDX"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    S=$(sbatch --parsable --job-name=fc3314-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size (cell $IDX)"
    echo "size job: $S (cell $IDX)  ->  $BASE/sizing/logs/size-$S.out"
    ;;

  list)
    # The cell count, and which index is which environment - so `size` picks a
    # real pair cell instead of guessing at the ordering.
    activate_venv
    export CALIB_EXP="$BASE/screen" CALIB_RESULTS="$PREP"
    (cd "$HERE" && python - <<'PYLIST'
import json
from pathlib import Path

import common

common.setup_env()
import experiment_config as cfg
from run_cells import _categories_by_dataset

info = json.loads((common.RESULTS / "prepare_info.json").read_text())
cells = cfg.array_cells(_categories_by_dataset(info))
print(f"{len(cells)} cells (order={cfg.CELL_ORDER})")
for i, c in enumerate(cells):
    styles = cfg.styles_for(c["dataset"], c["embedder"])
    print(f"  {i:4d}  {c['dataset']}/{c['embedder']}  styles={','.join(styles)}  {c['category']}  seed={c['seed']}")
PYLIST
    )
    ;;

  baseline)
    # The click-0 anchor: what typing the query was worth before any clicking.
    # `curves` refuses to draw a quality-over-clicks figure without it, and it
    # should not be optional - the distance between the far left and the far
    # right IS the study's subject.  Computed ONCE off the shared prepare, since
    # every stage shares the categories and the text sort does not depend on K.
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/analysis" "$BASE/logs"
    ENVX="export CALIB_EXP=$BASE CALIB_RESULTS=$PREP VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    T=$(sbatch --parsable --job-name=fc3314-baseline --mem=24G --cpus-per-task=2 \
      --time=2:00:00 --partition=cpu --export=ALL \
      --output="$BASE/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $PREP --out $BASE/analysis/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T  ->  $BASE/analysis/text_baseline.csv"
    ;;

  screen)
    if [[ ! -f "$PREP/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
      exit 1
    fi
    activate_venv
    export CALIB_EXP="$BASE/screen"
    export CALIB_RESULTS="$CALIB_EXP/results"
    mkdir -p "$CALIB_EXP/logs"
    link_prepare "$CALIB_RESULTS"
    # Nothing is pinned away from production here: the run LIVES at
    # `calibrate_count=2` and `CALIB_FOLD_COUNTS` is an eval-only counterfactual
    # screen that cannot touch the trajectory.  So no `--diverges` - and if
    # check 12 ever flags something, that is a real finding about this launcher.
    run_preflight "fc3314-screen"
    echo "=== stage A: nested fold-count screen -> $CALIB_EXP ==="
    CALIB_JOB_NAME="fc3314-screen" bash "$HERE/launch_cells.sh" || {
      echo "screen FAILED to submit" >&2; exit 1
    }
    ARM_ID="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
    require_jobid "$ARM_ID" "the screen's cells array"
    echo "screen array: $ARM_ID"
    ;;

  ab)
    # STAGE B, gated.  Two live arms; the screen's own run is the K=2 control
    # trajectory (it lives at production's count under identical everything
    # else), so it is not re-run.  Both arms are FULL runs: a knob upstream of
    # acquisition cannot be screened inside one trajectory.
    #
    #   $2 = the fixed count for `k_best` (from the screen's banded table)
    #   $3 = the schedule for `k_adaptive`, e.g. "6@25" for
    #        K(n_votes) = 6 while n_votes < 25, else 2
    K_BEST="${2:?usage: $0 ab K_BEST SCHEDULE  (e.g. $0 ab 6 6@25)}"
    SCHED="${3:?usage: $0 ab K_BEST SCHEDULE  (e.g. $0 ab 6 6@25)}"
    if [[ ! -f "$PREP/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
      exit 1
    fi
    activate_venv
    # The A/B arms measure the TRAJECTORY a live K collects, so the
    # counterfactual screen is off: it would cost 4x the fold fits to re-answer
    # a question stage A already answered exactly.  `2,$K` keeps the paired
    # within-step control that licenses each arm's own delta.
    DEPS=()
    for ARM in "k$K_BEST" "kadapt"; do
      export CALIB_EXP="$BASE/ab-$ARM"
      export CALIB_RESULTS="$CALIB_EXP/results"
      mkdir -p "$CALIB_EXP/logs"
      link_prepare "$CALIB_RESULTS"
      DIV=(--diverges "calibrate_count")
      if [[ "$ARM" == "kadapt" ]]; then
        export CALIB_FOLD_COUNT_SCHEDULE="$SCHED"
        export CALIB_CALIBRATE_COUNT=2
        export CALIB_FOLD_COUNTS="2"
        # The schedule knob is itself the divergence; the live count it starts
        # from is production's.
        DIV=(--diverges "fold_count_schedule")
      else
        unset CALIB_FOLD_COUNT_SCHEDULE
        export CALIB_CALIBRATE_COUNT="$K_BEST"
        export CALIB_FOLD_COUNTS="2,$K_BEST"
      fi
      run_preflight "fc3314-$ARM" "${DIV[@]}"
      echo "=== stage B arm: $ARM -> $CALIB_EXP ==="
      CALIB_ANALYZE=noop.py CALIB_JOB_NAME="fc3314-$ARM" bash "$HERE/launch_cells.sh" || {
        echo "arm $ARM FAILED to submit" >&2; exit 1
      }
      ID="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
      require_jobid "$ID" "arm $ARM's cells array"
      DEPS+=("$ID")
    done
    # ONE cross-arm analysis after both arms drain.  `afterany`, not `afterok`:
    # an arm that loses cells to a node failure still has to be read and its
    # loss COUNTED, and an analyzer that never runs reports nothing at all.
    DEPSTR="$(IFS=:; echo "${DEPS[*]}")"
    ALOGS="$BASE/logs"; mkdir -p "$ALOGS"
    AENVX="export CALIB_EXP=$BASE/screen CALIB_RESULTS=$BASE/screen/results VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME CALIB_FOLD_CHECKPOINTS=$CALIB_FOLD_CHECKPOINTS"
    A=$(sbatch --parsable --dependency="afterany:$DEPSTR" --job-name=fc3314-ab-analyze \
      --mem="$CALIB_ANALYZE_MEM" --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" \
      --partition=cpu --export=ALL --output="$ALOGS/ab-analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_folds_3314.py --ab $BASE/ab-k$K_BEST/results --ab $BASE/ab-kadapt/results")
    require_jobid "$A" "the cross-arm analyze step"
    echo "ab analyze: $A"
    ;;

  *)
    echo "usage: $0 {prepare|list|size [idx]|baseline|screen|ab K_BEST SCHEDULE}" >&2
    exit 2
    ;;
esac
