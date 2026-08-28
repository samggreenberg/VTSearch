#!/usr/bin/env bash
# #3312: does the #3308 voted-media exclusion buy anything on real data, and is
# EXCLUSION_MIN_REMAINDER=60 the right floor?
#
#   bash launch_exclusion_3308.sh prepare        # stage 0, ONCE, shared by every arm
#   bash launch_exclusion_3308.sh baseline       # the click-0 text-sort anchor
#   bash launch_exclusion_3308.sh size A [cell]  # time ONE cell per stage first
#   bash launch_exclusion_3308.sh arms           # both stages
#   bash launch_exclusion_3308.sh arms A         # just stage A
#
# WHAT IS BEING MEASURED.  PR #3311 stopped the calibrated threshold grounding
# its quantiles in distributions that include the very votes its models trained
# on.  Everything justifying that today is synthetic: a Gaussian simulation, and
# one 60-media eval environment which is what forced the floor.  This run is the
# first time the change meets a real embedding.
#
# THE ARM AXIS IS ONE NUMBER.  `exclusion_min_remainder` is the smallest
# unlabeled remainder at which the exclusion still fires, so the arms are
# ordered and need no sentinel:
#
#   off    = inf  the exclusion never fires - the pre-#3308 baseline
#   always = 0    unconditional, no floor - what #3311 first implemented
#   (unset)= app  the shipped floor, resolved through the app's own
#                 `resolve_exclusion_floor`.  This is the INCUMBENT and it is
#                 deliberately not pinned to `60`: pinning would freeze the arm
#                 against a constant that can move underneath the study.
#   f250          a more conservative floor
#
# WHY TWO STAGES AND NOT ONE GRID.  The effect is bounded by the votes' share of
# the haystack, and the floor can only be measured where the remainder actually
# crosses it.  Those are different environments, and a single grid would answer
# neither well:
#
#   Stage A - production scale.  sim_fraction 0.5 on vg_scale_any is a ~2100-media
#             haystack; 150 clicks vote 7% of it and the remainder never falls
#             below ~1950.  The floor is therefore INERT here by construction, so
#             `always` and the app arm are the same arm and only `off` vs `on` is
#             a contrast.  Two arms, and the pre-registered expectation is that
#             they are not resolvable - the useful output is a BOUND.
#   Stage B - deep voting on a modest collection.  sim_fraction 0.10 gives a
#             ~420-media haystack and 380 clicks drive the remainder 419 -> 40,
#             so each arm switches off at a different, KNOWN step: f250 at ~170
#             votes, the app floor at ~360, `always` never.  That is what makes
#             a difference attributable to the floor rather than to the arm.
#
# WHY FULL RUNS AND NOT PAIRED RE-CUTS.  The floor sets the threshold, the
# threshold sets the acquisition cut, and the cut sets which item Autopilot's
# Hard pick samples next - so two arms have collected different votes by their
# second trained step.  Same reason `calibrate_count` (#2897) and
# `calibration_fraction` (#3287) each needed live A/B runs after their cheap
# screens.  Arms pair on (dataset, category, seed, style), which is what the
# analyzer bootstraps over.
#
# Design + pre-registered decision rules:
#   docs/experiments/voted-exclusion-3308/PLAN.md
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-exclusion-3312}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# One study, one base; one ARM, one CALIB_EXP.  Each arm is a different
# trajectory, so sharing a results dir would interleave two grids' cells under
# indistinguishable task indices with no way to separate them afterwards.
BASE="${EXCL_BASE:-/expscratch/$USER/exclusion-3312}"

# Arm names, per stage.  `app` is the incumbent and is an arm rather than a
# number read from somewhere else: it has to be measured on this grid, under
# this code, or the others have nothing to be a difference from.
STAGE_A_ARMS="${EXCL_A_ARMS:-off,app}"
STAGE_B_ARMS="${EXCL_B_ARMS:-off,always,app,f250}"

# The env value each arm name pins.  `app` pins NOTHING - an unset
# CALIB_EXCLUDE_VOTED is what makes it the production arm.
arm_env() {
  case "$1" in
    off)    echo "off" ;;
    always) echo "always" ;;
    app)    echo "" ;;
    f*)     echo "${1#f}" ;;
    *)      echo "__BAD__" ;;
  esac
}
arm_dir() { printf '%s/stage%s/%s' "$BASE" "$1" "$2"; }

HAS_PATCH=0
case ",${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}," in
  *_patch,*|*_patch) HAS_PATCH=1 ;;
esac

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path.  `docs/ML.md`: "Every trained threshold fuses the
# haystack into the cut.  There is no setting for this."  The harness default
# for `CALIB_SAFE_THRESHOLDS` is 0 - the #2781-era unfused control - and the
# exclusion lives INSIDE the fused path, so leaving this alone would sweep an
# arm axis that never executes.  Preflight check 12 is what catches that.
export CALIB_SAFE_THRESHOLDS=1
# Anchored / fold-count / cut-inclusion grids stay OFF (their defaults): none is
# on this study's axis, and the region cell's cost is dominated by whatever
# per-step fitting is switched on.
# Each arm chains `noop.py`; the real analysis is CROSS-ARM and is submitted once
# below, dependent on every arm's array (the #2847 precedent behind `noop.py`).
export CALIB_ANALYZE=noop.py

# CALIB_HEAD deliberately UNSET (the production linear SVM since PR #3198), and
# CALIB_CALIBRATION_FRACTION likewise (the per-space default, #3290).  Pinning
# either would measure the exclusion on a detector nobody has.

# --- environment -------------------------------------------------------------
# ONE dataset, `vg_scale_any` (#3156 + #3115): 12 hand-checked classes, 300
# positives each against one shared 3900-image negative pool, so the evaluable
# pool is 4200 at 7.1% prevalence, IDENTICAL in every cell.  That uniformity is
# the instrument here as it was in #3287: this study's axis is the votes' share
# of the haystack, so a grid whose cells' haystacks differed 60-fold in size
# would confound the swept axis with itself.
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""

# A small sim set is thin in positives - at 7.1% prevalence stage B's ~420-media
# haystack holds ~30 - and the split is a plain permutation, not stratified.  A
# cell that draws too few cannot seed and contributes no trainable step, which
# would leave the arm's mean computed over the cells that happened to work.
export CALIB_MIN_SIM_POSITIVES="${CALIB_MIN_SIM_POSITIVES:-8}"

# --- sizing ------------------------------------------------------------------
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-48G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-2:00:00}"

# Per-stage environment.  Stage B's cells are cheaper per step (a 5x smaller
# haystack is 5x less scoring) but run 2.5x more steps and carry more votes per
# step, so its cost is NOT stage A's divided by five - `size` both.
stage_env() {
  case "$1" in
    A)
      export CALIB_SIM_FRACTION="${EXCL_A_SIM_FRACTION:-0.5}"
      export CALIB_MAX_STEPS="${EXCL_A_MAX_STEPS:-150}"
      ;;
    B)
      # 0.10 x 4200 = 420 sim media; 380 steps leaves 40.  Preflight check 16
      # fails the pair if the horizon ever exceeds the sim set, because the loop
      # then TRUNCATES silently and max_steps stops being the design.
      export CALIB_SIM_FRACTION="${EXCL_B_SIM_FRACTION:-0.10}"
      export CALIB_MAX_STEPS="${EXCL_B_MAX_STEPS:-380}"
      ;;
    *) echo "ERROR: unknown stage '$1' (expected A or B)" >&2; exit 2 ;;
  esac
  if [[ "$HAS_PATCH" == "1" ]]; then
    export CALIB_MEM="${CALIB_MEM:-12G}"
    export CALIB_CONC="${CALIB_CONC:-8}"
  else
    export CALIB_MEM="${CALIB_MEM:-4G}"
    export CALIB_CONC="${CALIB_CONC:-16}"
  fi
}

# Pin the BLAS pools to one thread each: concurrent cells each spawning a
# node-sized pool oversubscribe whatever node they land on.  Exported HERE so
# `size` and `arms` measure and run under the SAME environment
# (`lessons/2026-08-24-a-login-node-timing-nearly-cut-an-arm.md`).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

# Stage 0 is shared by every arm of BOTH stages and must be.  `prepare` selects
# the categories and builds the exemplar crops, neither of which depends on the
# exclusion floor or on sim_fraction - but `run_cells.py --print-cells`
# ENUMERATES from `prepare_info.json`, so independent prepares would be chances
# for array index 37 to mean a different cell in different arms.
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

submit_stage() {
  # Submit every arm of one stage; append each array's job id to DEPS.
  local stage="$1" want="$2"
  stage_env "$stage"
  local n_arms=0
  for _a in ${want//,/ }; do n_arms=$((n_arms + 1)); done
  # Preflight's memory check asks "does this claim your whole allowance?", and
  # for a multi-array study the honest answer is about the SUM.  Passing one
  # arm's %N would report a fraction and wave through a study that sits at 90%.
  local study_conc=$((CALIB_CONC * n_arms))
  echo "=== stage $stage: sim_fraction=$CALIB_SIM_FRACTION max_steps=$CALIB_MAX_STEPS"
  echo "    arms: $want  ($n_arms x %$CALIB_CONC x $CALIB_MEM = %$study_conc concurrent)"

  for ARM in ${want//,/ }; do
    local envval; envval="$(arm_env "$ARM")"
    [[ "$envval" == "__BAD__" ]] && { echo "ERROR: unknown arm '$ARM'" >&2; exit 1; }

    export CALIB_EXCLUDE_VOTED="$envval"
    export CALIB_EXP; CALIB_EXP="$(arm_dir "$stage" "$ARM")"
    export CALIB_RESULTS="$CALIB_EXP/results"
    mkdir -p "$CALIB_EXP/logs"
    link_prepare "$CALIB_RESULTS"

    # Every arm but `app` pins the floor away from the shipped constant, which
    # is the point and is exactly what check 12 flags.  Declare it rather than
    # skipping the check: `--diverges` names the divergence in the run's own
    # record, so a reader of the results dir can see the pin was deliberate and
    # not a stale leftover (which is what CALIB_HEAD=linear turned out to be in
    # #2865).  `--diverges` takes knob NAMES, not name=value.
    local DIV=()
    [[ -n "$envval" ]] && DIV=(--diverges "exclusion_floor")

    local GEO=()
    if [[ "$HAS_PATCH" == "1" ]]; then
      GEO=(--require-region-voting vg_scale_any:siglip+dinov3_patch --contrasts-voting-modes --patch)
    fi
    if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
      bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
        --require-min-positives 100 \
        --reuse-prepare "$PREP" \
        "${GEO[@]}" \
        "${DIV[@]}" \
        --job-name "excl-$stage-$ARM" --mem "$CALIB_MEM" --conc "$study_conc" || {
        echo "preflight FAILED for stage $stage arm $ARM" >&2
        [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
      }
    fi

    echo "--- arm: stage $stage / $ARM (CALIB_EXCLUDE_VOTED='${envval:-<unset>}') -> $CALIB_EXP"
    # A distinct job name per arm: the per-name completion waiter in the
    # grid-experiments skill counts jobs BY NAME, so arrays sharing one name
    # would make "has arm `off` drained?" unanswerable.
    CALIB_JOB_NAME="excl-$stage-$ARM" bash "$HERE/launch_cells.sh" || {
      echo "stage $stage arm $ARM FAILED to submit" >&2; exit 1
    }
    local ARM_ID; ARM_ID="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
    require_jobid "$ARM_ID" "stage $stage arm $ARM's cells array"
    DEPS+=("$ARM_ID")
    SUBMITTED=$((SUBMITTED + 1))
  done
}

case "$MODE" in
  prepare)
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    stage_env A
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=ex3312-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    echo "cells enumerate from $PREP/prepare_info.json"
    ;;

  size)
    # Time ONE cell per stage before committing to six arrays.  Quoting another
    # grid's seconds is how #3129 produced a 90-minute overestimate, and stage B
    # is NOT stage A scaled: its haystack is 5x smaller but its horizon is 2.5x
    # longer and its per-step vote count is far higher.
    #
    # Under CALIB_CELL_ORDER=seed the first seed's 24 environments come first:
    # cells 0-11 are `siglip` (binary), cells 12-23 the `siglip+dinov3_patch`
    # pair (region).  Size from cell 0 AND cell 12 - the region cell sets the
    # memory and the critical path.
    STAGE="${2:-A}"
    IDX="${3:-0}"
    ARM="${4:-app}"
    stage_env "$STAGE"
    ENVVAL="$(arm_env "$ARM")"
    export CALIB_EXCLUDE_VOTED="$ENVVAL"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/stage$STAGE-$ARM"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS CALIB_EXCLUDE_VOTED=$ENVVAL CALIB_SIM_FRACTION=$CALIB_SIM_FRACTION CALIB_MAX_STEPS=$CALIB_MAX_STEPS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    S=$(sbatch --parsable --job-name=ex3312-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (stage $STAGE, cell $IDX, arm $ARM)  ->  $BASE/sizing/logs/size-$S.out"
    ;;

  baseline)
    # The click-0 anchor: what typing the query was worth before any clicking.
    # `curves` refuses to draw a quality-over-clicks figure without it, and it
    # should not be optional - the distance between the far left and the far
    # right IS the study's subject.  Computed ONCE off the shared prepare: the
    # text sort depends on neither the floor nor sim_fraction.
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/analysis" "$BASE/logs"
    ENVX="export CALIB_EXP=$BASE CALIB_RESULTS=$PREP VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    T=$(sbatch --parsable --job-name=ex3312-baseline --mem=24G --cpus-per-task=2 \
      --time=2:00:00 --partition=cpu --export=ALL \
      --output="$BASE/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $PREP --out $BASE/analysis/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T  ->  $BASE/analysis/text_baseline.csv"
    ;;

  arms)
    if [[ ! -f "$PREP/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
      exit 1
    fi
    # Preflight's checks are mostly PYTHON: it imports vtscore to assert the
    # region-voting premise and to compare every pinned knob against its shipped
    # constant.  A non-interactive login shell has no venv, where the system
    # python is old enough that `X | None` raises at import time - so those
    # checks come back FAIL for a reason unrelated to the run.  Loud, and
    # therefore survivable; the dangerous version is preflight reporting `ok`
    # without having looked (#2905).  Activate the venv first.
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh" >/dev/null 2>&1 || {
      echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
    }

    WHICH="${2:-AB}"
    SUBMITTED=0
    DEPS=()
    [[ "$WHICH" == *A* ]] && submit_stage A "$STAGE_A_ARMS"
    [[ "$WHICH" == *B* ]] && submit_stage B "$STAGE_B_ARMS"
    if [[ "$SUBMITTED" -eq 0 ]]; then
      echo "ERROR: '$WHICH' selected no stage (expected A, B or AB)" >&2; exit 2
    fi

    # ONE analysis, after every arm drains.  `afterany` rather than `afterok` on
    # purpose: an arm that loses cells to a node failure still has to be read and
    # its loss COUNTED, and an analyzer that never runs reports nothing at all.
    # The analyzer is what enforces the completeness rule, not the dependency.
    DEPSTR="$(IFS=:; echo "${DEPS[*]}")"
    ALOGS="$BASE/logs"; mkdir -p "$ALOGS"
    AENVX="export VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    A=$(sbatch --parsable --dependency="afterany:$DEPSTR" --job-name=ex3312-analyze \
      --mem="$CALIB_ANALYZE_MEM" --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" \
      --partition=cpu --export=ALL --output="$ALOGS/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_exclusion.py --base $BASE --out $BASE/analysis --stages $WHICH --baseline $BASE/analysis/text_baseline.csv")
    require_jobid "$A" "the cross-arm analyze step"

    echo
    echo "Submitted $SUBMITTED arm(s): ${DEPS[*]}"
    echo "cross-arm analyze: $A (afterany on every arm)  ->  $ALOGS/analyze-$A.out"
    echo "report -> $BASE/analysis/REPORT_exclusion.md"
    echo
    echo "A submission is not a launch: confirm the ids above are numeric and that"
    echo "cells appear under $BASE/stage*/*/results/cells before quoting an ETA."
    ;;

  *)
    echo "usage: $0 {prepare|baseline|size [A|B] [cell] [arm]|arms [A|B|AB]}" >&2
    exit 2
    ;;
esac
