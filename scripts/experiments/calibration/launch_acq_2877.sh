#!/usr/bin/env bash
# #2877 on the pile: the acquisition cut's generalisation check, in a second
# environment that is BOTH voting modes.
#
#   bash launch_acq_2877.sh prepare              # stage 0, ONCE, shared by every arm
#   bash launch_acq_2877.sh baseline             # the click-0 text-sort anchor
#   bash launch_acq_2877.sh size bin|reg [cell]  # time ONE cell per half FIRST
#   bash launch_acq_2877.sh arms 8               # 8 seeds, both halves
#   bash launch_acq_2877.sh arms 8 bin           # just the cheap half
#   ACQ_JOB_TAG=t2 bash launch_acq_2877.sh arms 16-21 reg   # a top-up wave
#
# WHAT IS BEING MEASURED.  `ACQUISITION_INCLUSION_OFFSET` is the gap between the
# reported decision line and the rank position Autopilot's Hard pick samples
# around.  #2876 measured it on `coco_val x siglip2` and found an interior
# optimum at k=-3, which shipped; #2877 was the generalisation check and #2905
# the voting-mode one.  Neither answer survives:
#
#   * #2877 ran `visual_genome_m x siglip` believing it region-voted.  It does
#     not - no `patch_grid`, so `region_voting=True` fell back to whole-image
#     training, whole-image scoring and the binary blend.  A second BINARY
#     environment, and one that REJECTED -3.
#   * #2905 ran the real region arm and then lost it: dev `b7d528d8` (#2943,
#     two days later) fixed `_score_pool` scoring the pool whole-image while the
#     threshold was cut on region max-pooled scores, so the aggressive arms were
#     clamped exactly where the decision was.  PR #3119 voided the report.
#
# So the shipped constant is **-1** (#2909), a compromise chosen because it was
# the only value passing everywhere measured - and every measurement behind it
# is either binary-only or void.  This run is the generalisation check #2877
# asked for, on data and code that can carry it.
#
# THE ENVIRONMENT, AND WHY IT IS NOT `visual_genome_m`.  `vg_scale_any` (#3156 +
# #3115): 12 hand-checked classes, 300 positives each against ONE shared
# 3900-image negative pool, labelled from COCO's exhaustive annotation rather
# than VG's free text and repaired by a human review pass.  Prevalence is
# therefore IDENTICAL in every cell (7.1%) by construction.  That is what a
# calibration study wants and what `visual_genome_m` is not: its selected
# categories run 25 to 1645 positives, and its thin ones produce cells with no
# trainable step at all (#3115 launched 208 of them and its first two completed
# cells were header-only).  #2910 measured the offset's benefit as a decreasing
# function of positive SUPPLY; on `visual_genome_m` supply varies 60-fold
# ACROSS the cells being averaged, so a per-arm mean there is an average over
# the very axis the effect runs on.  Here it is held flat, which is what makes
# an arm difference attributable to the arm.
#
# TWO VOTING MODES, ONE EMBEDDER BETWEEN THEM.  #3115's headline was reported
# per voting mode off a grid whose binary cells were all SigLIP and whose region
# cells were all DINOv3: the sign flip was real, its attribution to the mode was
# not.  Preflight check 13b exists for that.  Here the pair
# `siglip+dinov3_patch` runs BOTH styles inside one task, off one loaded pickle:
#
#   siglip+dinov3_patch x whole_image  -> binary  }  same cell, same sim/test
#   siglip+dinov3_patch x max_patch    -> region  }  split, same exemplar
#
# so the mode contrast is paired on identical data and carries no embedder with
# it.  The `bin` half adds `siglip x whole_image` - the shipped default
# embedder, and the arm directly comparable to #2876/#2877, which were both
# single-vector whole-image environments.
#
# The region arm is the PAIR, not bare `dinov3_patch`.  DINOv3 has no text
# tower, so alone it opens on three random known-goods while every other arm
# opens on a typed query (#3269/#3278) - and k_acq is an offset applied to a
# RANK POSITION in a ranking the opening creates, so a seeding contrast inside
# the mode contrast would not be a detail.  SigLIP ranks the query, DINOv3 does
# every piece of learning.  `CALIB_REQUIRE_OPENING=text` asserts it per cell.
#
# WHY TWO HALVES AND NOT ONE ARRAY PER ARM.  A region cell holds a 2.4 GB patch
# pickle and peaks near 9 GB; a whole-image cell peaks near 1 GB.  One array per
# arm would reserve the region memory for all of it, and memory - not CPU - is
# the binding per-user quota here (`cpu_limit`: mem=1100000M).  Splitting lets
# the cheap half run wide and cheap and land a result while the expensive half
# grinds.  The two halves are separate index spaces on purpose; they share only
# `prepare_info.json`.
#
# ARMS are #2876/#2877/#2905's verbatim, so all the environments stay
# comparable.  `prod` MUST name its 0 explicitly - the default is -1, so a
# launcher copied from a pre-#2878 template would run seven arms of the same
# thing - and `acq_m1` IS the incumbent, which is why it is an arm and not a
# number read from a previous report.
#
#   prod      k_acq =  0   control - one threshold doing both jobs (pre-#2876)
#   acq_m1    k_acq = -1   THE SHIPPED DEFAULT (#2909)
#   acq_m2    k_acq = -2   where #2877 found the ranking benefit saturates
#   acq_m3    k_acq = -3   what #2876 shipped and #2877 rejected
#   acq_m4    k_acq = -4   far end
#   acq_p2    k_acq = +2   FALSIFICATION arm - must make positives WORSE
#   rank_pin  cut pinned at the conformal path's own pool percentile (0.959)
#
# Keep `acq_p2`.  It is the only thing separating "the lever works" from "any
# perturbation of the sampling position changes the numbers", and it is what
# made the rest interpretable in all three prior environments.
#
# SEEDS ARE AN ARRAY PREFIX, NOT A REMAP.  #2877's sizing mistake was inheriting
# #2876's 8 seeds: the decision-endpoint CI came back [-0.014, +0.019], a null
# too wide to certify and spanning two opposite decisions.  The fix is to
# re-derive n from a pilot's paired SD on `final_cost` - which cannot be known
# until cells exist.  So `CALIB_CELL_ORDER=seed` makes the array seed-major:
# index = seed * 12 + category, every seed block is a complete design, and a
# top-up is `arms 8-23` with every cell already on disk still counting.  Under
# the default `category` ordering, changing the seed count remaps every index
# and makes every earlier cell unresumable - the mistake #2877 could not undo.
#
# Design + pre-registered decision rules: same as #2876.  Decision endpoint is
# `final_cost` at t=100, paired at the (category, seed) cell; mechanism is
# positives at t=100/t=50 and `final_ap`; guardrails are deep-spike incidence,
# worst-step regret and oracle cost.  The ship rule and its tolerance live in
# `analyze_acq.py`, which reports them PER VOTING MODE.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-acq-2877}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# Not /exp: that 50G quota is mostly the venv, and a study's cells do not belong
# on it (GRID-PLAYBOOK section 4).
BASE="${ACQ_BASE:-/expscratch/$USER/acq-2877}"

ARMS_ALL="${ACQ_ARMS:-prod,acq_m1,acq_m2,acq_m3,acq_m4,acq_p2,rank_pin}"

# arm -> "CALIB_ACQ_INCLUSION_OFFSET CALIB_ACQ_RANK_PERCENTILE" ("-" = unset).
# Nothing here is left unset: an unset offset resolves to the shipped constant,
# which would silently make one arm a duplicate of `acq_m1` the day that
# constant moves.  `acq_m1` naming -1 explicitly is the same discipline as
# `prod` naming 0 -- and preflight check 12 sees -1 as MATCHING production, so
# only the other five arms declare the divergence.
arm_offset() {
  case "$1" in
    prod)     echo "0 -" ;;
    acq_m1)   echo "-1 -" ;;
    acq_m2)   echo "-2 -" ;;
    acq_m3)   echo "-3 -" ;;
    acq_m4)   echo "-4 -" ;;
    acq_p2)   echo "2 -" ;;
    rank_pin) echo "0 0.959" ;;
    *)        echo "__BAD__ __BAD__" ;;
  esac
}
arm_dir() { printf '%s/%s/%s' "$BASE" "$1" "$2"; }

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path.  `docs/ML.md`: "Every trained threshold fuses the
# haystack into the cut.  There is no setting for this."  Named rather than left
# unset: the acquisition cut is taken off the FUSED threshold, so this is the
# premise of the arm axis rather than incidental, and the run's path is readable
# here instead of from a harness default three modules away.  (That default was
# the #2781-era unfused control until #3400; the pin is what kept this study off
# it, and is also what keeps it right when submitted into an older worktree.)
export CALIB_SAFE_THRESHOLDS=1
# Every arm chains `noop.py`; the analysis is CROSS-ARM and CROSS-MODE and is
# submitted once, dependent on every array.
export CALIB_ANALYZE=noop.py
# Anchored / fold-count / cut-inclusion grids stay off (their defaults): none is
# on this study's axis, and the region cell's cost is dominated by whatever
# per-step fitting is switched on.
#
# CALIB_HEAD deliberately UNSET -> the production linear SVM (#3198).
# CALIB_CALIBRATION_FRACTION unset -> the per-space default (#3290).
# CALIB_EXCLUDE_VOTED unset -> the app's own floor (#3308).
# CALIB_BLEND_SCHEDULE unset -> `production_schedule_for` picks per voting mode
#   (#2849), so the region cells blend under `slow_cap50` and the binary ones
#   under `cap50`.  Pinning one would measure the offset under a schedule no
#   user of that mode runs -- and "the schedule is already mode-gated while the
#   offset is not" is half of what this question is about.

# --- environment -------------------------------------------------------------
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
# `vg_scale_any` carries its design in its category list, but `all` would also
# be right; `prevalence` at 12 takes every class the pickle holds and is what
# #3312 ran on this dataset, so the two studies select identically.
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
# BOTH styles on the pair, which is what makes the mode contrast paired within a
# cell rather than across two embedders (preflight check 13b).  `max_patch` is
# PRODUCTION_PATCH_STYLE, so check 12's `must_contain` is satisfied too.
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
# The t=100 horizon every prior environment used, so the decision endpoint means
# the same thing across all four.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-100}"
export CALIB_SIM_FRACTION="${CALIB_SIM_FRACTION:-0.5}"
# At 7.1% prevalence a 2100-media sim half holds ~150 positives, so this floor
# never binds here -- it is a tripwire, not a filter.  A cell that cannot seed
# would otherwise leave the arm's mean computed over the cells that happened to
# work.
export CALIB_MIN_SIM_POSITIVES="${CALIB_MIN_SIM_POSITIVES:-8}"

# Seed-major, and DECLARED at 24 while the study runs a prefix.  See the header.
export CALIB_CELL_ORDER=seed
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-24}"

# --- ops ---------------------------------------------------------------------
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
# Per-half; see half_env.  A 17-minute cell asking for 8 hours backfills slower
# for no benefit.
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-48G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-2:00:00}"

# Pin the BLAS pools to one thread each: concurrent single-cpu cells each
# spawning a node-sized pool oversubscribe whatever node they land on.  Exported
# HERE so `size` and `arms` measure and run under the SAME environment.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

# Per-half environment.  The halves differ ONLY in which embedder the grid
# enumerates, which is what makes them two index spaces off one prepare.
#
# MEASURED ON THIS GRID, on 2026-08-29, by `size` (jobs 590192/590193/590194),
# not inherited from a table:
#
#   reg (`siglip+dinov3_patch`, whole_image + max_patch)  16m05s / 16m42s, 5.0 GB
#   bin (`siglip`, whole_image)                            3m21s,          0.9 GB
#
# The region request is 12G anyway; see the note in `half_env`.
#
# GRID-PLAYBOOK's table would have said 16G for the region cell (its ~9.1 GB
# entry is `vg_scale x max_patch` under a different horizon); the cell actually
# peaks at 5.0 GB, and the difference is 8 slots of concurrency per arm.  That is
# the whole reason the playbook says to size from a cell rather than a table.
#
# The two `%N` throttles are then chosen to (a) stay inside `cpu_limit`
# (cpu=240 with 2 charged per task => 120 concurrent tasks; 7 arms x 16 = 112,
# leaving room for the analyze step) and (b) make the halves FINISH TOGETHER.
# A region cell is ~4.9x a binary one, so equal throttles would leave the cheap
# half idle for hours and then the analyze step waiting on the expensive one
# anyway.
half_env() {
  case "$1" in
    bin)
      export CALIB_VGSCALE_EMBEDDERS="siglip"
      export CALIB_MEM="${ACQ_BIN_MEM:-3G}"
      export CALIB_CONC="${ACQ_BIN_CONC:-4}"
      export CALIB_TIME="${ACQ_BIN_TIME:-0:30:00}"
      ;;
    reg)
      export CALIB_VGSCALE_EMBEDDERS="siglip+dinov3_patch"
      # 12G, not the 5.0 GB two cells of this exact kind actually peaked at.
      # Preflight check 7b refuses a patch array under 12G, because sizing one
      # from a cell that silently fell back to `whole_image` is not a near miss
      # -- it is OUT_OF_MEMORY on most of the arm, hours in, after the array has
      # been running long enough to look healthy (#3156: 74 of 108 cells).
      #
      # That is not what happened here: the sizing cells resolved to
      # `styles=['whole_image', 'max_patch']` and emitted 3201 max_patch rows
      # each.  So the floor is a TABLE disagreeing with a MEASUREMENT, and the
      # measurement is 2.4x under it.  Obeying the floor anyway is deliberate:
      # `sacct`'s MaxRSS is sampled, so it can miss a short peak, and the cost
      # of being wrong is asymmetric -- an over-request costs concurrency, an
      # under-request costs the arm.  What it costs is real and worth naming:
      # 7G x 63 slots = 441G of a 1074G quota, which is ~40% of this run's wall
      # clock.  Recorded rather than silently paid, and the honest fix is a
      # check that reads a study's own sizing evidence, not `--warn-only`.
      export CALIB_MEM="${ACQ_REG_MEM:-12G}"
      export CALIB_CONC="${ACQ_REG_CONC:-9}"
      export CALIB_TIME="${ACQ_REG_TIME:-1:30:00}"
      ;;
    *) echo "ERROR: unknown half '$1' (expected bin or reg)" >&2; exit 2 ;;
  esac
}

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

# How many (dataset, embedder, category) environments this half enumerates per
# seed.  Read from the harness rather than assumed: a seed block's width is what
# the array spec is built out of, and hardcoding 12 would silently mis-slice the
# array the day a category is dropped for want of a query.
envs_per_seed() {
  local n
  n=$(cd "$HERE" && CALIB_RESULTS="$PREP" CALIB_N_SEEDS=1 python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -eq 0 ]]; then
    echo "ERROR: could not determine the per-seed environment count (got '$n')" >&2
    exit 1
  fi
  echo "$n"
}

case "$MODE" in
  prepare)
    # ONE prepare, covering BOTH embedders, shared by every arm of both halves.
    # `run_cells.py --print-cells` enumerates from `prepare_info.json`, so
    # independent prepares would be chances for index 37 to mean a different
    # cell in different arms.  Keyed by embedder name, which is why the pair
    # needs its own entry and a rename means re-running this (check 15).
    export CALIB_VGSCALE_EMBEDDERS="siglip,siglip+dinov3_patch"
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=acq2877-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    echo "cells enumerate from $PREP/prepare_info.json"
    ;;

  size)
    # Time ONE cell per half before committing to fourteen arrays.  A region
    # cell is not a whole-image cell scaled: it loads a 2.4 GB patch pickle and
    # max-pools over region nodes at every step.  Quoting another grid's seconds
    # is how #3129 produced a 90-minute overestimate.
    HALF="${2:-reg}"
    IDX="${3:-0}"
    ARM="${4:-acq_m3}"
    half_env "$HALF"
    read -r OFF PCT <<<"$(arm_offset "$ARM")"
    [[ "$OFF" == "__BAD__" ]] && { echo "ERROR: unknown arm '$ARM'" >&2; exit 2; }
    export CALIB_ACQ_INCLUSION_OFFSET="$OFF"
    export CALIB_ACQ_RANK_PERCENTILE=""
    [[ "$PCT" != "-" ]] && export CALIB_ACQ_RANK_PERCENTILE="$PCT"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/$HALF-$ARM"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS CALIB_ACQ_INCLUSION_OFFSET=$CALIB_ACQ_INCLUSION_OFFSET CALIB_ACQ_RANK_PERCENTILE=${CALIB_ACQ_RANK_PERCENTILE:-} VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    S=$(sbatch --parsable --job-name=acq2877-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (half $HALF, cell $IDX, arm $ARM)  ->  $BASE/sizing/logs/size-$S.out"
    echo "read it with: sacct -j $S --format=JobID,JobName%18,MaxRSS,Elapsed,State"
    ;;

  baseline)
    # The click-0 anchor: what typing the query was worth before any clicking.
    # `curves` refuses to draw a quality-over-clicks figure without it, and it
    # should not be optional -- the distance between the far left and the far
    # right IS the study's subject.  Computed ONCE off the shared prepare: the
    # text sort depends on no arm.
    export CALIB_VGSCALE_EMBEDDERS="siglip,siglip+dinov3_patch"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/analysis" "$BASE/logs"
    ENVX="export CALIB_EXP=$BASE CALIB_RESULTS=$PREP CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    T=$(sbatch --parsable --job-name=acq2877-baseline --mem=24G --cpus-per-task=2 \
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
    # without having looked (#2905).
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh" >/dev/null 2>&1 || {
      echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
    }

    SEEDSPEC="${2:-8}"
    WHICH="${3:-bin,reg}"
    if [[ "$SEEDSPEC" == *-* ]]; then
      SLO="${SEEDSPEC%%-*}"; SHI="${SEEDSPEC##*-}"
    else
      SLO=0; SHI=$(( SEEDSPEC - 1 ))
    fi
    if (( SLO < 0 || SHI >= CALIB_N_SEEDS || SLO > SHI )); then
      echo "ERROR: seeds '$SEEDSPEC' fall outside 0..$((CALIB_N_SEEDS-1))" >&2; exit 2
    fi

    # The shipped constant, read from the tree that is about to run - once,
    # rather than per arm, and never written down here.  Check 12 compares
    # `CALIB_ACQ_INCLUSION_OFFSET` against it, so this is what decides which arms
    # have a divergence to declare; a number in this file would go stale
    # silently and the study would then either declare a divergence it does not
    # have or hide one it does.
    SHIPPED_K=$(python -c 'from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET as k; print(k)')
    if ! [[ "$SHIPPED_K" =~ ^-?[0-9]+$ ]]; then
      echo "ERROR: could not read ACQUISITION_INCLUSION_OFFSET (got '$SHIPPED_K')" >&2; exit 1
    fi
    echo "shipped ACQUISITION_INCLUSION_OFFSET = $SHIPPED_K"

    DEPS=()
    SUBMITTED=0
    for HALF in ${WHICH//,/ }; do
      half_env "$HALF"
      NENV="$(envs_per_seed)"
      LO=$(( SLO * NENV )); HI=$(( (SHI + 1) * NENV - 1 ))
      NPER=$(( HI - LO + 1 ))
      n_arms=0; for _a in ${ARMS_ALL//,/ }; do n_arms=$((n_arms + 1)); done
      # Preflight's memory check asks "does this claim your whole allowance?",
      # and for a multi-array study the honest answer is about the SUM.  Passing
      # one arm's %N would report a fraction and wave through a study that sits
      # at 90% of the quota.
      study_conc=$(( CALIB_CONC * n_arms ))
      echo "=== half $HALF: embedders=$CALIB_VGSCALE_EMBEDDERS  $NENV envs/seed"
      echo "    seeds ${SLO}..${SHI} of $CALIB_N_SEEDS declared -> array ${LO}-${HI} = $NPER cells/arm"
      echo "    $n_arms arms x %$CALIB_CONC x $CALIB_MEM = %$study_conc concurrent"
      GEO_DONE=0

      for ARM in ${ARMS_ALL//,/ }; do
        read -r OFF PCT <<<"$(arm_offset "$ARM")"
        [[ "$OFF" == "__BAD__" ]] && { echo "ERROR: unknown arm '$ARM'" >&2; exit 1; }
        export CALIB_ACQ_INCLUSION_OFFSET="$OFF"
        export CALIB_ACQ_RANK_PERCENTILE=""
        [[ "$PCT" != "-" ]] && export CALIB_ACQ_RANK_PERCENTILE="$PCT"
        export CALIB_EXP; CALIB_EXP="$(arm_dir "$HALF" "$ARM")"
        export CALIB_RESULTS="$CALIB_EXP/results"
        mkdir -p "$CALIB_EXP/logs"
        link_prepare "$CALIB_RESULTS"

        # A top-up is a SECOND array over the same results dir, and preflight
        # refuses a job name already in the queue -- rightly, since the per-name
        # completion waiter counts jobs by name and two waves sharing one name
        # make "has this arm drained?" unanswerable.  `ACQ_JOB_TAG=t2` names the
        # wave; the results dir, the index space and the cells are unchanged, so
        # the tag is about the QUEUE, never about the data.
        JOB_NAME="acq2877-$HALF-$ARM${ACQ_JOB_TAG:+-$ACQ_JOB_TAG}"

        # Check 12 reads `acq_offset` against the SHIPPED constant, which is -1.
        # So `acq_m1` matches and every other arm is a declared divergence --
        # which is the point: a study is always allowed to pin the axis it
        # sweeps, and never allowed to pin one silently.
        DIV=()
        [[ "$OFF" != "$SHIPPED_K" ]] && DIV=(--diverges "acq_offset")

        # The region-voting premise and the mode-contrast confound are
        # properties of the (dataset, embedder) CELL, so they are the same for
        # every arm of a half -- and asserting the first one costs a 2.4 GB
        # pickle load on the login node.  Checked once per half, on its first
        # arm, rather than seven times: a gate that is expensive enough to be
        # skipped is worse than one that runs.  `--patch` (the memory sizing)
        # rides along with it because it is equally per-half.
        GEO=()
        if [[ "$HALF" == "reg" && "$GEO_DONE" == "0" ]]; then
          GEO=(--require-region-voting "vg_scale_any:siglip+dinov3_patch" --contrasts-voting-modes --patch)
          GEO_DONE=1
        fi

        if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
          bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
            --require-min-positives 100 \
            --reuse-prepare "$PREP" \
            "${GEO[@]}" "${DIV[@]}" \
            --job-name "$JOB_NAME" --mem "$CALIB_MEM" --conc "$study_conc" || {
            echo "preflight FAILED for half $HALF arm $ARM" >&2
            [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
          }
        fi

        ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
        ENVX="$ENVX CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS"
        ENVX="$ENVX CALIB_ACQ_INCLUSION_OFFSET=$CALIB_ACQ_INCLUSION_OFFSET"
        ENVX="$ENVX CALIB_ACQ_RANK_PERCENTILE=${CALIB_ACQ_RANK_PERCENTILE:-}"
        ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

        # A distinct job name per (half, arm): the per-name completion waiter in
        # the grid-experiments skill counts jobs BY NAME, so two arrays sharing
        # one name make "has this arm drained?" unanswerable.
        J=$(sbatch --parsable --job-name="$JOB_NAME" --array="${LO}-${HI}%${CALIB_CONC}" \
          --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" --time="$CALIB_TIME" \
          --partition="$CALIB_PARTITION" --export=ALL \
          --output="$CALIB_EXP/logs/cells-%A_%a.out" \
          --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
        require_jobid "$J" "half $HALF arm $ARM's cells array"
        echo "--- $JOB_NAME (k=$OFF pct='${CALIB_ACQ_RANK_PERCENTILE:-}') job=$J -> $CALIB_EXP"
        echo "$J" > "$CALIB_EXP/logs/.cells_jobid"
        DEPS+=("$J")
        SUBMITTED=$((SUBMITTED + 1))
      done
    done

    if [[ "$SUBMITTED" -eq 0 ]]; then
      echo "ERROR: '$WHICH' selected no half (expected bin, reg or bin,reg)" >&2; exit 2
    fi

    # ONE analysis, after every array drains.  `afterany` rather than `afterok`
    # on purpose: an arm that loses cells to a node failure still has to be read
    # and its loss COUNTED, and an analyzer that never runs reports nothing at
    # all.  The analyzer enforces completeness, not the dependency.
    DEPSTR="$(IFS=:; echo "${DEPS[*]}")"
    ALOGS="$BASE/logs"; mkdir -p "$ALOGS"
    AENVX="export VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

    # The analysis scope is WHAT IS ON DISK, not what this invocation submitted.
    # The two halves can go in as two invocations -- they did here, because the
    # first stopped at preflight's patch-memory floor -- and an analyze scoped to
    # one of them does not merely report half a study: with one voting mode in
    # the frame `by_mode` is absent entirely, so the POOLED table silently
    # becomes the verdict.  That is the failure the per-mode split exists to
    # prevent, arriving through the launcher instead of through the statistics.
    # So the scope is discovered, and a dependency narrower than the scope is
    # said out loud rather than left to be noticed in the report.
    ASCOPE=""
    for h in bin reg; do
      [[ -d "$BASE/$h" ]] && ASCOPE="${ASCOPE:+$ASCOPE,}$h"
    done
    ASCOPE="${ASCOPE:-$WHICH}"

    A=$(sbatch --parsable --dependency="afterany:$DEPSTR" --job-name=acq2877-analyze \
      --mem="$CALIB_ANALYZE_MEM" --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" \
      --partition=cpu --export=ALL --output="$ALOGS/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_acq.py --base $BASE --halves $ASCOPE --out $BASE/analysis")
    require_jobid "$A" "the cross-arm analyze step"

    echo
    echo "Submitted $SUBMITTED array(s): ${DEPS[*]}"
    echo "cross-arm analyze: $A (afterany on this invocation's arrays)  ->  $ALOGS/analyze-$A.out"
    echo "analysis scope: --halves $ASCOPE (read off $BASE, not off this invocation)"
    if [[ "$ASCOPE" != "${WHICH//,/ }" && "$ASCOPE" != "$WHICH" ]]; then
      echo
      echo "WARNING: this analyze will read halves [$ASCOPE] but only waits on [$WHICH]."
      echo "         Extend it once the other half's arrays are known:"
      echo "           scontrol update JobId=$A Dependency=afterany:<id>:<id>:..."
      echo "         An analysis that runs before its inputs exist reports a study that"
      echo "         is missing a voting mode, and reports it as complete."
    fi
    echo "report -> $BASE/analysis/REPORT_acq.md"
    echo
    echo "A submission is not a launch: confirm the ids above are numeric and that"
    echo "cells appear under $BASE/*/*/results/cells before quoting an ETA."
    ;;

  *)
    echo "usage: $0 {prepare|baseline|size [bin|reg] [cell] [arm]|arms [seeds] [bin,reg]}" >&2
    exit 2
    ;;
esac
