#!/usr/bin/env bash
# #3319: where does the acquisition-offset frontier TURN, and does the answer
# survive the deep regime?
#
#   bash launch_acq_3319.sh prepare            # stage 0, ONCE, shared by every arm
#   bash launch_acq_3319.sh baseline           # the click-0 text-sort anchor
#   bash launch_acq_3319.sh size bin [cell]    # time ONE cell FIRST
#   bash launch_acq_3319.sh arms 16 bin        # the main sweep, 16 seeds
#   ACQ_DEEP=1 bash launch_acq_3319.sh arms 8 bin    # the 400-click arm
#
# WHAT #3318 LEFT OPEN.  It shipped `ACQUISITION_INCLUSION_OFFSET = -3` on
# `vg_scale_any x siglip x whole_image`, 192 paired cells, and on that arm `-4`
# was **at least as good as `-3` on every endpoint** (+18 positives, AP +0.057,
# cost -0.017 against -0.020).  The trend had not turned, so `-4` was the EDGE
# OF THE GRID, not an optimum, and `-3` shipped as the conservative choice
# because `-4` costs a small resolvable regression under region voting.
#
# Two things follow, and neither is answerable from those cells.
#
# 1. WHERE DOES THE FRONTIER TURN?  Run past the edge: -5, -6, -8.  Three
#    outcomes, all informative - an interior optimum between -4 and -6 (then -3
#    is leaving quality on the table), a frontier that keeps improving (then the
#    object is not an offset at all but a separate acquisition rule), or a sharp
#    degradation (then -3 is confirmed interior, with a BOUND rather than by
#    assumption).
#
# 2. AT WHAT RESOLUTION?  Every environment behind this constant was measured on
#    INTEGER steps, and the integer grid is an assumption nobody has tested.  The
#    knob is a log2 scale - `inclusion_cost_weights` is `2**-k` - so one step
#    DOUBLES the evidence demanded and the integer grid is coarse by
#    construction: between -3 and -4 the cut jumps from 8:1 to 16:1 evidence with
#    nothing in between.  The half steps (-2.5, -3.5, -4.5) are a factor of
#    sqrt(2) apart and are real, realisable operating points, verified distinct
#    on the shipped path before this grid was written (they survive
#    `snap_cut_to_sample`; see `test_fractional_cuts_land_strictly_between_...`).
#    They matter because #3318's own numbers put -3 and -4 within each other's
#    CIs: if the optimum is at -3.5 the integer grid CANNOT see it, and the
#    "conservative -3 vs aggressive -4" framing is an artefact of the spacing.
#
# 3. DOES IT SURVIVE THE DEEP REGIME?  Every environment behind this constant -
#    #2876 and #2877 included - was measured at a 100-CLICK horizon, and #2910
#    measured the offset's benefit as concentrated where positives are SCARCE.
#    Deep voting is exactly where scarcity ends, so the sign may flip somewhere
#    past the point every prior study stopped.  `ACQ_DEEP=1` runs prod/-1/-3/-4
#    at 400 clicks.  NOTE the exhaustion hazard, which is why the arm is
#    reported with a positives-remaining trace: at 7.1% prevalence a 2100-media
#    sim half holds only ~150 positives, so a 400-click trajectory can run out
#    and flatten its own tail - an artefact that would read as "the offset stops
#    mattering", which is precisely the finding being tested for.
#
# WHAT A VALUE MEANS (why the half steps are principled, not a fishing grid).
# The loss `inclusion_cost_weights` names is a weighted sum of *rates*, each
# normalised by its own class, so prevalence divides out (`rate_crossing` puts
# the prior-odds factor back into `lam` precisely so the cut does not carry it).
# What is left is a pure likelihood-ratio threshold:
#
#     include x   <=>   f_pos(x) / f_neg(x)  >  2**-k
#
# So k=0 is the neutral-evidence point and **each step is one bit of evidence**.
# `-3` means "sample where the evidence for Good is 8:1", and `-3.5` means
# 11.3:1 - a real point on the same scale, not an interpolation between settings.
#
# THE ENVIRONMENT IS #2877'S, VERBATIM.  `vg_scale_any`, `CALIB_CATEGORY_MODE=
# prevalence`, 12 categories, `CALIB_REQUIRE_OPENING=text`, `CALIB_CELL_ORDER=
# seed`, everything else at production.  That is deliberate and it is the reason
# this file is a copy rather than a new design: the whole value of extending a
# grid is that the new arms are comparable to the old ones.
#
# BUT THE OLD CELLS ARE NOT REUSED.  #2877 ran on dev `53dd14cb4`; this runs on
# `917e7c0ec`, and the production defaults moved in between (#3287/#3290's
# per-space calibration fraction, #3308's exclusion floor). Mixing cells across
# those would confound the arm axis with a dev-commit axis, so EVERY arm -
# including the incumbents `-1`, `-3` and `-4` - is re-run here on one commit.
# The re-run of -3 and -4 is also the replication of #3318's headline, which is
# worth having for its own sake.
#
# ONLY THE `bin` HALF.  The shipped arm is `siglip x whole_image`.  #3318 showed
# the region half costs 16 min/cell against 3, i.e. most of the money, and the
# region cross-check is only needed for whatever new value WINS - so it is a
# follow-up wave (`reg`), not part of the first launch.  The half machinery is
# inherited intact so that wave is one command.
#
# ARMS.  #2877's five, plus the three half steps and the three deep values.
#
#   prod      k_acq =  0     control - one threshold doing both jobs (pre-#2876)
#   acq_m1    k_acq = -1     the pre-#3318 shipped value
#   acq_m2    k_acq = -2
#   acq_m2h   k_acq = -2.5   HALF STEP
#   acq_m3    k_acq = -3     THE SHIPPED DEFAULT (#3318) - the incumbent
#   acq_m3h   k_acq = -3.5   HALF STEP  (brackets -log2 of the prior odds, 3.71)
#   acq_m4    k_acq = -4     #3318's edge-of-grid, tied with -3 on every endpoint
#   acq_m4h   k_acq = -4.5   HALF STEP
#   acq_m5    k_acq = -5     PAST THE EDGE
#   acq_m6    k_acq = -6     PAST THE EDGE
#   acq_m8    k_acq = -8     PAST THE EDGE - reachable from the bottom of the
#                            slider (reporting is clamped to [-10,10] and the
#                            offset is deliberately unclamped), so a real
#                            operating point, not a synthetic one
#   acq_p2    k_acq = +2     FALSIFICATION arm - must make positives WORSE
#
# Keep `acq_p2`.  It has behaved in five environments and it is the only thing
# separating "the lever works" from "any perturbation changes the numbers".
# `rank_pin` is DROPPED: it has been rejected in five environments and this grid
# is about the offset's VALUE, not about replacing it with a pinned quantile.
#
# Design + pre-registered decision rules: `PLAN_3319.md`, which inherits #2876's.
# Decision endpoint is `final_cost` at t=100, paired at the (category, seed)
# cell; mechanism is positives at t=100/t=50 and `final_ap`; guardrails are
# deep-spike incidence, worst-step regret and oracle cost.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-acq-3319}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# Not /exp: that 50G quota is mostly the venv, and a study's cells do not belong
# on it (GRID-PLAYBOOK section 4).
BASE="${ACQ_BASE:-/expscratch/$USER/acq-3319}"

# The deep-regime wave is a SUBSET of the same arms at a longer horizon, never a
# different arm table: the two waves have to name the same cuts for the deep
# answer to be about the horizon rather than about the arms.
if [[ "${ACQ_DEEP:-0}" == "1" ]]; then
  ARMS_ALL="${ACQ_ARMS:-prod,acq_m1,acq_m3,acq_m4}"
  # The horizon nothing behind this constant has ever been measured at.
  export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-400}"
  # A tripwire, not a filter: every `vg_scale_any` cell holds ~150 sim positives
  # by construction, so this never excludes a category -- it fails loudly if the
  # pile is ever rebuilt thinner, rather than letting a 400-click arm quietly
  # average over cells whose tail is flat because they ran OUT of positives.
  export CALIB_MIN_SIM_POSITIVES="${CALIB_MIN_SIM_POSITIVES:-100}"
  # 400 steps is ~4x the work of 100 and the per-step fit grows with the vote
  # count, so this is NOT the 100-step wall time scaled; `size` measures it.
  export ACQ_BIN_TIME="${ACQ_BIN_TIME:-4:00:00}"
  ACQ_JOB_TAG="${ACQ_JOB_TAG:-deep}"
else
  ARMS_ALL="${ACQ_ARMS:-prod,acq_m1,acq_m2,acq_m2h,acq_m3,acq_m3h,acq_m4,acq_m4h,acq_m5,acq_m6,acq_m8,acq_p2}"
fi

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
    acq_m2h)  echo "-2.5 -" ;;
    acq_m3)   echo "-3 -" ;;
    acq_m3h)  echo "-3.5 -" ;;
    acq_m4)   echo "-4 -" ;;
    acq_m4h)  echo "-4.5 -" ;;
    acq_m5)   echo "-5 -" ;;
    acq_m6)   echo "-6 -" ;;
    acq_m8)   echo "-8 -" ;;
    acq_p2)   echo "2 -" ;;
    *)        echo "__BAD__ __BAD__" ;;
  esac
}
arm_dir() { printf '%s/%s/%s' "$BASE" "$1" "$2"; }

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path.  `docs/ML.md`: "Every trained threshold fuses the
# haystack into the cut.  There is no setting for this."  The harness default is
# the #2781-era unfused control, and the acquisition cut is taken off the FUSED
# threshold, so leaving this alone would sweep an arm axis that never executes.
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
      # 12 arms x %6 = 72 concurrent tasks.  `cpu_limit` charges 2 cpu per task
      # (240 => 120 tasks) and the memory quota is 1074G, so this run claims 144
      # cpu and 216G -- roughly 60% and 20%.  #2877 used %4 across 7 arms; the
      # arm count nearly doubled, so holding the per-arm throttle would have
      # doubled the wall clock instead of the width.
      export CALIB_CONC="${ACQ_BIN_CONC:-6}"
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

# The prepare is arm- AND horizon-independent (crops plus the cell enumeration),
# so the deep wave points `ACQ_PREP` at the main study's rather than building a
# second identical one -- which would also be a second index space, and index 37
# meaning two different cells is the failure `prepare` is centralised to avoid.
PREP="${ACQ_PREP:-$BASE/prepare/results}"

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
    P=$(sbatch --parsable --job-name=acq3319-prep --mem=24G --cpus-per-task=2 \
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
    S=$(sbatch --parsable --job-name=acq3319-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
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
    T=$(sbatch --parsable --job-name=acq3319-baseline --mem=24G --cpus-per-task=2 \
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
    # Integer today and expected to stay one -- production ships whole steps.  A
    # fractional shipped value would still work everywhere below (the comparison
    # against each arm is a STRING compare, exactly as preflight check 12 does
    # it), so this gate is about catching an unreadable value, not a float.
    if ! [[ "$SHIPPED_K" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
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
        JOB_NAME="acq3319-$HALF-$ARM${ACQ_JOB_TAG:+-$ACQ_JOB_TAG}"

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

    A=$(sbatch --parsable --dependency="afterany:$DEPSTR" --job-name=acq3319-analyze \
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
