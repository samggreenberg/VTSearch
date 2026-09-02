#!/usr/bin/env bash
# #3287: sweep `calibration_fraction` per voting mode.  Is the 50/50 Train/
# Calibrate split of each calibration fold's labelset actually optimal?
#
#   bash launch_calfrac_3287.sh prepare       # stage 0, ONCE, shared by every arm
#   bash launch_calfrac_3287.sh size [cell]   # time ONE cell before committing
#   bash launch_calfrac_3287.sh arms          # every fraction, one full run each
#   bash launch_calfrac_3287.sh arms 0.5      # just one fraction
#
# WHY THIS IS FIVE RUNS AND NOT FIVE PAIRED ARMS.  `calibration_fraction` sets
# the threshold, the threshold sets the acquisition cut, and the acquisition cut
# sets which item Autopilot's Hard pick samples next.  A run that genuinely lives
# at 0.3 therefore collects DIFFERENT votes from its first trained step, and the
# regret it ends at is not the regret a re-cut of the 0.5 trajectory would
# attribute to 0.3.  This is the same reason `calibrate_count` needed
# `launch_folds_2897_ab.sh` after its screen: a knob upstream of acquisition
# cannot be screened inside one trajectory.  The arms are consequently NOT
# paired on votes - they are paired on (dataset, category, seed, style), which
# is what the analyzer bootstraps over.
#
# WHAT IT MEASURES.  Not a mechanism and not an artifact: the cost a VTSearch
# user ends up paying, on average, at each split fraction -- under a simulation
# held as close to the shipped app as the harness can hold it.  The fused
# threshold path, the production linear-SVM head, `calibrate_count=2`, the app's
# per-mode blend schedule, and the same text-sort opening a user gets by typing
# a query.  The only thing that differs between arms is the fraction.
#
# "On average, across scenarios" is the shape of the answer, so the grid spends
# its cells on scenarios rather than on precision at one of them: three
# geometries (two voting modes, with the embedder separable from the mode), 12
# classes at identical prevalence, 4 seeds, and four vote bands across a
# 150-click horizon.
#
# Design + pre-registered decision rules:
#   docs/experiments/2026-08-27-calibration-fraction-3287/PLAN.md
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-calfrac-3287}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# One study, one base; one ARM, one CALIB_EXP.  Each fraction is a different
# trajectory, so sharing a results dir would interleave two grids' cells under
# indistinguishable task indices with no way to separate them afterwards.
BASE="${CALFRAC_BASE:-/expscratch/$USER/calfrac-3287}"

# The fractions the issue names.  0.5 is the incumbent and is an arm, not a
# baseline read from somewhere else: it has to be measured on this grid, under
# this code, or the four other arms have nothing to be a difference from.
FRACTIONS="${CALFRAC_FRACTIONS:-0.3,0.4,0.5,0.6,0.7}"

arm_dir() { printf '%s/f%03d' "$BASE" "$(python3 -c "print(round(float('$1')*100))")"; }

# Does this grid contain any patch-capable embedder?  Three preflight flags and
# the memory floor depend on the answer, and getting it from the embedder list
# rather than from a hand-set variable is what keeps a follow-up grid honest:
# `--contrasts-voting-modes` FAILS on a single-vector grid (nothing is in both
# modes, correctly), `--require-region-voting` would assert a premise about a
# cell the grid does not contain, and `--patch` would impose the 12G floor on
# cells that peak under 1 GB - which is the difference between 80 and 120
# concurrent tasks.
HAS_PATCH=0
case ",${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}," in
  *_patch,*|*_patch) HAS_PATCH=1 ;;
esac

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path.  `docs/ML.md`: "Every trained threshold fuses the
# haystack into the cut.  There is no setting for this."  Named rather than left
# unset: the fraction this study sweeps splits a fold INSIDE the fused path, so
# the path is the premise of the sweep, not incidental.  (The harness default
# was 0 - the #2781-era unfused control - until #3400; the pin is what kept this
# study off it, and is also what keeps it right when submitted into an older
# worktree.  Preflight check 12 now reads the resolved value, so an unset knob
# can no longer pass while the run measures the control.)
export CALIB_SAFE_THRESHOLDS=1
# Anchored / fold-count / cut-inclusion arms all stay OFF (their defaults).
# None of them is on the axis this study sweeps, and the region cell's cost is
# dominated by whatever per-step fitting is switched on.
# Each arm chains `noop.py`, not the analyzer.  `launch_cells.sh` always submits
# an `afterany` analyze step, and this study's analysis is CROSS-ARM - a
# fraction's number is only a result next to the other four - so pointing every
# arm at `analyze_calfrac.py` would run it five times on a fifth of the study
# each (the #2847 precedent, which is what `noop.py` was written for).  The real
# analyze job is submitted once below, dependent on every arm's array.
export CALIB_ANALYZE=noop.py

# CALIB_HEAD deliberately UNSET: `head=None` resolves to the linear SVM a live
# detector has trained since PR #3198.  Pinning the old logistic head would
# measure a split fraction on a detector nobody has (preflight check 12).
# CALIB_BLEND_SCHEDULE likewise unset: an explicit schedule overrides the app's
# per-mode default (#2841).

# --- environment -------------------------------------------------------------
# ONE dataset, `vg_scale_any` (#3156 + #3115): 12 hand-checked classes, 300
# positives each against one shared 3900-image negative pool, so the evaluable
# pool is 4200 at 7.1% prevalence, IDENTICAL in every cell.
#
# That uniformity is not a nicety here, it is the instrument.  A threshold IS a
# quantile of the calibration set, and this study's whole subject is how big
# that set should be - so a grid whose cells' calibration sets differ 60-fold in
# size (which `visual_genome_m`'s 25-to-1645 positives give) would confound the
# swept axis with itself.  Uniform prevalence also means a difference between
# two fractions cannot be a prevalence difference wearing a disguise.
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
# The region half is the PAIR `siglip+dinov3_patch` (#3278/#3276).  DINOv3 has
# no text tower, so a bare `dinov3_patch` arm opens on three random known-goods
# while every SigLIP arm opens on a typed query - which would put a SEEDING
# contrast inside the voting-mode contrast this study reports per mode.
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
# BOTH styles on the patch embedder, so voting mode is separable from embedder
# (#3115's confound-breaking corner):
#
#   siglip/whole  vs  dinov3/whole      -> the EMBEDDER, at fixed voting mode
#   dinov3/whole  vs  dinov3/max_patch  -> the VOTING MODE, at fixed embedder
#
# The issue asks for a per-MODE answer, and `PRODUCTION_SCHEDULE_BY_MODE` is the
# precedent it cites - so if a per-mode default falls out of this run, the
# contrast it rests on had better be the mode and not the representation.
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""

# --- sizing ------------------------------------------------------------------
# 150 steps because the issue's read is ACROSS VOTE BANDS, not pooled: few votes
# should favour spending them on the model, many votes should favour resolution,
# so the trade-off is predicted to REVERSE inside the horizon.  A pooled winner
# over a crossing is precisely the number that hides it.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# Seed-major, not the `category` default.  This run has a wall-clock deadline,
# and `CELL_ORDER`'s own note says which ordering that calls for: a SLURM array
# dispatches roughly in index order, so a truncated `category` run is missing its
# last CATEGORIES entirely - whole environments gone, and the per-mode contrast
# short at one end - while a truncated `seed` run is missing its last SEEDS,
# uniformly across every environment.  The first is a design failure and the
# second is a power one, and only the second is something a report can state and
# still be read.  It also means every arm degrades the SAME way, which matters
# more here than in a single-array study: five arms that lost different
# categories would not be comparable at all.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
# 4 seeds is the minimum that makes `sd(threshold)` - one of the issue's four
# metrics - computable at all: it is taken ACROSS seeds at a fixed step.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

# No GPU work anywhere: the cells train a small head on cached pile embeddings.
# `launch_cells.sh` defaults to `--partition=gpu --gres=gpu:v100:1`, where the
# `4gpu_tier` QOS would cap the array at 2 concurrent tasks.  BOTH must be named
# - the flag is dropped rather than passed as `--gres=none`, which this
# cluster's submit filter rewrites and then rejects (#2897 lost both A/B arms to
# exactly that, with an empty job id as the only symptom).
export CALIB_PARTITION=cpu
export CALIB_GRES=none

# --- MEASURED on THIS grid; see `size` below and the PLAN's sizing table ------
# Sized from #3115's cells, which ran the SAME dataset, categories, seeds, styles
# and step count, and then confirmed by `size` on this code before `arms` went
# in.  This run is strictly CHEAPER than that one: #3115 carried the K<=16
# fold-count grid, whose `fold_seconds` was 31% of its region cell, and the
# anchored EM arms.  Both are off here.
#
#   #3115, vg_scale_any x siglip        whole_image   20m55s  0.53 GB
#   #3115, vg_scale_any x dinov3_patch  max_patch     ~1h32m  7.12 GB
#
# 12G: the region peak measured 7.12 GB under MORE per-step fitting than this
# run does, held flat across three sstat samples.  12G leaves 41% headroom; an
# OOM here is a lost cell, not a slow one.  Do not size from the binary cell -
# 0.53 GB vs 7.12 GB is a 13x difference and the region cell sets the knob.
#
# CONCURRENCY IS PER ARM, AND THE STUDY'S BUDGET IS THE SUM.  16 x 5 arms x 12G
# = 960G of the 1074G per-user allowance under QOS `cpu_limit` (89%), just
# inside preflight check 8's 90% line.
#
# 16-and-not-80 is a scheduling decision, not a politeness one.  Five arrays
# submitted at %80 each request 4800G against a 1074G allowance, so SLURM would
# run the FIRST arm at 80 wide and park the rest in QOSMaxMemoryPerUser (#3129,
# three times in one evening).  Total throughput is the same either way - the
# allowance is the cap - but the SHAPE of a truncated run is completely
# different: at %80 a deadline that arrives early loses whole ARMS, and an arm
# is the axis this study sweeps.  At %16 every arm advances in lockstep, so the
# same deadline loses the last SEEDS, uniformly, across all five.  That is the
# same argument as `CALIB_CELL_ORDER=seed` above, one level up, and the two only
# work together: seed-major ordering inside an arm is worth nothing if the arms
# themselves run one after another.
# A patch grid is memory-bound (7.71 GB peak measured, 12G floor enforced by
# preflight check 7b); a single-vector grid is CPU-bound (0.94 GB peak measured,
# so 4G is ~4x headroom) and can therefore run to the QOS's CPU cap instead.
#   patch:        5 arms x %16 x 12G = 960G of 1074G  (memory-capped)
#   single-vector: 5 arms x %24 x  4G = 480G, 120 tasks (cpu=240, 2/task: CPU-capped)
if [[ "$HAS_PATCH" == "1" ]]; then
  export CALIB_MEM="${CALIB_MEM:-12G}"
  export CALIB_CONC="${CALIB_CONC:-16}"
else
  export CALIB_MEM="${CALIB_MEM:-4G}"
  export CALIB_CONC="${CALIB_CONC:-24}"
fi
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-48G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-2:00:00}"

# Pin the BLAS pools to one thread each: 80 concurrent cells each spawning a
# node-sized pool oversubscribes whatever node they land on.  Exported HERE, not
# inside a mode, so `size` and `arms` measure and run under the SAME environment
# - a cell timed with different threading than the array will use is a guess
# with a unit attached (`lessons/2026-08-24-a-login-node-timing-nearly-cut-an-arm.md`).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

# Stage 0 is shared by every arm and must be, not merely may be.  `prepare`
# selects the categories and builds the exemplar crops, and neither depends on
# `calibration_fraction` - but `run_cells.py --print-cells` ENUMERATES from
# `prepare_info.json`, so five independent prepares would be five chances for
# array index 37 to mean a different cell in different arms.  One prepare, five
# symlinks, one cell numbering.
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
  # Point one arm's results dir at the shared stage 0.
  local rd="$1"
  mkdir -p "$rd/cells"
  [[ -e "$rd/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$rd/prepare_info.json"
  [[ -e "$rd/crops" ]] || ln -s "$PREP/crops" "$rd/crops"
}

case "$MODE" in
  prepare)
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=cf3287-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    echo "cells enumerate from $PREP/prepare_info.json"
    ;;

  size)
    # Time ONE cell before committing to five arrays.  Quoting a previous grid's
    # seconds is how #3129 produced a 90-minute overestimate; this run's per-cell
    # cost is not #3115's, because its fold-count grid and anchored EM are off.
    # Under `CALIB_CELL_ORDER=seed` the first seed's 24 environments come first:
    # cells 0-11 are `siglip` (binary), cells 12-23 the `siglip+dinov3_patch`
    # pair (region).  So size from cell 0 AND cell 12 - the region cell sets the
    # memory and the critical path, and sizing off the binary one alone would
    # pick a limit the region arm cannot run in.  (Note this differs from
    # #3115's 0/48: that run took the `category` default, where the ordering -
    # and therefore what a cell index MEANS - is different.)
    IDX="${2:-0}"
    F="${3:-0.5}"
    export CALIB_CALIBRATION_FRACTION="$F"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/${CALIB_DATASETS//,/_}-f$F"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS CALIB_CALIBRATION_FRACTION=$F VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    S=$(sbatch --parsable --job-name=cf3287-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, fraction $F)  ->  $BASE/sizing/logs/size-$S.out"
    ;;

  baseline)
    # The click-0 anchor: what typing the query was worth before any clicking.
    # `curves` refuses to draw a quality-over-clicks figure without it, and it
    # should not be optional -- the distance between the far left and the far
    # right IS the study's subject.  Computed ONCE off the shared prepare, since
    # every arm shares the categories and the text sort does not depend on the
    # split fraction.
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/analysis" "$BASE/logs"
    ENVX="export CALIB_EXP=$BASE CALIB_RESULTS=$PREP VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    T=$(sbatch --parsable --job-name=cf3287-baseline --mem=24G --cpus-per-task=2 \
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

    WANT="${2:-$FRACTIONS}"
    SUBMITTED=0
    DEPS=()
    ARM_DIRS=()
    # Preflight's memory check asks "does this claim your whole allowance?", and
    # for a five-array study the honest answer is about the SUM, not about one
    # array.  Passing one arm's %N would report 18% and wave through a study
    # that actually sits at 89%.
    N_ARMS=0
    for _f in ${WANT//,/ }; do N_ARMS=$((N_ARMS + 1)); done
    STUDY_CONC=$((CALIB_CONC * N_ARMS))
    echo "study footprint: $N_ARMS arms x %$CALIB_CONC x $CALIB_MEM = %$STUDY_CONC concurrent"
    for F in ${WANT//,/ }; do
      python3 -c "
import sys
f = float('$F')
sys.exit(0 if 0.0 < f < 1.0 else 1)
" || { echo "ERROR: '$F' is not a fraction in (0, 1)" >&2; exit 1; }

      export CALIB_CALIBRATION_FRACTION="$F"
      export CALIB_EXP="$(arm_dir "$F")"
      export CALIB_RESULTS="$CALIB_EXP/results"
      mkdir -p "$CALIB_EXP/logs"
      link_prepare "$CALIB_RESULTS"

      # Every arm but the incumbent pins a knob away from its shipped constant,
      # which is the whole point and is exactly what check 12 flags.  Declare it
      # rather than skipping the check: `--diverges` names the divergence in the
      # run's own record, so a reader of the results dir can see that 0.3 was a
      # deliberate arm and not a stale pin someone forgot (which is what
      # `CALIB_HEAD=linear` turned out to be in #2865).
      # `--diverges` takes knob NAMES, not name=value: the check compares the
      # env var against the shipped default itself and this only acknowledges
      # that the difference is deliberate.
      DIV=()
      [[ "$F" != "0.5" ]] && DIV=(--diverges "calibration_fraction")

      # Only assert what this grid actually contains.  A check that cannot
      # apply is not a check that passes - it is a claim in the run's record
      # that nobody verified.
      GEO=()
      if [[ "$HAS_PATCH" == "1" ]]; then
        GEO=(--require-region-voting vg_scale_any:siglip+dinov3_patch --contrasts-voting-modes --patch)
      fi
      if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
        bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
          --require-min-positives 100 \
          --reuse-prepare "$PREP" \
          "${GEO[@]}" \
          "${DIV[@]}" \
          --job-name "cal-cells-f$F" --mem "$CALIB_MEM" --conc "$STUDY_CONC" || {
          echo "preflight FAILED for fraction $F" >&2
          [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
        }
      fi

      echo "=== arm: calibration_fraction=$F -> $CALIB_EXP ==="
      # A distinct job name per arm: the per-name completion waiter in the
      # grid-experiments skill counts jobs BY NAME, so five arrays sharing one
      # name would make "has arm 0.3 drained?" unanswerable.
      CALIB_JOB_NAME="cal-cells-f$F" bash "$HERE/launch_cells.sh" || {
        echo "arm $F FAILED to submit" >&2; exit 1
      }
      ARM_ID="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
      require_jobid "$ARM_ID" "arm $F's cells array"
      DEPS+=("$ARM_ID")
      ARM_DIRS+=("$CALIB_EXP")
      SUBMITTED=$((SUBMITTED + 1))
    done

    # ONE analysis, after every arm drains.  `afterany` rather than `afterok` on
    # purpose: an arm that loses cells to a node failure still has to be read and
    # its loss COUNTED, and an analyzer that never runs reports nothing at all.
    # The analyzer is what enforces the completeness rule, not the dependency.
    DEPSTR="$(IFS=:; echo "${DEPS[*]}")"
    ALOGS="$BASE/logs"; mkdir -p "$ALOGS"
    AENVX="export VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    A=$(sbatch --parsable --dependency="afterany:$DEPSTR" --job-name=cf3287-analyze \
      --mem="$CALIB_ANALYZE_MEM" --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" \
      --partition=cpu --export=ALL --output="$ALOGS/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_calfrac.py --base $BASE --out $BASE/analysis --baseline $BASE/analysis/text_baseline.csv")
    require_jobid "$A" "the cross-arm analyze step"

    echo
    echo "Submitted $SUBMITTED arm(s): ${DEPS[*]}"
    echo "cross-arm analyze: $A (afterany on every arm)  ->  $ALOGS/analyze-$A.out"
    echo "report -> $BASE/analysis/REPORT_calfrac.md"
    echo
    echo "A submission is not a launch: confirm the ids above are numeric and that"
    echo "cells appear under $BASE/f*/results/cells before quoting an ETA."
    ;;

  *)
    echo "usage: $0 {prepare|baseline|size [cell] [fraction]|arms [fractions]}" >&2
    exit 2
    ;;
esac
