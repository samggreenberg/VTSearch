#!/usr/bin/env bash
# Good Mining - does a different Autopilot *opening* find better positives?  (#3267)
#
#   bash launch_good_mining.sh prepare   # stage 0 (cpu, reads the pile in place)
#   bash launch_good_mining.sh size      # time ONE cell before committing
#   bash launch_good_mining.sh arms      # the 8-arm grid
#
# Getting enough Goods looks like what separates a VTSearch run that works from
# one that fails, and the opening is where Goods come from.  Today it is fixed:
# the top of the seed sort until 3 positives, that sort's cutoff until 4
# negatives, then the learned Hard sort forever.  Both of those phases are the
# same operation - a rank-space `hard` select against a cut on the seed sort -
# at two different cuts, so the opening is really "how many clicks, at what
# depth", and this run sweeps that.  See vtscore/eval/startup_schedule.py for
# the grammar and docs/EVAL.md for how the arms are read.
#
#   prod           (unset)                     CONTROL - today's opening, so this
#                                              arm is comparable to every prior run
#   top_long       g8@top,b4@mid               the simplest hypothesis: just mine
#                                              more Goods before the Bad round
#   easy_med_hard  n5@q0.02,n5@q0.10,n6@mid    the issue's Easy/Medium/Hard bands
#   band_wide      n5@q0.05,n5@q0.25,n6@mid    the same shape, spread wider
#   incl_k         n5@k-6,n5@k-2,n6@k0         the SHIPPABLE lever: the seed sort's
#   incl_k_wide    n5@k-10,n5@k-4,n6@k0        own GMM split at three inclusions
#   flat_mid       n16@mid                     length-matched control: 16 opening
#                                              clicks, none of them Good-mining
#   deep_first     n10@q0.35,n6@mid            FALSIFIER - opens below the good
#                                              mass; must mine FEWER positives
#
# The falsifier is load-bearing.  If opening deliberately *below* the positives
# does not lose positives, then depth is not the mechanism and nothing else in
# the run is interpretable.
#
# `flat_mid` is the other load-bearing arm: every banded arm spends 16 opening
# clicks against prod's ~7, so without a length-matched control a win could just
# be "spend more clicks before training".  Read `easy_med_hard` against BOTH.
#
# The k-family arms may turn out to be inert: how far a given inclusion moves the
# pick is a property of the fitted mixture, and on a steep sort the whole usable
# range can land inside a couple of rank percent.  The analyzer checks that from
# `startup_cut_percentile` in the pick log and reports "measured nothing" rather
# than "the lever does nothing" - the same discipline analyze_acq.py applies.
#
# Prepare is REUSED (no GPU stage): every cell reads a pile pickle.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-goodmine-3267}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# One study, one dir.  On the 500G scratch: the pick log emits a row per click
# per cell, which is the frame this study is actually about, and /exp is a
# shared quota.
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/good-mining-3267}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# --- stages ---------------------------------------------------------------
# `live` is the two environments the acquisition-inclusion study ran on, so this
# run's `prod` is directly comparable to that one's.  `bands` adds the box-scale
# axis and TRIPLES the grid - launch it only after `live` has told you what a
# cell actually costs (GRID-PLAYBOOK.md: size it from a real cell, not a guess).
GM_STAGE="${GM_STAGE:-live}"
case "$GM_STAGE" in
  live)  export CALIB_DATASETS=coco_val,visual_genome_m ;;
  bands) export CALIB_DATASETS=vg_box_small,vg_box_medium,vg_box_large ;;
  *) echo "ERROR: GM_STAGE must be 'live' or 'bands' (got '$GM_STAGE')" >&2; exit 1 ;;
esac

# One embedder everywhere: siglip is the shipped default and the only pile
# column present for all five datasets, so an arm difference is never confounded
# with which embedder that dataset happened to carry.
#
# It is also the only choice this study CAN make.  Every arm names a position on
# the **seed sort**, and the seed sort is a text sort - the user types a query
# and votes down the ranking.  DINOv3 has no text tower (`embed_text` returns
# None), so a DINOv3 cell falls back to the app's other start, three random
# known-goods, and a cut at "the 2nd rank percentile" of that ranking is a cut on
# a different object.  A text-seeded and a known-good-seeded cell cannot be arms
# of one experiment.
export CALIB_COCO_EMBEDDERS=siglip
export CALIB_VG_EMBEDDERS=siglip
export CALIB_VGBOX_EMBEDDERS=siglip

# ...and the other half of that premise, asserted rather than assumed.  Whether
# a cell seeds from text is decided silently by whether a query text exists for
# its (dataset, category): `coco_val` and `vg_box_*` had NONE (they are
# experiment fixtures, not demo datasets, so they are absent from the app's
# EVAL_DATASETS query table), which is the gap
# lessons/2026-08-26-the-harness-seeded-from-a-crop.md closed for `vg_scale` and
# left open here in as many words.  Two controls now:
#   - EXPERIMENT_QUERIES gained a COCO-80 table and the vg_box_* bands;
#   - CALIB_REQUIRE_SEED_QUERY makes prepare select only from categories that
#     have one, so a category with no sensible query is replaced rather than
#     silently seeded the other way;
#   - preflight's --require-text-seed refuses to launch if any selected cell
#     would still take the known-good start.
export CALIB_REQUIRE_SEED_QUERY=1

# --- the horizon ----------------------------------------------------------
# 200 CLICKS PER RUN, and the opening's clicks are inside that number.
# `max_steps` bounds the whole voting loop (`n_steps = min(max_steps, len(pool))`
# and every iteration is one vote) - the opening's clicks simply emit no *metric*
# row, because before one Good and one Bad coexist there is no model to score.
# Counting them is what makes the arms comparable: every banded arm spends 16
# clicks opening against prod's ~7, so at a fixed 200 clicks a win is a better
# use of the same user effort rather than more of it.
export CALIB_MAX_STEPS=${CALIB_MAX_STEPS:-200}

# --- environments ---------------------------------------------------------
# PREVALENCE, not scale bands.  Both datasets are boxed, so selection would
# default to stratifying on box scale - but the mechanism this study runs on is
# how RARE the positives are (Good-starvation), and #3156 established that
# scatter is a property of an image rather than of a class, which is what a
# per-class scale band claims to be.  A prevalence spread gives the analyzer the
# axis to band on.
export CALIB_CATEGORY_MODE=prevalence
export CALIB_N_CATEGORIES=${CALIB_N_CATEGORIES:-12}
# Applied BEFORE the spread is drawn, so the rare end of the axis is the rarest
# category the horizon can actually sustain rather than one that gets dropped
# afterwards and shortens the axis.
export CALIB_MIN_CAT_COUNT=${CALIB_MIN_CAT_COUNT:-50}
# ...and the same floor again as a tripwire on the sim half (a header-only CSV
# passes every "N/N cells" count - see preflight check 13).
export CALIB_MIN_SIM_POSITIVES=${CALIB_MIN_SIM_POSITIVES:-25}

# Seeds, and the ordering that makes the number safe to be ambitious about.
#
# MEASURED on this grid (jobs 554487-554490, 2026-08-26): a 200-click cell is
# 4m19s - 5m26s at 0.86 GB peak on either dataset and on both a schedule arm and
# `prod`.  That is a quarter of #3115's cell, because this run carries no fold
# grid and no anchored arms - which is exactly why its own number had to be
# measured rather than scaled from that study's 20m55s.
#
# CELL_ORDER=seed walks every environment at seed 0, then every environment at
# seed 1.  A SLURM array dispatches roughly in index order, so this decides what
# a run that does not finish LOSES: seed-major loses high seeds uniformly (wider
# error bars, design intact), category-major loses whole environments off the end
# of the prevalence axis.  With that in place, over-provisioning seeds is the
# right call rather than a gamble - the grid degrades into a smaller version of
# itself instead of a different one.
export CALIB_CELL_ORDER=${CALIB_CELL_ORDER:-seed}
export CALIB_N_SEEDS=${CALIB_N_SEEDS:-30}

# --- production-faithful fixed choices ------------------------------------
# CALIB_HEAD is deliberately UNSET: `head=None` resolves to
# `voting_iterations.PRODUCTION_HEAD`, the linear SVM a live detector has trained
# since PR #3198.  The shipped launcher pinned `CALIB_HEAD=linear`, the logistic
# head that was production for #2790-#2865 - carrying that pin forward would
# measure an opening on a detector nobody has.  That is the failure preflight
# check 12 exists for (#2865).
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=0
export CALIB_SCHEDULE_VARIANTS=
export CALIB_REPOOL_VARIANTS=
# The pick log IS the study: the opening emits no main row (no detector exists
# yet), so without this an arm's mining behaviour is invisible.
export CALIB_EMIT_PICKS=1

# --- ops: cpu partition, single-threaded cells (see launch_acq_incl.sh) ---
export CALIB_PARTITION=cpu
export CALIB_GRES=none
# 3G against a measured 0.86 GB peak (3.5x headroom): an OOM here is a lost
# cell, not a slow one.  120 x 3G = 360G of the 1074G per-user allowance (34%),
# well inside preflight check 8's 90% line.
export CALIB_MEM=${CALIB_MEM:-3G}
export CALIB_CPUS=1
# 11x the measured cell, which is generous on purpose: a tight limit on a
# shared node buys nothing and turns a slow node into a lost cell.
export CALIB_TIME=${CALIB_TIME:-1:00:00}
# Per ARM, and there are eight arrays: 15 x 8 = 120, which is the whole cap the
# `cpu_limit` QOS allows (cpu=240, 2 charged per task).
#
# Deliberately split EVENLY rather than letting each array ask for the full 120.
# The arms then advance in lockstep, so a run that is stopped early has the same
# seeds finished in every arm - and every contrast in this study is PAIRED within
# (dataset, category, seed), so an arm that raced ahead would contribute cells
# with no partner and buy nothing.
export CALIB_CONC=${CALIB_CONC:-15}
export CALIB_ANALYZE_MEM=${CALIB_ANALYZE_MEM:-32G}
export CALIB_ANALYZE_TIME=${CALIB_ANALYZE_TIME:-1:00:00}

# Pin the BLAS pools to one thread each: 120 concurrent cells each spawning a
# node-sized pool oversubscribes whatever node they land on (#2883, and the
# half of that lesson that survived measurement).  Exported HERE so `size` and
# `arms` measure and run under the same environment - a cell timed with
# different threading than the array will use is a guess with a unit attached.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# The shipped launcher pointed VTSEARCH_DATA_DIR at "$CALIB_EXP/datadir", which
# does not exist - every cell would have re-fetched and re-embedded.
source "$WT/scripts/experiments/pile/pile_env.sh"

# Analysis is cross-arm and runs once, by hand, after every arm drains.
export CALIB_ANALYZE=${CALIB_ANALYZE:-noop.py}

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"

ENVX="export CALIB_EXP=$CALIB_EXP VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

ARM_ORDER=(prod top_long easy_med_hard band_wide incl_k incl_k_wide flat_mid deep_first)
declare -A ARMS=(
  [prod]=""
  [top_long]="g8@top,b4@mid"
  [easy_med_hard]="n5@q0.02,n5@q0.10,n6@mid"
  [band_wide]="n5@q0.05,n5@q0.25,n6@mid"
  [incl_k]="n5@k-6,n5@k-2,n6@k0"
  [incl_k_wide]="n5@k-10,n5@k-4,n6@k0"
  [flat_mid]="n16@mid"
  [deep_first]="n10@q0.35,n6@mid"
)

case "$MODE" in
  prepare)
    # Stage 0 on the CPU partition: every pair is already in the pile, so this
    # loads each pickle, re-derives the selected categories and writes the
    # startup-exemplar vectors.  No model is constructed.
    P=$(sbatch --parsable --job-name=gm-prep --mem=24G --cpus-per-task=2 \
      --time=1:00:00 --partition=cpu --export=ALL \
      --output="$LOGS/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$CALIB_RESULTS && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $LOGS/prepare-$P.out"
    ;;

  size)
    # Time ONE cell before committing to the array.  This run's horizon is 200
    # clicks against #3115's 150 and its per-step training cost grows with the
    # label count, so that study's 20m55s is not this study's number - and
    # quoting a previous grid's seconds is how #3129 produced a 90-minute
    # overestimate.  A sizing dir per GRID: a cell index means something
    # different under a different dataset.
    IDX="${2:-0}"
    ARM="${3:-prod}"
    export CALIB_STARTUP_SCHEDULE="${ARMS[$ARM]}"
    SIZING="$CALIB_EXP/sizing/${GM_STAGE}_${ARM}"
    mkdir -p "$SIZING"
    S=$(sbatch --parsable --job-name=gm-size --mem=8G --cpus-per-task="$CALIB_CPUS" \
      --time=2:00:00 --partition=cpu --export=ALL \
      --output="$LOGS/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$CALIB_RESULTS CALIB_STARTUP_SCHEDULE='${ARMS[$ARM]}' && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, arm $ARM)  ->  $LOGS/size-$S.out   cells -> $SIZING"
    ;;

  arms)
    if [[ ! -f "$CALIB_RESULTS/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $CALIB_RESULTS - run '$0 prepare' first." >&2
      exit 1
    fi
    # Preflight's checks are mostly PYTHON: it imports vtscore to compare every
    # pinned knob against its shipped constant.  A non-interactive login shell
    # has no venv, where the system python is old enough that `X | None` raises
    # at import time - so those checks come back FAIL for a reason that has
    # nothing to do with the run.  Activate the venv first so they actually run;
    # the dangerous version of this is preflight reporting `ok` without having
    # looked (#2905).
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh" >/dev/null 2>&1 || {
      echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
    }

    # Reject a typo'd schedule here, not four hours into the array.  Every arm is
    # parsed by the same code the cells will run.
    for arm in "${ARM_ORDER[@]}"; do
      [[ -z "${ARMS[$arm]}" ]] && continue
      ( cd "$HERE" && python -c "
import sys
sys.path.insert(0, '$WT')
from vtscore.eval.startup_schedule import parse_startup_schedule
parse_startup_schedule('${ARMS[$arm]}')
" ) || { echo "ERROR: arm '$arm' has an unparseable schedule '${ARMS[$arm]}'" >&2; exit 1; }
    done

    if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
      # No --diverges: this run pins nothing off-production.  The opening is the
      # axis it sweeps and the shipped opening is the `prod` arm, so even that is
      # a comparison rather than a divergence.
      bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" \
        --arms "$(IFS=,; echo "${ARM_ORDER[*]}")" --job-name cal-cells \
        --mem "$CALIB_MEM" --conc "$CALIB_CONC" --need-gb 20 \
        --require-min-positives 50 --require-text-seed || {
        echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
      }
    fi

    for arm in "${ARM_ORDER[@]}"; do
      export CALIB_STARTUP_SCHEDULE="${ARMS[$arm]}"
      export CALIB_RESULTS="$CALIB_EXP/results/$arm"
      mkdir -p "$CALIB_RESULTS/cells"
      ln -sfn "$CALIB_EXP/results/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
      ln -sfn "$CALIB_EXP/results/crops" "$CALIB_RESULTS/crops"
      echo "=== arm $arm (startup_schedule='${CALIB_STARTUP_SCHEDULE:-app default}')"
      bash "$HERE/launch_cells.sh" || echo "ARM $arm SUBMIT FAILED" >&2
    done

    cat <<NOTE

Submitted.  A submission is not a launch: check every arm came back with a
numeric job id, then watch cells appear rather than watching squeue.

  squeue -u \$USER -n cal-cells
  for a in ${ARM_ORDER[*]}; do
    printf '%-14s %s\n' "\$a" "\$(ls $CALIB_EXP/results/\$a/cells/task_*.csv 2>/dev/null | grep -c . )"
  done

When the queue drains:  python analyze_startup.py
NOTE
    ;;

  *)
    echo "usage: $0 {prepare|size [cell] [arm]|arms}" >&2
    exit 2
    ;;
esac
