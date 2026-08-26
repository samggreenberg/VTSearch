#!/usr/bin/env bash
# Good Mining — does a different Autopilot *opening* find better positives?  (#3267)
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

export VTS_REPO=${VTS_REPO:-/exp/$USER/projects/vts-good-mining}
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/good-mining"

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
export CALIB_COCO_EMBEDDERS=siglip
export CALIB_VG_EMBEDDERS=siglip
export CALIB_VGBOX_EMBEDDERS=siglip

export CALIB_MAX_STEPS=${CALIB_MAX_STEPS:-100}
export CALIB_N_SEEDS=${CALIB_N_SEEDS:-6}
export CALIB_N_CATEGORIES=${CALIB_N_CATEGORIES:-6}
export CALIB_N_PER_BAND=${CALIB_N_PER_BAND:-3}
export CALIB_HEAD=linear
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
export CALIB_MEM=${CALIB_MEM:-8G}
export CALIB_CPUS=1
export CALIB_TIME=${CALIB_TIME:-2:00:00}
export CALIB_CONC=${CALIB_CONC:-18}

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

# Analysis is cross-arm and runs once, by hand, after every arm drains.
export CALIB_ANALYZE=${CALIB_ANALYZE:-noop.py}

mkdir -p "$CALIB_EXP/logs"

if [[ ! -f "$CALIB_EXP/results/prepare_info.json" ]]; then
  echo "ERROR: no prepare_info.json at $CALIB_EXP/results" >&2
  echo "  Reuse a finished study's prepare, or run prepare_data.py for this grid." >&2
  exit 1
fi

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
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" \
    --arms "$(IFS=,; echo "${ARM_ORDER[*]}")" --job-name cal-cells \
    --mem "$CALIB_MEM" --conc "$CALIB_CONC" --need-gb 4 || {
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

cat <<'NOTE'

Submitted.  A submission is not a launch: check every arm came back with a
numeric job id, then watch cells appear rather than watching squeue.

  squeue -u $USER -n cal-cells
  for a in prod top_long easy_med_hard band_wide incl_k incl_k_wide flat_mid deep_first; do
    printf '%-14s %s\n' "$a" "$(ls $CALIB_EXP/results/$a/cells/task_*.csv 2>/dev/null | grep -c . )"
  done

When the queue drains:  python analyze_startup.py
NOTE
