#!/usr/bin/env bash
# Acquisition/reporting decoupling, REGION-VOTING environment — issue #2905.
#
# The third environment for `ACQUISITION_INCLUSION_OFFSET`, and the first that
# actually region-votes.  #2876 measured `coco_val x siglip2` (binary) and found
# an interior optimum at k=-3, which #2878 shipped as the global default.  #2877
# measured a second environment believing it was region voting; it was not
# (`visual_genome_m x siglip` carries no `patch_grid`, so `region_voting=True`
# silently fell back to whole-image training, whole-image scoring and the binary
# `cap50` blend).  Both measured environments are binary, and they disagree:
# -3 ships on COCO and FAILS its own ship rule on VG-siglip.
#
# So the open question is whether the offset should be gated by voting mode, and
# it cannot be answered without a genuine region-voting run.  This is that run.
#
#   ENVIRONMENT: visual_genome_m x dinov3_patch x max_patch
#
# It is the only VG arm carrying a patch grid, so the dragged ground-truth box
# is pooled over patches and a media's score is a max over its region nodes —
# an extreme-value statistic, a different score distribution for the cut to be
# realized on, and the regime docs/ML.md records as supplying MORE positive
# anchors than binary voting.  It blends under `slow_cap50` (the region schedule
# #2849 shipped), which is left to `production_schedule_for` rather than pinned.
#
# THE PREMISE IS ASSERTED, NOT ASSUMED.  `premise_check.py` in $CALIB_EXP reports
# patch_grid on 4193/4193 medias for this arm and 0/4193 for siglip.  Run it
# before launching; that one line is the whole difference between this study and
# #2877's.
#
# Arms are #2876/#2877's verbatim so all three environments are comparable:
#
#   prod      k_acq =  0   control - the pre-#2876 coupled behaviour
#   acq_m1    k_acq = -1   the only value passing in both prior environments
#   acq_m2    k_acq = -2   where #2877 found ranking benefit saturates
#   acq_m3    k_acq = -3   the shipped default
#   acq_m4    k_acq = -4   far end
#   acq_p2    k_acq = +2   FALSIFICATION arm - must make positives WORSE
#   rank_pin  cut pinned at the conformal path's own pool percentile (0.959)
#
# `prod` MUST name its 0 explicitly: the default is -3, so a launcher copied
# from a pre-#2878 template would run seven arms of the same thing.
#
# SEED COUNT IS DECLARED AT 48 AND NEVER CHANGED — the study RUNS 24.
#
# A cell's array index is `category_index * N_SEEDS + seed`, so moving the seed
# count remaps every index and makes every earlier cell unresumable.  #2877's
# lesson was to re-derive n from a pilot rather than inherit a seed count, but
# the sizing input (the paired SD on `final_cost`) is not knowable until cells
# exist, and a pilot big enough to measure it is most of the run.
#
# So the seed count is declared at the largest value this study could ever need
# and a seed RANGE is run as an array index subset (`--seeds 24`, `--seeds
# 24-47`).  24 seeds x 23 categories = 552 pairs, against the n≈473 #2877
# derived for a ±0.010 half-width at its SD of 0.111.  If the realized SD here
# is larger and the CI comes back too wide to certify, the top-up is
# `--seeds 24-47` and every cell already on disk still counts — the mistake
# #2877 could not undo becomes a second submission.
#
# "Still counts" is about the INDEX mapping, and it stops at the seeding fix:
# cells written before #3269 seeded from a crop, and cells written now open on a
# text sort (see the environment note below), so the two cannot be pooled into
# one number however stable the indices are.  `_cells_io.assert_one_opening`
# refuses that pooling at analysis time rather than leaving it to be noticed.
#
# A seed's trajectory does not depend on the declared count: `seed` is passed
# to the simulator directly and the exemplar is `candidates[seed % len]`.  Only
# the index mapping moves, which is exactly what this pins.
#
# Prepare is REUSED (no GPU stage): the dinov3_patch pickle and exemplar crops
# from the #2861 anchor-rate run, so the categories are the same 23.  The pair
# reads that same pickle (`pickle_name` resolves the LEARN half) and opens the
# `visual_genome_m__siglip.pkl` beside it, which that run also embedded.  What
# it cannot reuse is a prepare_info.json keyed by the old bare name: the grid
# enumerates by embedder, so a missing key means the arm contributes ZERO cells
# rather than failing.  Re-run prepare for the paired name -- it reuses the
# cached pickles, so it costs no encoder time -- and preflight check 15 is what
# says whether you did.
#
# Usage:
#   bash launch_acq_region_2905.sh --seeds 24                  # the study
#   bash launch_acq_region_2905.sh --seeds 24-47               # the top-up
#   bash launch_acq_region_2905.sh --seeds 2 --arms prod,acq_m3  # a smoke pair
set -uo pipefail

SEEDS_TO_RUN=""
ARMS_TO_RUN="prod acq_m1 acq_m2 acq_m3 acq_m4 acq_p2 rank_pin"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS_TO_RUN="$2"; shift 2 ;;
    --arms) ARMS_TO_RUN="${2//,/ }"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

export VTS_REPO=${VTS_REPO:-/exp/$USER/projects/vts-acq-region}
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# NOT under /exp: that 50G quota is ~92% full and 13G of it is a .venv every
# gridenv.sh activates, so it cannot be reclaimed (#2897).
export CALIB_EXP="${CALIB_EXP:-/home/hltcoe/$USER/experiments/acq-region-2905}"

# --- environment: the one VG arm that genuinely region-votes ---
# ...as the PAIR `siglip+dinov3_patch` (#3278).  This study has a single
# environment, so nothing inside it can be confounded -- but its whole purpose
# is a COMPARISON ACROSS environments: k_acq=-3 ships on COCO-siglip and fails
# its own rule on VG-siglip, and this run is the third, region-voting reading
# that decides whether the offset should be gated by voting mode.  The other two
# were measured on siglip, which since #3269 opens on a typed query; bare
# `dinov3_patch` has no text tower and would open on three random known-goods,
# so the environment that breaks the tie would have differed from both of them
# in two ways at once.  k_acq is an offset applied to a RANK POSITION in the
# acquisition ranking, which is precisely the object the opening creates.
export CALIB_DATASETS=visual_genome_m
export CALIB_VG_EMBEDDERS=siglip+dinov3_patch
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_PATCH_STYLES=max_patch
export CALIB_REPOOL_VARIANTS=
export CALIB_SCHEDULE_VARIANTS=
export CALIB_MAX_STEPS=100
export CALIB_N_SEEDS=48   # DECLARED; see the header. The study runs --seeds 24.
# The head a live detector actually has, because the acquisition offset is only worth
# measuring on the model users actually get.  NOT `linear`: that was production
# when this launcher was written, and PR #3198 moved `PRODUCTION_HEAD` to the
# linear SVM, so the pin outlived the thing it was pinning -- preflight check 12
# has been failing on it since.  Named rather than left unset, which is what
# `launch_transfer_2883.sh` settled on: the run's head is then readable from the
# launcher instead of from a default three modules away.
export CALIB_HEAD=linear_svm
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=0
# CALIB_BLEND_SCHEDULE deliberately unset: the run must live under whatever
# schedule production picks for its voting mode.  #2849 made that per-mode, so
# region voting gets `slow_cap50` where the two binary environments got `cap50`.
# Pinning one here would measure the offset under a schedule no region-voting
# user runs — and the blend schedule being already mode-gated while the offset
# is not is half of what #2905 exists to resolve.

# --- ops: cpu partition, single-threaded cells.  The binding constraint is the
# `cpu_limit` QOS (cpu=240/user, 2 charged per task) => 120 concurrent.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM=${CALIB_MEM:-8G}
export CALIB_CPUS=1
export CALIB_TIME=${CALIB_TIME:-4:00:00}
export CALIB_CONC=${CALIB_CONC:-120}

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

# Analysis is cross-arm and runs once, by hand, after all arms drain.
export CALIB_ANALYZE=${CALIB_ANALYZE:-noop.py}

RESULTS_ROOT="${RESULTS_ROOT:-$CALIB_EXP/results}"
LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$RESULTS_ROOT"

if [[ ! -f "$CALIB_EXP/results/prepare_info.json" ]]; then
  echo "ERROR: no prepare_info.json at $CALIB_EXP/results" >&2; exit 1
fi

# `--require-region-voting` is the gate this study exists because of: it opens
# the pickle and refuses to submit unless the patch geometry region voting needs
# is actually there.  #2877 had every other check green and still measured a
# binary environment for a region-voting question.
if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 4 \
    --arms "${ARMS_TO_RUN// /,}" \
    --require-region-voting "$CALIB_DATASETS:$CALIB_VG_EMBEDDERS" || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

# --- how many cells, and which ---------------------------------------------
# Count against the shared prepare dir: the per-arm dirs do not have their
# prepare_info.json symlink until the arm loop below creates it.
N=$(cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1; \
    export CALIB_RESULTS="$CALIB_EXP/results"; python run_cells.py --print-cells 2>/dev/null | tail -1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
  echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
fi
NCAT=$(( N / CALIB_N_SEEDS ))

# A seed subset is an array index subset: index = category*N_SEEDS + seed.
if [[ -n "$SEEDS_TO_RUN" ]]; then
  if [[ "$SEEDS_TO_RUN" == *-* ]]; then
    SLO="${SEEDS_TO_RUN%%-*}"; SHI="${SEEDS_TO_RUN##*-}"
  else
    SLO=0; SHI=$(( SEEDS_TO_RUN - 1 ))
  fi
  if (( SLO < 0 || SHI >= CALIB_N_SEEDS || SLO > SHI )); then
    echo "ERROR: --seeds $SEEDS_TO_RUN is outside 0..$((CALIB_N_SEEDS-1))" >&2; exit 2
  fi
  spec=""
  for ((c = 0; c < NCAT; c++)); do
    lo=$(( c * CALIB_N_SEEDS + SLO )); hi=$(( c * CALIB_N_SEEDS + SHI ))
    spec+="${spec:+,}${lo}-${hi}"
  done
  ARRAY_SPEC="$spec"
  NPER=$(( NCAT * (SHI - SLO + 1) ))
  echo "cells: $NCAT categories x seeds ${SLO}..${SHI} (of $CALIB_N_SEEDS declared) = $NPER per arm"
else
  ARRAY_SPEC="0-$((N-1))"
  NPER="$N"
  echo "cells: $NCAT categories x $CALIB_N_SEEDS seeds = $N per arm"
fi

# arm -> "ACQ_INCLUSION_OFFSET ACQ_RANK_PERCENTILE"  ("-" = unset)
declare -A ARMS=(
  [prod]="0 -"
  [acq_m1]="-1 -"
  [acq_m2]="-2 -"
  [acq_m3]="-3 -"
  [acq_m4]="-4 -"
  [acq_p2]="2 -"
  [rank_pin]="0 0.959"
)

for arm in $ARMS_TO_RUN; do
  read -r inc pct <<<"${ARMS[$arm]}"
  export CALIB_ACQ_INCLUSION_OFFSET=""
  export CALIB_ACQ_RANK_PERCENTILE=""
  [[ "$inc" != "-" ]] && export CALIB_ACQ_INCLUSION_OFFSET="$inc"
  [[ "$pct" != "-" ]] && export CALIB_ACQ_RANK_PERCENTILE="$pct"
  export CALIB_RESULTS="$RESULTS_ROOT/$arm"
  mkdir -p "$CALIB_RESULTS/cells"
  ln -sfn "$CALIB_EXP/results/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_EXP/results/crops" "$CALIB_RESULTS/crops"

  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
  ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
  ENVX="$ENVX CALIB_DATASETS=$CALIB_DATASETS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS"
  ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS="
  ENVX="$ENVX CALIB_MAX_STEPS=$CALIB_MAX_STEPS CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_HEAD=$CALIB_HEAD"
  ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS CALIB_ANCHORED=$CALIB_ANCHORED"
  ENVX="$ENVX CALIB_ACQ_INCLUSION_OFFSET=$CALIB_ACQ_INCLUSION_OFFSET"
  ENVX="$ENVX CALIB_ACQ_RANK_PERCENTILE=$CALIB_ACQ_RANK_PERCENTILE"

  J=$(sbatch --parsable --job-name="acq905-$arm" --array="$ARRAY_SPEC%$CALIB_CONC" \
      --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" --time="$CALIB_TIME" \
      --partition="$CALIB_PARTITION" --export=ALL \
      --output="$LOGS/${arm}-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
  # A submission is not a launch: --parsable returns an EMPTY id when the submit
  # filter refuses a job, and dependents then die with an unrelated message
  # ("Job dependency problem").  Both A/B arms of #2897 silently failed this way.
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "=== arm $arm (k='${CALIB_ACQ_INCLUSION_OFFSET}' pct='${CALIB_ACQ_RANK_PERCENTILE}') job=$J"
    echo "$J" > "$LOGS/.jobid_$arm"
  else
    echo "ARM $arm SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
  fi
done
