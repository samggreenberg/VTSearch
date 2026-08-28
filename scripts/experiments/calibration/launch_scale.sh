#!/usr/bin/env bash
# The scale study (#3156): one class list, three box-size bands.
#
#   bash launch_scale.sh prepare      # exemplar crops for all 36 cells
#   bash launch_scale.sh size 0,40    # time one whole-image and one patch cell
#   bash launch_scale.sh cells        # the full array
#   bash launch_scale.sh status
#
# The question is whether detection cost rises as the target shrinks, and
# whether region voting's advantage depends on target size. `vg_box_*` cannot
# answer it: those sets band each category by its median box, so their
# vocabularies are disjoint and a small-vs-large gap confounds size with class
# identity. Here the twelve classes are held fixed and only the band moves,
# paired on identical negatives at identical prevalence.
#
# Shipped defaults only, as in launch_bench.sh: the contrast under test is the
# BAND, so any other knob left non-default would be a second, uncontrolled one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/scale-3156}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale}"
# siglip is the shipped default and votes binary; siglip2_l is the premium
# whole-image end; the region arm is the PAIR `siglip+dinov3_patch` -- DINOv3
# supplies the patch grid that makes region voting real, SigLIP supplies the
# text sort the run opens on.
#
# Bare `dinov3_patch` was the region arm until #3276 and it could not open the
# way the app does: DINOv3 has no text tower, so its cells fell back to the
# three-random-known-goods start while both whole-image arms opened on a typed
# query. The study's headline axis is VOTING MODE, and a seeding difference sat
# inside it -- an arm-dependent difference, so unlike the #3156 seeding fix it
# does not cancel in a contrast. The pair removes it: all three arms now open on
# the same SigLIP text sort of the same 7749 medias, and only the space the
# detector learns in differs.
#
# All three columns are already embedded in the pile, so a column costs training
# and inference only -- no encoder time; the pair adds one 26 MB pickle open per
# cell. And the encoder here is a BLOCKING factor, not the contrast: the
# question is whether the band effect holds across encoders, so a second
# whole-image encoder replicates the finding rather than competing with the
# first. ("siglip vs siglip2_l is unresolvable on cost at three seeds" is a fact
# about comparing the two encoders, and says nothing about whether each one
# shows the same size penalty.)
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip2_l,siglip+dinov3_patch}"
# Every category is a designated cell; selecting a subset would discard the
# design, and prevalence-spreading is meaningless when prevalence is 0.0250
# everywhere by construction.
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-3}"
# Walk every environment at seed 0, then every environment at seed 1, and so on.
# A SLURM array dispatches roughly in index order, so if this run is cut short
# it loses SEEDS uniformly rather than whole categories: the design stays intact
# and only the standard errors widen, which a report can simply state.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"

# Declare the opening this study is FOR (#3278), rather than leaving it to be
# decided per cell by whether a query happens to exist.  The pair guards itself
# -- `siglip+dinov3_patch` exists to take the text sort, so falling back raises
# -- but the WHOLE-IMAGE arms have no such guard: a category with no typed query
# would open on three random known-goods while the others opened on a sort, and
# that is the same arm-dependent seeding confound the pair was added to remove,
# entering from the other side.  `all` categories are designated here, so the
# `REQUIRE_SEED_QUERY` filter drops nothing: EXPERIMENT_QUERIES["vg_scale"]
# covers every `<class>@<band>` cell.  It is a guard, not a selection knob.
export CALIB_REQUIRE_OPENING="${CALIB_REQUIRE_OPENING:-text}"
export CALIB_REQUIRE_SEED_QUERY="${CALIB_REQUIRE_SEED_QUERY:-1}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"

# Sized from `size 0,72` on THIS configuration -- the text-seeded, paired grid --
# and cross-checked against the SAME cell in the crop-seeded run it replaces
# (job 549465 index 4320: backpack@large, seed 0, max_patch).
#
#   whole_image cell (index 0)   47s     MaxRSS 0.50 G   0.7 cores
#   max_patch   cell (index 72)  18m02s  MaxRSS 5.02 G   1.0 cores
#   ...the same cell, crop-seeded: 11m28s MaxRSS 10.36 G
#
# The opening moved BOTH numbers, in opposite directions. Slower, because a text
# sort finds positives immediately and the detector then trains on more of them.
# Cheaper, because the crop path scored the query against every PATCH of every
# media (7749 x ~197 x 768) where a text sort scores whole-image vectors only.
#
# For the array's shape, what matters is the whole region arm, not one cell.
# Across the 532 region cells that finished in job 549465:
#
#   elapsed  p50 627s   p90 1206s  max 1345s
#   MaxRSS   p50 4.9 G  p90 7.8 G  max 9.9 G
#
# So 12G clears the worst region cell ever observed on this grid, on a path that
# has since become the cheaper of the two.
#
# Two things follow, and both were wrong before:
#
# * 6 CPUs was waste. user+sys is 17m47s against 18m02s wall -- the cell is
#   effectively SINGLE-THREADED, so the extra cores bought nothing and cost
#   quota. `cpu_limit` is cpu=240, so 6 CPUs/task caps the array at 40 tasks;
#   2 CPUs/task lifts that to 120 and lets memory be the binding constraint
#   instead, which is the one that reflects real usage.
# * 16G was 3.2x the measured peak. Memory is the other half of `cpu_limit`
#   (~1074G), so an oversized --mem is a direct concurrency cut: 12G allows 89
#   concurrent tasks where 16G allows 67. 12G keeps 2.4x headroom over the
#   measured 5.02G AND is preflight check 7b's floor for patch cells, which
#   exists because UNDER-sizing killed 74 of 108 cells in #3156.
#
# CONC is then bounded by preflight check 8, not by the raw cap: an array may
# claim at most 90% of the ~1074G allowance before your OWN later jobs -- the
# analyzer, a `redo`, a diagnostic -- start queueing behind it in
# QOSMaxMemoryPerUser. 85 asks for 1020G (95%) and is refused, correctly; 70
# asks for 840G (78%, 140 CPUs) and leaves 234G free for exactly those jobs.
# Note the whole-image cells hold a 12G slot while needing 0.5G, which is the
# price of running one array over a grid with two cost classes.
MEM="${CALIB_MEM:-12G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-6:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-70}"
JOB_NAME="${CALIB_JOB_NAME:-scale-$(basename "$CALIB_EXP")}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
# Named explicitly rather than left to --export=ALL: the launcher computes the
# cell COUNT and each task computes the cell LIST, and the two must enumerate
# the same grid.  A knob that reaches one but not the other maps an index to a
# different cell.
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
# `REQUIRE_SEED_QUERY` filters CATEGORIES, so it reaches the cell list: it has to
# travel with the rest or a task enumerates a different grid than the launcher
# counted.  `REQUIRE_OPENING` is checked per cell and travels beside it.
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_REQUIRE_SEED_QUERY=$CALIB_REQUIRE_SEED_QUERY"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms exactly this way).
submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
    return 1
  fi
}

case "${1:-status}" in
prepare)
  # The dataset's own review coverage is a precondition, not a detail: a cell
  # whose reviewed images have been rebuilt away is measuring labels nobody
  # checked (see scripts/experiments/lessons/).
  ( cd "$WT/scripts/experiments/pile" && python check_review_coverage.py ) || {
    echo "REVIEW COVERAGE FAILED — refusing to launch a study on it" >&2; exit 3; }
  # Assert the arm the study is FOR actually exists. A patch embedder on a
  # dataset the config forgot to mark boxed runs whole_image without complaint,
  # which is indistinguishable from success in every output.
  ( cd "$HERE" && python - <<'PYCHK' ) || { echo "REGION VOTING IS OFF for the patch arm — refusing to launch" >&2; exit 4; }
import os
import sys
sys.path.insert(0, os.getcwd())
import experiment_config as cfg
ds = cfg.DATASETS[0]
bad = [e for e in cfg.embedders_for_dataset(ds)
       if cfg.is_patch_embedder(e) and not cfg.region_voting_for(ds, e)]
if bad:
    print(f"  {ds}: patch embedders {bad} would run whole_image", file=sys.stderr)
    raise SystemExit(1)
for e in cfg.embedders_for_dataset(ds):
    print(f"  {ds} x {e}: styles={cfg.styles_for(ds, e)} region_voting={cfg.region_voting_for(ds, e)} "
          f"learn={cfg.learn_embedder(e)} text={cfg.text_embedder(e)}")
PYCHK
  echo "CALIB_EXP=$CALIB_EXP  datasets=$CALIB_DATASETS  embedders=$CALIB_VGSCALE_EMBEDDERS  seeds=$CALIB_N_SEEDS"
  submit prepare --job-name=scale-prep --mem=96G --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py"
  ;;

size)
  IDXS="${2:?usage: launch_scale.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="scale-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $HERE && time python run_cells.py --index $idx"
  done
  ;;

cells)
  N=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  PATCH_FLAG=""
  case "$CALIB_VGSCALE_EMBEDDERS" in *_patch*) PATCH_FLAG="--patch" ;; esac
  # Both premises this study now rests on, asserted before the array rather
  # than discovered in the rows afterwards: every cell opens on a TYPED QUERY
  # (check 14, which for a paired arm probes the text half's tower and the text
  # pickle's coverage), and the region arm really region-votes (check 6, which
  # reads the learn half's pickle for a patch grid).
  REGION_FLAG=""
  case "$CALIB_VGSCALE_EMBEDDERS" in
    *_patch*) REGION_FLAG="--require-region-voting ${CALIB_DATASETS%%,*}:$(
      tr ',' '\n' <<<"$CALIB_VGSCALE_EMBEDDERS" | grep -- '_patch' | head -1)" ;;
  esac
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" $PATCH_FLAG \
    --require-text-seed $REGION_FLAG || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

redo)
  # Re-run a specific set of array indices. A failed task leaves its PREVIOUS
  # output in place, so a partial re-run silently mixes two runs' cells unless
  # the stale files are deleted first -- delete them, then pass the indices
  # here. `--mem` must be sized from a cell of the SAME KIND that failed: a
  # max_patch cell peaks near 9G where a whole_image one peaks under 1G, and
  # sizing the array from the wrong kind is what produced the OOMs this
  # subcommand exists to repair.
  IDXS="${2:?usage: launch_scale.sh redo <comma-separated indices>}"
  echo "re-running indices: $IDXS (mem=$MEM)"
  submit redo --job-name="$JOB_NAME-redo" --array="$IDXS%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/redo-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E "scale|JOBID" || true
  echo "=== cells written ==="
  ls "$CALIB_RESULTS/cells" 2>/dev/null | wc -l
  ;;
*)
  echo "usage: launch_scale.sh {prepare|size <idx>|cells|redo <idx-list>|status}" >&2; exit 1 ;;
esac
