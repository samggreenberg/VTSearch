#!/usr/bin/env bash
# Re-run selected benchmark cells with per-media prediction dumping on.
#
#   bash launch_errdump.sh wave1  135:coco_siglip_clock 177:coco_dinov3_clock
#   bash launch_errdump.sh vgbox  3:vgsmall_siglip_glasses 63:vgsmall_dinov3_glasses
#   bash launch_errdump.sh binary 21:vg_dinov3_sky_binary
#   bash launch_errdump.sh text   wave1
#
# Reuses the source run's prepare_info.json and crops by symlink, so the cell
# index, category selection and startup exemplar are identical to the original
# run - these are the same cells, not similar ones.  The cell's own log line
# ("cell N/TOTAL: dataset=... category=... seed=...") is the check that the
# index still means what it meant: compare it against the source run's log
# before believing a dump.
#
# The profile must reproduce the source run's grid exactly, because the index
# is a position in an enumerated list.  A profile that sets a different dataset
# or embedder list silently dumps a DIFFERENT cell under the tag you asked for.
set -euo pipefail

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$WT/scripts/experiments/calibration"
ROOT="${ERRDUMP_ROOT:-/expscratch/$USER/bench-errors}"
DUMP="$ROOT/dumps"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"

# No braces in this message: a `}` inside ${1:?...} ends the expansion early and
# the rest of it lands in PROFILE.
PROFILE="${1:?usage: launch_errdump.sh wave1|vgbox|binary|text  idx:tag ...}"
shift

# --- per-profile grid (must match the source run's sizing env) --------------
case "$PROFILE" in
text) ;;  # resolved below, after the source run is known
wave1)
  SRC=/expscratch/$USER/bench-overview
  GRID="CALIB_DATASETS=visual_genome_m,caltech101_m,coco_val"
  GRID="$GRID CALIB_VG_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
  GRID="$GRID CALIB_CALTECH_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
  GRID="$GRID CALIB_COCO_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
  GRID="$GRID CALIB_N_SEEDS=3 CALIB_N_PER_BAND=2 CALIB_N_CATEGORIES=6"
  GRID="$GRID CALIB_PATCH_STYLES=max_patch"
  ;;
vgbox)
  SRC=/expscratch/$USER/bench-vgbox2
  GRID="CALIB_DATASETS=vg_box_small,vg_box_medium,vg_box_large"
  GRID="$GRID CALIB_VGBOX_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
  GRID="$GRID CALIB_N_SEEDS=3 CALIB_N_CATEGORIES=10"
  GRID="$GRID CALIB_PATCH_STYLES=max_patch"
  ;;
binary)
  # The box-averse arm: dinov3_patch forced through the whole-image style, so
  # the only difference from its boxed twin is whether a box is dragged.
  SRC=/expscratch/$USER/bench-binary
  GRID="CALIB_DATASETS=visual_genome_m,coco_val"
  GRID="$GRID CALIB_VG_EMBEDDERS=dinov3_patch CALIB_COCO_EMBEDDERS=dinov3_patch"
  GRID="$GRID CALIB_N_SEEDS=3 CALIB_N_PER_BAND=2 CALIB_N_CATEGORIES=6"
  GRID="$GRID CALIB_PATCH_STYLES=whole_image"
  ;;
*)
  echo "unknown profile '$PROFILE' (wave1|vgbox|binary|text)" >&2; exit 1
  ;;
esac

# One results dir per profile.  `prepare_info.json` is a symlink into the source
# run and is READ AT RUNTIME, so two profiles sharing a results dir would flip it
# under each other's jobs and dump a different cell than the tag claims.
EXP="$ROOT/$PROFILE"
mkdir -p "$EXP/results/cells" "$DUMP" "$EXP/logs"

if [[ "$PROFILE" == "text" ]]; then
  # Zero-click side of the same question: what does a *typed* query flag?
  # Same dump schema, so error_report.py / label_noise.py read both.
  case "${1:?usage: launch_errdump.sh text wave1|vgbox}" in
  wave1) SRC=/expscratch/$USER/bench-overview ;;
  vgbox) SRC=/expscratch/$USER/bench-vgbox2 ;;
  *) echo "text: unknown source run '$1'" >&2; exit 1 ;;
  esac
  J=$(sbatch --parsable --job-name="errdump-text" --mem=48G --cpus-per-task=6 \
      --time=3:00:00 --partition=cpu --export=ALL \
      --output="$EXP/logs/text-%j.out" \
      --wrap="source $WT/gridenv.sh && export VTS_REPO=$WT \
        VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME \
        && cd $HERE && python text_baseline.py --results $SRC/results \
             --out $EXP/text_baseline_$1.csv --dump-dir $DUMP/text")
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "text SUBMIT FAILED (empty job id)" >&2; exit 1; }
  echo "text dumps ($1) -> job $J"
  exit 0
fi

ln -sfn "$SRC/results/prepare_info.json" "$EXP/results/prepare_info.json"
ln -sfn "$SRC/results/crops" "$EXP/results/crops"

ENVX="export CALIB_EXP=$EXP CALIB_RESULTS=$EXP/results"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$WT"
ENVX="$ENVX $GRID CALIB_MAX_STEPS=150"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
ENVX="$ENVX VTS_DUMP_TEST_SCORES=$DUMP"

for spec in "$@"; do
  idx="${spec%%:*}"; tag="${spec##*:}"
  # A patch cell is 11-13x a whole-image one, so give every dump the long limit.
  # Name carries the profile: two profiles can hold the same index, and a shared
  # job name breaks every per-name query, the completion waiter included.
  J=$(sbatch --parsable --job-name="errdump-$PROFILE$idx" --mem=48G --cpus-per-task=6 \
      --time=3:00:00 --partition=cpu --export=ALL \
      --output="$EXP/logs/err-$PROFILE-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export VTS_DUMP_TAG=$tag && cd $HERE && python run_cells.py --index $idx")
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "idx $idx ($tag) -> job $J"
  else
    echo "idx $idx SUBMIT FAILED (empty job id)" >&2
  fi
done
