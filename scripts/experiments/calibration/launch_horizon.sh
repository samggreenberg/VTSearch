#!/usr/bin/env bash
# Re-run a finished benchmark grid with a LONGER VOTING HORIZON.
#
#   bash launch_horizon.sh size   250 vgbox 3,63     # time one cheap + one patch cell
#   bash launch_horizon.sh cells  250 wave1          # the whole grid at 250 votes
#   bash launch_horizon.sh cells  250 vgbox
#   bash launch_horizon.sh cells  250 binary
#   bash launch_horizon.sh status 250
#
# It reuses the source study's `prepare_info.json` and exemplar crops by symlink,
# so the category selection, the seeds and the startup exemplar are identical and
# the longer run is the *same cells carried further* rather than a new draw. Steps
# 1..150 should therefore reproduce the original run bit-for-bit; `status` prints
# the check, because "the same cells, longer" is a premise worth asserting rather
# than assuming.
#
# That premise is also why the arms here are NOT paired (#3278).  Most launchers
# with a bare `dinov3_patch` arm now run `siglip+dinov3_patch`, so that every arm
# of one study opens the same way; this is not a study but a continuation of one,
# read against cells the source run already wrote.  Its grid is keyed to that
# run's `prepare_info.json` (renaming an arm drops it from the enumeration and
# shifts every later index), so each profile keeps the source's arms verbatim and
# declares the mix with `CALIB_REQUIRE_OPENING` instead.  Re-pair a profile only
# when its SOURCE study is re-launched paired.
#
# Note the bit-for-bit claim has a seam of its own: cells re-run today open on a
# text sort wherever they can (#3269), and the runs on disk were crop-seeded.  So
# until a source grid is re-run, `status`'s reproduction check is comparing across
# that fix and will not match -- the horizon run is the same CELLS, not the same
# trajectory.  Its own cells stay internally consistent (they land in this
# profile's results dir, not the source's), which is what keeps
# `_cells_io.assert_one_opening` quiet about them.
#
# #3400 opened a second seam of the same kind: `CALIB_SAFE_THRESHOLDS` defaulted
# to the #2781-era unfused control, so every benchmark grid written before that
# ran on a threshold the app cannot produce.  A continuation is pinned to the
# SHIPPED path below, which is the right threshold and the wrong one for
# reproducing a pre-#3400 grid -- set `CALIB_SAFE_THRESHOLDS=0` to continue one
# of those verbatim, and read the difference as the seam it is.
#
# Per-step cost RISES with the vote count (each Good vote adds patch rows to the
# training set): measured at 3.7 s/step early against 7.3 s/step at vote 150 for a
# patch cell, so a horizon extension is superlinear in wall time and has to be
# sized from a real cell at the NEW horizon, not scaled from the old one.
set -euo pipefail

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"

MODE="${1:?usage: launch_horizon.sh size|cells|status HORIZON [profile] [indices]}"
HORIZON="${2:?horizon (max votes per run), e.g. 250}"
PROFILE="${3:-}"
ROOT="${HORIZON_ROOT:-/expscratch/$USER/bench-h$HORIZON}"

# --- per-profile grid: must match the source study's sizing env exactly, or the
# --- cell index means something different than the source run's log says.
profile_env() {
  case "$1" in
  wave1)
    SRC=/expscratch/$USER/bench-overview
    # Both openings: SigLIP arms take a typed query, the bare DINOv3 arm takes
    # three random known-goods.  Declared so the mix reads as the mirror it is.
    OPENING=mixed
    GRID="CALIB_DATASETS=visual_genome_m,caltech101_m,coco_val"
    GRID="$GRID CALIB_VG_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
    GRID="$GRID CALIB_CALTECH_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
    GRID="$GRID CALIB_COCO_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
    GRID="$GRID CALIB_N_SEEDS=3 CALIB_N_PER_BAND=2 CALIB_N_CATEGORIES=6"
    GRID="$GRID CALIB_PATCH_STYLES=max_patch"
    ;;
  vgbox)
    SRC=/expscratch/$USER/bench-vgbox2
    OPENING=mixed
    GRID="CALIB_DATASETS=vg_box_small,vg_box_medium,vg_box_large"
    GRID="$GRID CALIB_VGBOX_EMBEDDERS=siglip,siglip2_l,dinov3_patch"
    GRID="$GRID CALIB_N_SEEDS=3 CALIB_N_CATEGORIES=10"
    GRID="$GRID CALIB_PATCH_STYLES=max_patch"
    ;;
  binary)
    SRC=/expscratch/$USER/bench-binary
    # Uniform: every arm is bare DINOv3, so every cell takes the known-good
    # start.  Pinned, so pairing an arm here fails instead of re-seeding a
    # continuation of cells that started the other way.
    OPENING=known_good
    GRID="CALIB_DATASETS=visual_genome_m,coco_val"
    GRID="$GRID CALIB_VG_EMBEDDERS=dinov3_patch CALIB_COCO_EMBEDDERS=dinov3_patch"
    GRID="$GRID CALIB_N_SEEDS=3 CALIB_N_PER_BAND=2 CALIB_N_CATEGORIES=6"
    GRID="$GRID CALIB_PATCH_STYLES=whole_image"
    ;;
  *)
    echo "unknown profile '$1' (wave1|vgbox|binary)" >&2; exit 1 ;;
  esac
  EXP="$ROOT/$PROFILE"
  RESULTS="$EXP/results"
  ENVX="export CALIB_EXP=$EXP CALIB_RESULTS=$RESULTS"
  ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
  ENVX="$ENVX VTS_REPO=$WT $GRID CALIB_MAX_STEPS=$HORIZON"
  ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
  ENVX="$ENVX CALIB_SAFE_THRESHOLDS=${CALIB_SAFE_THRESHOLDS:-1}"
  ENVX="$ENVX CALIB_REQUIRE_OPENING=$OPENING"
}

link_prepare() {
  mkdir -p "$RESULTS/cells" "$EXP/logs"
  ln -sfn "$SRC/results/prepare_info.json" "$RESULTS/prepare_info.json"
  ln -sfn "$SRC/results/crops" "$RESULTS/crops"
}

cell_count() {
  (cd "$HERE" && env $(echo "$GRID") CALIB_RESULTS="$RESULTS" VTS_REPO="$WT" \
      python run_cells.py --print-cells 2>/dev/null | tail -1)
}

case "$MODE" in
size)
  IDXS="${4:?usage: launch_horizon.sh size HORIZON PROFILE idx[,idx...]}"
  profile_env "$PROFILE"
  RESULTS="$EXP/sizing"
  ENVX="${ENVX/CALIB_RESULTS=$EXP\/results/CALIB_RESULTS=$EXP\/sizing}"
  link_prepare
  for idx in ${IDXS//,/ }; do
    J=$(sbatch --parsable --job-name="h$HORIZON-size$idx" --mem=48G --cpus-per-task=6 \
        --time=4:00:00 --partition=cpu --export=ALL \
        --output="$EXP/logs/size-$idx-%j.out" \
        --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $idx")
    [[ "$J" =~ ^[0-9]+$ ]] && echo "size idx $idx -> job $J" || echo "size idx $idx SUBMIT FAILED (empty job id)" >&2
  done
  ;;

cells)
  profile_env "$PROFILE"
  link_prepare
  N=$(cell_count)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  MEM="${CALIB_MEM:-48G}"; CPUS="${CALIB_CPUS:-6}"; CONC="${CALIB_CONC:-24}"
  TIME="${CALIB_TIME:-8:00:00}"
  JOB_NAME="h$HORIZON-$PROFILE"
  echo "cells: $N (array 0-$((N-1))%$CONC), horizon $HORIZON, exp $EXP"
  bash "$WT/scripts/experiments/preflight.sh" --exp "$EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" \
    --reuse-prepare "$SRC/results" || { echo "PREFLIGHT FAILED" >&2; exit 2; }
  J=$(sbatch --parsable --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --partition=cpu --export=ALL \
      --output="$EXP/logs/cells-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "$JOB_NAME SUBMIT FAILED (empty job id) - NOT LAUNCHED" >&2; exit 1; }
  echo "$J" > "$EXP/logs/.jobid"
  echo "$JOB_NAME -> job $J"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R"
  for p in wave1 vgbox binary; do
    d="$ROOT/$p/results/cells"
    [ -d "$d" ] || continue
    # find, not `ls "$d"/task_*.csv`: past ~25k cells the expanded glob exceeds
    # ARG_MAX, ls dies with "Argument list too long", and `grep -vc` turns that
    # into a count of 0 - a finished grid reporting as "0 cell files".
    # `! -name '*__*'`, not a list of the side frames by name: the by-name form
    # this used to have listed three of the five run_cells.py writes, so every
    # count here was inflated by the __picks and __fitq frames.  The `__` is the
    # convention (`_cells_paths.main_frame_files` filters on the same thing).
    files=$(find "$d" -maxdepth 1 -name 'task_*.csv' ! -name '*__*' 2>/dev/null | wc -l)
    rows=0
    for f in "$d"/task_*.csv; do
      case "$f" in *__*) continue;; esac
      [ -f "$f" ] || continue
      [ "$(wc -l < "$f")" -gt 1 ] && rows=$((rows+1))
    done
    # A header-only CSV passes `-size +0`, so count DATA ROWS, not bytes.
    echo "$p: $files cell files, $rows with data"
  done
  ;;

*)
  echo "unknown mode '$MODE' (size|cells|status)" >&2; exit 1 ;;
esac
