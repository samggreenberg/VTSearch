#!/usr/bin/env bash
# #4160 literal examples: rerun seed-0 cells of three classes with per-image dumps.
set -u
E=/expscratch/sgreenberg/size-vs-size-4160
cd /expscratch/sgreenberg/worktrees/vts-4160/scripts/experiments/calibration
source ../../../gridenv.sh >/dev/null 2>&1; source ../pile/pile_env.sh >/dev/null 2>&1
export CALIB_EXP=$E/examples CALIB_RESULTS=$E/examples CALIB_DATASETS=coco_better CALIB_COCO_BETTER_EMBEDDERS=siglip \
  CALIB_CATEGORY_MODE=all CALIB_N_SEEDS=20 CALIB_CELL_ORDER=seed CALIB_TRAIN_MIXES=equal,natural \
  CALIB_MIX_SHARES=$E/mix_shares.json CALIB_TEST_BANDS=all CALIB_TEST_BAND_AUROC=1 CALIB_MAX_STEPS=150 \
  CALIB_SAFE_THRESHOLDS=1 CALIB_REQUIRE_OPENING=text CALIB_REQUIRE_SEED_QUERY=1 CALIB_REPOOL_VARIANTS= \
  CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS= CALIB_PATCH_STYLES=max_patch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $E/examples/cells $E/examples/dumps
ln -sfn $E/results/prepare_info.json $E/examples/prepare_info.json
ln -sfn $E/results/crops $E/examples/crops
awk '{print $1}' $E/example_indices.txt | xargs -P 3 -I{} bash -c '
  cat=$(awk -v i={} "\$1==i{\$1=\"\"; sub(/^ /,\"\"); print}" '"$E"'/example_indices.txt | tr " " "_")
  VTS_DUMP_TEST_SCORES='"$E"'/examples/dumps VTS_DUMP_TAG="$cat" nice python run_cells.py --index {} > '"$E"'/examples/log_{}.txt 2>&1
  echo "done {} $cat exit $?"'
