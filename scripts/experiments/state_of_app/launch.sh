#!/usr/bin/env bash
# State of the App (#4159): run the eval the way a user meets the app, on every
# coco_quarry cell, with the two production paths, and nothing tuned.
#
#   bash launch.sh prepare          # select cells, cache crops (one CPU job)
#   bash launch.sh cells            # the array: every class x band x path x seed
#   bash launch.sh status
#   bash analyze.sh                 # after the array: report, viewer, per-image influence
#
# What is pinned, and why (see .claude/skills/state-of-the-app/SKILL.md):
#   * the two production paths: SigLIP whole-image binary voting (`siglip`) and
#     DINOv3 region voting opened on SigLIP's text sort (`siglip+dinov3_patch`,
#     max_patch) -- the default coco_quarry embedders, named here so a changed
#     default cannot silently change what "the app" means;
#   * shipped defaults for everything else: no repool/schedule/fold variants,
#     the fused threshold, the text opening (refused otherwise);
#   * every class at every band (CALIB_CATEGORY_MODE=all -> 144 cells);
#   * the supervised ceiling on both paths (skyline_train_full; the region
#     path's uses oracle boxes, #4159) so each cell reads text-only -> clicks ->
#     full labels.
#
# A dated directory per review, so this month's run never overwrites last month's.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB="$HERE/../calibration"

export SOTA_DATE="${SOTA_DATE:-$(date +%Y-%m-%d)}"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/state-of-the-app/$SOTA_DATE}"
export CALIB_DATASETS=coco_quarry
export CALIB_COCO_QUARRY_EMBEDDERS="siglip,siglip+dinov3_patch"
export CALIB_PATCH_STYLES=max_patch
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${SOTA_SEEDS:-1}"
export CALIB_MAX_STEPS="${SOTA_MAX_STEPS:-150}"
export CALIB_SKYLINE_ARMS=skyline_train_full
# launch_bands.sh turns an empty value into `all`. That only ADDS per-band FNR
# columns (BAND_COLUMNS) and moves no headline one, so it is left on: free
# context for "does this detector miss the other sizes" without it being a study.
export CALIB_TEST_BANDS=all
export CALIB_SAFE_THRESHOLDS=1
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_EMIT_PICKS=1
export CALIB_JOB_NAME="${CALIB_JOB_NAME:-sota-$SOTA_DATE}"
# The region cells dominate: ~13-20 min and up to ~5 GB each on vg_scale.
export CALIB_MEM="${CALIB_MEM:-12G}"
export CALIB_TIME="${CALIB_TIME:-6:00:00}"

exec bash "$CALIB/launch_bands.sh" "$@"
