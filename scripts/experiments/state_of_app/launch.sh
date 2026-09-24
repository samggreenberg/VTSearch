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
# Two passes for the region path (measured 2026-09-23): its CLICKS peak at ~15 GB
# but the full-label ceiling, trained on every patch of every sim negative once
# at the end of the run, peaks at ~70 GB. The ceiling reads only the seed's
# sim/test split, never the votes, so it can run on its own and let the clicks
# run ~4x as wide under the per-user memory QOS.
#   SOTA_PASS=both        clicks + ceiling in one run (the default; 80 GB)
#   SOTA_PASS=trajectory  clicks only, no ceiling (24 GB)
#   SOTA_PASS=ceiling     the ceiling only (1 click), into $CALIB_EXP/ceiling/results
case "${SOTA_PASS:-both}" in
  both) ;;
  trajectory)
    export CALIB_SKYLINE_ARMS=""
    export CALIB_MEM="${CALIB_MEM:-24G}" ;;
  ceiling)
    export CALIB_MAX_STEPS=1
    export CALIB_RESULTS="$CALIB_EXP/ceiling/results"
    mkdir -p "$CALIB_RESULTS/cells"
    ln -sfn "$CALIB_EXP/results/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
    ln -sfn "$CALIB_EXP/results/crops" "$CALIB_RESULTS/crops" ;;
  *) echo "SOTA_PASS must be both, trajectory or ceiling" >&2; exit 2 ;;
esac
# launch_bands.sh turns an empty value into `all`. That only ADDS per-band FNR
# columns (BAND_COLUMNS) and moves no headline one, so it is left on: free
# context for "does this detector miss the other sizes" without it being a study.
export CALIB_TEST_BANDS=all
export CALIB_SAFE_THRESHOLDS=1
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_EMIT_PICKS=1
export CALIB_JOB_NAME="${CALIB_JOB_NAME:-sota-$SOTA_DATE}"
# Sized on the first review (2026-09-23), NOT on vg_scale: a coco_quarry REGION
# run peaks at 66-70 GB (the 7.5 GB half-precision patch cell expands several-
# fold) and takes ~1 h; at 12 GB 12 of 23 died within minutes. A whole-image
# run is ~5 min and small, but one array carries both, so it is sized for the
# region runs. Expect the memory QOS, not the CPU count, to set the wall clock.
export CALIB_MEM="${CALIB_MEM:-80G}"
export CALIB_TIME="${CALIB_TIME:-4:00:00}"

# `subset "<class>,<class>,..."`: run only those classes (every band, both
# paths) out of an already-prepared grid. Indices are the grid's own, so the
# cells land exactly where the full run would put them and a later full run
# can skip them. The owner's rule: settle the presentation on a few classes,
# then widen classes and seeds together.
if [[ "${1:-}" == "subset" ]]; then
  CLASSES="${2:?usage: launch.sh subset \"airplane,dining table,...\"}"
  IDX=$(python3 - "$CALIB_EXP/results/prepare_info.json" "$CLASSES" "$CALIB_N_SEEDS" <<'PYIDX'
import json, sys
info = json.load(open(sys.argv[1]))["datasets"]["coco_quarry"]
keep = {c.strip() for c in sys.argv[2].split(",") if c.strip()}
n_seeds = int(sys.argv[3])
# array_cells order at CALIB_CELL_ORDER=seed: seed-major, then embedder, then category.
one, per_seed = [], 0
for emb in ("siglip", "siglip+dinov3_patch"):
    cats = info[emb]["selected_categories"]
    unknown = keep - {c.split("@")[0] for c in cats}
    if unknown:
        raise SystemExit(f"not in the grid: {sorted(unknown)}")
    one += [per_seed + i for i, c in enumerate(cats) if c.split("@")[0] in keep]
    per_seed += len(cats)
print(",".join(str(s * per_seed + i) for s in range(n_seeds) for i in one))
PYIDX
  )
  echo "$IDX" > "$CALIB_EXP/subset-indices.txt"
  exec bash "$CALIB/launch_bands.sh" redo "$IDX"
fi

exec bash "$CALIB/launch_bands.sh" "$@"
