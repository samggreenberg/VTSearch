#!/usr/bin/env bash
# State of the App (#4159): run the eval the way a user meets the app, on every
# coco_better cell, with the two production paths, and nothing tuned.
#
#   bash launch.sh prepare          # select cells, cache crops (one CPU job)
#   bash launch.sh cells            # the array: every class x band x path x seed
#   bash launch.sh status
#   bash analyze.sh                 # after the array: report, viewer, per-image influence
#
# What is pinned, and why (see .claude/skills/state-of-the-app/SKILL.md):
#   * the two production paths: SigLIP whole-image binary voting (`siglip`) and
#     DINOv3 region voting opened on SigLIP's text sort (`siglip+dinov3_patch`,
#     max_patch) -- the default coco_better embedders, named here so a changed
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

# One set of sessions per precision floor P (owner, 2026-10-01, #4408): the
# session itself depends on P (the line's count, the spot check, and acquisition
# once #4409 lands), so each P runs its own sessions. SOTA_FLOOR=0.1|0.5|0.9 sets
# their floor and suffixes the run dir (-p10, -p50, -p90); unset is the app's
# default floor in an unsuffixed dir.
if [[ -n "${SOTA_FLOOR:-}" && -n "${SOTA_BETA:-}" ]]; then
  echo "set SOTA_FLOOR or SOTA_BETA, not both: one preference draws the line" >&2; exit 2
fi
if [[ -n "${SOTA_FLOOR:-}" ]]; then
  export CALIB_MIN_PRECISION="$SOTA_FLOOR"
  _PTAG="-p$(python3 -c "import sys; print(round(float(sys.argv[1]) * 100))" "$SOTA_FLOOR")"
fi
# The balance (#4413): SOTA_BETA=0.5|1|2 runs every session at that F-beta
# balance (-b05, -b1, -b2); the review's sessions run once per beta.
if [[ -n "${SOTA_BETA:-}" ]]; then
  export CALIB_BETA="$SOTA_BETA"
  _PTAG="-b$(python3 -c "import sys; b=float(sys.argv[1]); print(f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}')" "$SOTA_BETA")"
fi
export SOTA_DATE="${SOTA_DATE:-$(date +%Y-%m-%d)${_PTAG:-}}"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/state-of-the-app/$SOTA_DATE}"
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS="siglip,siglip+dinov3_patch"
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
# Where the positives sit in each ranking (#4357): the test half at these clicks
# and at the last one before the spot check, and the full-label ceiling. Every
# line metric the report reads (precision of the kept set at each floor X, the
# oracle's recall at X, whether the check's range covered the truth) comes from
# it; the metric rows record the line only at the default floor.
export CALIB_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
export CALIB_JOB_NAME="${CALIB_JOB_NAME:-sota-$SOTA_DATE}"
# Sized on the first review (2026-09-23), NOT on vg_scale: a coco_better REGION
# run peaks at 66-70 GB (the 7.5 GB half-precision patch cell expands several-
# fold) and takes ~1 h; at 12 GB 12 of 23 died within minutes. A whole-image
# run is ~5 min and small, but one array carries both, so it is sized for the
# region runs. Expect the memory QOS, not the CPU count, to set the wall clock.
export CALIB_MEM="${CALIB_MEM:-80G}"
export CALIB_TIME="${CALIB_TIME:-4:00:00}"

# `subset "<class>,<class>,..."`: run only those classes (every band) out of an
# already-prepared grid, on the path SOTA_PATH names: `binary` (the default),
# `region` or `all`. Binary is the default because a review runs Binary Photo
# only unless the owner asks for Region (2026-09-26); before #4363 this mode
# queued both paths, so a smoke run silently carried 80 GB region cells. Indices are the grid's own, so the
# cells land exactly where the full run would put them and a later full run
# can skip them. The owner's rule: settle the presentation on a few classes,
# then widen classes and seeds together.
# A review's run directory defaults to TODAY. Adding cells to an existing review
# after midnight therefore needs SOTA_DATE pinned, or every task lands in a new,
# unprepared directory and dies in 0 s (576 of them did, 2026-09-24). Refuse it.
case "${1:-}" in subset|redo|cells|size)
  if [[ ! -s "$CALIB_EXP/results/prepare_info.json" ]]; then
    echo "no prepared grid at $CALIB_EXP (set SOTA_DATE=<the review's date>, or run prepare first)" >&2
    exit 3
  fi ;;
esac

if [[ "${1:-}" == "subset" ]]; then
  CLASSES="${2:?usage: launch.sh subset \"airplane,dining table,...\"}"
  SUBSET_PATH="${SOTA_PATH:-binary}"
  IDX=$(python3 - "$CALIB_EXP/results/prepare_info.json" "$CLASSES" "$CALIB_N_SEEDS" "$SUBSET_PATH" <<'PYIDX'
import json, sys
info = json.load(open(sys.argv[1]))["datasets"]["coco_better"]
keep = {c.strip() for c in sys.argv[2].split(",") if c.strip()}
n_seeds = int(sys.argv[3])
paths = {"binary": ("siglip",), "region": ("siglip+dinov3_patch",), "all": ("siglip", "siglip+dinov3_patch")}
if sys.argv[4] not in paths:
    raise SystemExit(f"SOTA_PATH must be one of {sorted(paths)}, got {sys.argv[4]!r}")
# array_cells order at CALIB_CELL_ORDER=seed: seed-major, then embedder, then category.
# Both embedders still count toward the seed's block size, so an index means the
# same cell whichever path is selected.
one, per_seed = [], 0
for emb in ("siglip", "siglip+dinov3_patch"):
    cats = info[emb]["selected_categories"]
    unknown = keep - {c.split("@")[0] for c in cats}
    if unknown:
        raise SystemExit(f"not in the grid: {sorted(unknown)}")
    if emb in paths[sys.argv[4]]:
        one += [per_seed + i for i, c in enumerate(cats) if c.split("@")[0] in keep]
    per_seed += len(cats)
print(",".join(str(s * per_seed + i) for s in range(n_seeds) for i in one))
PYIDX
  )
  echo "$IDX" > "$CALIB_EXP/subset-indices.txt"
  echo "subset: path=$SUBSET_PATH, $(tr ',' '\n' <<<"$IDX" | wc -l) runs"
  exec bash "$CALIB/launch_bands.sh" redo "$IDX"
fi

exec bash "$CALIB/launch_bands.sh" "$@"
