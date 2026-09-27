#!/usr/bin/env bash
# Size vs size (#4160): "if class C were size X in training and size Y in
# testing, we do this well", for X, Y in {S, M, L, SML}.
#
#   srun -p cpu --mem=8G -c 2 -t 60 bash launch.sh prepare   # mix shares + crops
#   bash launch.sh size <idx,idx>                            # time a pure and a mix cell
#   bash launch.sh cells                                     # the array
#   bash launch.sh status
#
# Rows of the table are TRAINING sizes, one arm each:
#   S, M, L       the pure cells, `<class>@small|medium|large`
#   SML (equal)   `<class>@mix-equal`, the class's bands at equal shares
#   SML (natural) `<class>@mix-natural`, at the shares the whole corpus holds
# Both mixes vote over as many positives as a pure arm at the same prevalence
# (`scale_bands.paired_mix`), so a row differs from another only in sizes.
#
# Columns are TEST sizes. S, M and L are the per-band cohorts, which are the
# same images under every arm, so a column is paired down its length. Test SML
# is not a fourth cohort: the bands share one threshold and one negative pool,
# so the miss rate and the AUROC over any mix of the three cohorts are that
# mix's weighted average of the three. The analysis weights them equally and
# at the natural shares.
#
# SigLIP binary only. Region Photo runs only when the owner asks for it
# (state-of-the-app skill, 2026-09-26); this script refuses it unless
# SVS_REGION=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/size-vs-size-4160}"
export CALIB_DATASETS=coco_better
if [[ "${SVS_REGION:-0}" == 1 ]]; then
  export CALIB_COCO_BETTER_EMBEDDERS="siglip,siglip+dinov3_patch"
  export CALIB_MEM="${CALIB_MEM:-80G}"
else
  export CALIB_COCO_BETTER_EMBEDDERS=siglip
  export CALIB_MEM="${CALIB_MEM:-6G}"
fi
export CALIB_PATCH_STYLES=max_patch
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${SVS_SEEDS:-20}"
export CALIB_MAX_STEPS="${SVS_MAX_STEPS:-150}"
export CALIB_CELL_ORDER=seed
export CALIB_TEST_BANDS=all
export CALIB_TEST_BAND_AUROC=1
export CALIB_TRAIN_MIXES=equal,natural
export CALIB_MIX_SHARES="$CALIB_EXP/mix_shares.json"
export CALIB_SAFE_THRESHOLDS=1
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_SKYLINE_ARMS=""
export CALIB_CPUS="${CALIB_CPUS:-1}"
export CALIB_TIME="${CALIB_TIME:-2:00:00}"
export CALIB_CONC="${CALIB_CONC:-200}"
export CALIB_JOB_NAME="${CALIB_JOB_NAME:-svs-4160}"
mkdir -p "$CALIB_EXP"

if [[ "${1:-}" == prepare ]]; then
  # The natural shares come from the corpus, not the cell pickle, whose bands
  # all hold the same number of positives by construction.
  ( source "$WT/gridenv.sh" && source "$WT/scripts/experiments/pile/pile_env.sh" &&
    cd "$WT/scripts/experiments/pile" &&
    python coco_better_export.py --mix-census --mix-census-json "$CALIB_MIX_SHARES" )
fi
case "${1:-}" in size|cells|redo)
  [[ -s "$CALIB_MIX_SHARES" ]] || { echo "no $CALIB_MIX_SHARES: run prepare first" >&2; exit 3; } ;;
esac

exec bash "$CALIB/launch_bands.sh" "$@"
