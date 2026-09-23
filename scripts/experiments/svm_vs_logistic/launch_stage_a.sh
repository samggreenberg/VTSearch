#!/usr/bin/env bash
# #3197 Stage A: the heads on fixed vote sets.  One array task per
# (dataset, embedder, category), categories taken from Stage B's prepare so the
# two stages measure the same cells.
#
#   bash launch_stage_a.sh controlled   # every controlled condition (no Stage B needed)
#   bash launch_stage_a.sh replay       # Stage B's own vote sets (after harvest_picks.py)
#
# `coco_val x siglip@raw` rides along in `controlled`: the pile's un-normalised
# copy of that cell (norms 12-19, see make_datadir.py), as a natural
# input-SCALE arm for the regularisation question.
set -euo pipefail
MODE="${1:?controlled|replay}"
WT="${VTS_REPO:-/expscratch/$USER/worktrees/vts-svmlog-3197}"
HERE="$WT/scripts/experiments/svm_vs_logistic"
BASE="${SVMLOG_BASE:-/expscratch/$USER/svmlog-3197}"
PREP="$BASE/stageB/prepare/results/prepare_info.json"
OUT="$BASE/stageA/$MODE"
mkdir -p "$OUT/logs"
TASKS="$OUT/tasks.tsv"

python3 - "$PREP" "$MODE" > "$TASKS" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
mode = sys.argv[2]
for ds, embs in sorted(info["datasets"].items()):
    for emb, e in sorted(embs.items()):
        for cat in e["selected_categories"]:
            print(f"{ds}\t{emb}\t{cat}")
            if mode == "controlled" and ds == "coco_val" and emb == "siglip":
                print(f"{ds}\tsiglip@raw\t{cat}")
PY
N=$(wc -l < "$TASKS")
EXTRA=""
[[ "$MODE" == "replay" ]] && EXTRA="--replay-root $BASE/stageB --replay-only"
echo "$N tasks -> $OUT"
J=$(sbatch --parsable --job-name="svmlog-A-$MODE" --array=0-$((N-1))%60 --partition=cpu \
  --cpus-per-task=1 --mem=4G --time=2:00:00 --output="$OUT/logs/%A_%a.out" \
  --export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1 \
  --wrap="source $WT/gridenv.sh >/dev/null 2>&1 && IFS=\$'\t' read -r DS EMB CAT < <(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $TASKS) && python $HERE/stage_a.py --dataset \"\$DS\" --embedder \"\$EMB\" --category \"\$CAT\" --out $OUT $EXTRA")
[[ "$J" =~ ^[0-9]+$ ]] || { echo "sbatch refused: $J" >&2; exit 1; }
echo "array: $J"
