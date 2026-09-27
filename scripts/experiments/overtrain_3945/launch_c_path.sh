#!/usr/bin/env bash
# #3945 Stage R: the SVM head's C path along replayed Autopilot sessions.
# One array task per (dataset, embedder, category); categories from #3197's
# prepare, so the replay reads exactly the sessions that study ran.
#
#   bash launch_c_path.sh <stageB root> <arm> <out dir>
#   bash launch_c_path.sh /expscratch/$USER/svmlog-3197/stageB svm /expscratch/$USER/overtrain-3945/R150
#   bash launch_c_path.sh /expscratch/$USER/overtrain-3945/stageD svm /expscratch/$USER/overtrain-3945/R400
#
# The <stageB root>/<arm>/picks_all.csv.gz must exist (svm_vs_logistic/harvest_picks.py).
set -euo pipefail
ROOT="${1:?stageB root}"
ARM="${2:?arm}"
OUT="${3:?out dir}"
WT="${VTS_REPO:-/expscratch/$USER/worktrees/vts-overtrain-3945}"
HERE="$WT/scripts/experiments/overtrain_3945"
PREP="${PREP_INFO:-/expscratch/$USER/svmlog-3197/stageB/prepare/results/prepare_info.json}"
[[ -f "$ROOT/$ARM/picks_all.csv.gz" ]] || { echo "no $ROOT/$ARM/picks_all.csv.gz (run harvest_picks.py)" >&2; exit 1; }
[[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ]] || { echo "commit first: $WT is dirty" >&2; exit 1; }
mkdir -p "$OUT/logs"
TASKS="$OUT/tasks.tsv"
python3 - "$PREP" > "$TASKS" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
for ds, embs in sorted(info["datasets"].items()):
    for emb, e in sorted(embs.items()):
        for cat in e["selected_categories"]:
            print(f"{ds}\t{emb}\t{cat}")
PY
N=$(wc -l < "$TASKS")
git -C "$WT" rev-parse HEAD > "$OUT/commit.txt"
echo "$N tasks -> $OUT (commit $(cat "$OUT/commit.txt"))"
J=$(sbatch --parsable --job-name="ovt-R-$ARM" --array=0-$((N-1))%60 --partition=cpu \
  --cpus-per-task=1 --mem="${R_MEM:-6G}" --time="${R_TIME:-3:00:00}" --output="$OUT/logs/%A_%a.out" \
  --export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1 \
  --wrap="source $WT/gridenv.sh >/dev/null 2>&1 && IFS=\$'\t' read -r DS EMB CAT < <(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $TASKS) && python $HERE/c_path.py --dataset \"\$DS\" --embedder \"\$EMB\" --category \"\$CAT\" --replay-root $ROOT --arm $ARM --out $OUT")
[[ "$J" =~ ^[0-9]+$ ]] || { echo "sbatch refused: $J" >&2; exit 1; }
echo "array: $J"
