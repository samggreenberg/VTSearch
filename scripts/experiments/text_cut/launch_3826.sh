#!/usr/bin/env bash
# #3826 - what should draw the line on a typed-query (text) sort.
#
#   bash launch_3826.sh capture    # 16 (dataset, text embedder) pairs -> corpus/*.npz, scores + labels
#   bash launch_3826.sh gate       # every rule x every perturbation on every sort -> analysis/*.csv
#   bash launch_3826.sh analyse    # tables + figures from the gate's CSVs
#   bash launch_3826.sh status
#
# Set DEP=afterok:<id> to chain a stage behind a running one GRID-side.
#
# Four environments on purpose, so the answer is not about one retired dataset:
#   caltech101_m     838 medias, single-label, 25 categories (one object per image)
#   coco_val         4,952 medias, multi-label, 80 categories, prevalence 0.2%-54%
#   visual_genome_m  4,193 medias, multi-label, 100 categories incl. stuff (sky, wall)
#   vg_scale         ~13-16k medias per cell, 75 size-banded cells at 100 positives
# x four text towers (siglip, siglip2_l, clip, clip_l).  Read-only on the pile.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/textcut-3826}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
CORPUS="${CORPUS:-$CALIB_EXP/corpus}"
ANALYSIS="${ANALYSIS:-$CALIB_EXP/analysis}"
LOGS="$CALIB_EXP/logs"
OLD_CORPUS="${OLD_CORPUS:-/expscratch/$USER/gmm-3585/corpus}"
mkdir -p "$CORPUS" "$ANALYSIS" "$LOGS" "$CALIB_RESULTS"

# The rules differ in how much numpy they do; pin BLAS so a timing is the rule's.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PAIRS=(
  caltech101_m:siglip caltech101_m:siglip2_l caltech101_m:clip caltech101_m:clip_l
  coco_val:siglip coco_val:siglip2_l coco_val:clip coco_val:clip_l
  visual_genome_m:siglip visual_genome_m:siglip2_l visual_genome_m:clip visual_genome_m:clip_l
  vg_scale:siglip vg_scale:siglip2_l vg_scale:clip vg_scale:clip_l
)
ENVX="source $WT/gridenv.sh && source $WT/scripts/experiments/pile/pile_env.sh && export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
DEPARG=()
[[ -n "${DEP:-}" ]] && DEPARG=(--dependency="$DEP")

case "${1:-}" in
capture)
  last=$((${#PAIRS[@]} - 1))
  sbatch --parsable "${DEPARG[@]}" --job-name=tc3826-cap --partition=cpu --mem=24G --cpus-per-task=2 \
    --time=02:00:00 --array=0-"$last" --output="$LOGS/cap-%A_%a.out" \
    --wrap="$ENVX && P=(${PAIRS[*]}) && pr=\${P[\$SLURM_ARRAY_TASK_ID]} && ds=\${pr%%:*} && emb=\${pr##*:} && cd $HERE && python capture_text_sorts_3826.py --dataset \$ds --embedder \$emb --out $CORPUS/\${ds}__\${emb}.npz"
  ;;
gate)
  # One task per corpus file.  The old #3585 captures ride along as a
  # stability-only frame: they carry no labels, and they are what reproduces
  # the issue's own numbers before anything new is claimed.
  files=("$CORPUS"/*.npz "$OLD_CORPUS"/sorts_vgscale.npz "$OLD_CORPUS"/sorts_overview.npz)
  last=$((${#files[@]} - 1))
  printf '%s\n' "${files[@]}" > "$ANALYSIS/gate_inputs.txt"
  sbatch --parsable "${DEPARG[@]}" --job-name=tc3826-gate --partition=cpu --mem=8G --cpus-per-task=1 \
    --time=08:00:00 --array=0-"$last" --output="$LOGS/gate-%A_%a.out" \
    --wrap="$ENVX && cd $HERE && f=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $ANALYSIS/gate_inputs.txt) && python gate_3826.py --capture \$f --out $ANALYSIS/parts"
  ;;
gatechunk)
  # The vg_scale x clip / clip_l captures run ~300 s a sort (every EM on a flat
  # ridge, x20 resamples, x8 multistart starts) and do not fit 8 h as one task.
  # `bash launch_3826.sh gatechunk vg_scale__clip 6` splits one capture six ways.
  cap="${2:?capture stem}"; k="${3:-6}"
  sbatch --parsable "${DEPARG[@]}" --job-name=tc3826-gate --partition=cpu --mem=8G --cpus-per-task=1 \
    --time=08:00:00 --array=0-$((k - 1)) --output="$LOGS/gatechunk-$cap-%A_%a.out" \
    --wrap="$ENVX && cd $HERE && python gate_3826.py --capture $CORPUS/$cap.npz --out $ANALYSIS/parts --chunk \$SLURM_ARRAY_TASK_ID/$k"
  ;;
analyse)
  sbatch --parsable "${DEPARG[@]}" --job-name=tc3826-an --partition=cpu --mem=16G --cpus-per-task=2 \
    --time=01:00:00 --output="$LOGS/analyse-%j.out" \
    --wrap="$ENVX && cd $HERE && python analyze_3826.py --parts $ANALYSIS/parts --out $ANALYSIS --corpus $CORPUS && python figures_3826.py --parts $ANALYSIS/parts --corpus $CORPUS --out $WT/docs/experiments/2026-09-22-text-cut-3826/figures && mkdir -p $WT/docs/experiments/2026-09-22-text-cut-3826/tables && cp $ANALYSIS/tables/*.csv $ANALYSIS/summary.json $WT/docs/experiments/2026-09-22-text-cut-3826/tables/"
  ;;
status)
  echo "corpus:  $(ls -1 "$CORPUS"/*.npz 2>/dev/null | wc -l) / ${#PAIRS[@]} captures"
  echo "parts:   $(ls -1 "$ANALYSIS"/parts/*_cuts.csv 2>/dev/null | wc -l) gated"
  squeue -u "$USER" -n tc3826-cap,tc3826-gate,tc3826-an
  ;;
*)
  echo "usage: $0 {capture|gate|analyse|status}" >&2
  exit 2
  ;;
esac
