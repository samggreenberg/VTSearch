#!/usr/bin/env bash
# #3160, part 1: census every GPU node the picker can hand out.
#
#   bash launch_census.sh submit          # one job per node, pinned with --nodelist
#   bash launch_census.sh submit rack7n03 # just these nodes (a smoke test first)
#   bash launch_census.sh status
#   bash launch_census.sh analyze
#
# #3143 tested two V100 nodes and found they disagree.  The question that
# actually governs a pile rebuild is the one it could not answer: how many
# distinct *devices* hide behind the three type labels `pick_gpu.py` chooses
# between, and how many of them are outliers?  That is a probability -- the
# chance a rebuild lands somewhere that does not reproduce -- and it needs the
# whole pool, not a pair.
#
# h100/h200 nodes are excluded on purpose: the `4gpu_tier` QOS caps them at 0,
# so a job asking for one pends forever (GRID-PLAYBOOK section 2).  They are not
# in `pick_gpu.py`'s candidate list either, so they cannot receive a pile build.
set -euo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
STUDY="${VTS_NODE_STUDY:-/expscratch/$USER/gpu-node-3160}"
CENSUS="$STUDY/census"
LOGS="$STUDY/logs"
PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
IMAGES="${VTS_CENSUS_IMAGES:-256}"
TYPES="${VTS_CENSUS_TYPES:-v100 l40s a100}"

mkdir -p "$CENSUS" "$LOGS"

# Weights come from the shared pile, and VTSEARCH_DATA_DIR is pointed at a
# scratch dir of our own: the probe writes nothing but its own vectors, and must
# not be able to touch the pile even by accident.
ENVSET="source $WT/gridenv.sh"
ENVSET="$ENVSET && export VTS_REPO=$WT HF_HOME=$PILE/models VTSEARCH_MODELS_DIR=$PILE/models"
ENVSET="$ENVSET && export VTSEARCH_DATA_DIR=$STUDY/probe-datadir"
ENVSET="$ENVSET && export VTSEARCH_TORCH_THREADS=4 OMP_NUM_THREADS=4"

nodes_of_type() {
  sinfo -h -N -p gpu -o "%N %t %G" | sort -u | awk -v t="gpu:$1:" '$3 ~ t {print $1, $2}'
}

case "${1:-status}" in
submit)
  n=0
  ONLY="${*:2}"
  for type in $TYPES; do
    while read -r node state; do
      if [[ -n "$ONLY" ]] && [[ " $ONLY " != *" $node "* ]]; then continue; fi
      case "$state" in
        *drain*|*down*|*maint*|*resv*) echo "skip $node ($state)"; continue ;;
      esac
      out="$CENSUS/$node"
      if [[ -f "$out/device.json" ]] && [[ -z "${VTS_FORCE:-}" ]]; then
        echo "have $node already (VTS_FORCE=1 to redo)"; continue
      fi
      jid=$(sbatch --parsable \
        --job-name="census-$node" \
        --partition=gpu --nodelist="$node" --gres="gpu:$type:1" \
        --cpus-per-task=4 --mem=24G --time=1:00:00 \
        --output="$LOGS/census-$node-%j.out" \
        --wrap "bash -lc '$ENVSET && cd $HERE && python probe_device.py --out $CENSUS --images $IMAGES'")
      if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "FAILED to submit $node (empty job id) -- NOT LAUNCHED" >&2; exit 1
      fi
      echo "submitted $node ($type) -> job $jid"
      n=$((n + 1))
    done < <(nodes_of_type "$type")
  done
  echo "$n node jobs submitted"
  ;;

status)
  squeue -u "$USER" -o "%.10i %.20j %.9T %.11M %R" | grep -E 'census|JOBID' || echo "(no census jobs queued)"
  echo "nodes done: $(ls "$CENSUS"/*/device.json 2>/dev/null | wc -l)"
  ;;

analyze)
  source "$WT/gridenv.sh"
  cd "$HERE" && python analyze_census.py --census "$CENSUS" "${@:2}"
  ;;

*)
  echo "usage: $0 {submit|status|analyze}" >&2; exit 1 ;;
esac
