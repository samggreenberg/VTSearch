#!/usr/bin/env bash
# #3160, part 1: census every GPU node the picker can hand out.
#
#   bash launch_census.sh submit          # one job per node, pinned with --nodelist
#   bash launch_census.sh submit rack7n03 # just these nodes (a smoke test first)
#   bash launch_census.sh mechanism rack5n03 rack7n03   # deep probe, named nodes
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
  CTAG=""
  [[ -n "${VTS_CENSUS_TAG:-}" ]] && CTAG="--tag=${VTS_CENSUS_TAG}"
  CEXPORT=""
  [[ -n "${VTS_CENSUS_EXPORT:-}" ]] && CEXPORT="export ${VTS_CENSUS_EXPORT} && "
  for type in $TYPES; do
    while read -r node state; do
      if [[ -n "$ONLY" ]] && [[ " $ONLY " != *" $node "* ]]; then continue; fi
      case "$state" in
        *drain*|*down*|*maint*|*resv*) echo "skip $node ($state)"; continue ;;
      esac
      out="$CENSUS/$node${VTS_CENSUS_TAG:-}"
      if [[ -f "$out/device.json" ]] && [[ -z "${VTS_FORCE:-}" ]]; then
        echo "have $node already (VTS_FORCE=1 to redo)"; continue
      fi
      jid=$(sbatch --parsable \
        --job-name="census-$node${VTS_CENSUS_TAG:-}" \
        --partition=gpu --nodelist="$node" --gres="gpu:$type:1" \
        --cpus-per-task=4 --mem=24G --time=1:00:00 \
        --output="$LOGS/census-$node-%j.out" \
        --wrap "bash -lc '$ENVSET && ${CEXPORT}cd $HERE && python probe_device.py --out $CENSUS --images $IMAGES $CTAG'")
      if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "FAILED to submit $node (empty job id) -- NOT LAUNCHED" >&2; exit 1
      fi
      echo "submitted $node ($type) -> job $jid"
      n=$((n + 1))
    done < <(nodes_of_type "$type")
  done
  echo "$n node jobs submitted"
  ;;

mechanism)
  # The deep probe (probe_mechanism.py) answers *why*, so it only needs the two
  # nodes that disagree plus one from a third device -- not the pool.
  MECH="$STUDY/mechanism"
  mkdir -p "$MECH"
  # VTS_MECH_PIXELS points at another node's pixels.npy so this run also does the
  # forward on an *identical* input tensor; VTS_MECH_DEP chains a second round
  # behind the round that produces it, so the pair runs unattended.
  # VTS_MECH_TAG names the run when a node is probed more than once;
  # VTS_MECH_EXPORT injects env (e.g. ATEN_CPU_CAPABILITY=avx2) into the job.
  TAG=""
  [[ -n "${VTS_MECH_TAG:-}" ]] && TAG="--tag=${VTS_MECH_TAG}"
  EXPORT=""
  [[ -n "${VTS_MECH_EXPORT:-}" ]] && EXPORT="export ${VTS_MECH_EXPORT} && "
  EMB=""
  [[ -n "${VTS_MECH_EMBEDDER:-}" ]] && EMB="--embedder=${VTS_MECH_EMBEDDER}"
  PIX=""
  [[ -n "${VTS_MECH_PIXELS:-}" ]] && PIX="--pixels=${VTS_MECH_PIXELS}"
  DEP=()
  [[ -n "${VTS_MECH_DEP:-}" ]] && DEP=(--dependency="${VTS_MECH_DEP}")
  for node in "${@:2}"; do
    type=$(sinfo -h -N -n "$node" -o "%G" | head -1 | sed -E 's/.*gpu:([A-Za-z0-9_.-]+):.*/\1/')
    if [[ -z "$type" ]]; then echo "cannot resolve a GPU type for $node" >&2; exit 1; fi
    jid=$(sbatch --parsable \
      --job-name="mech-$node" \
      --partition=gpu --nodelist="$node" --gres="gpu:$type:1" \
      --cpus-per-task=4 --mem=24G --time=1:00:00 \
      "${DEP[@]}" \
      --output="$LOGS/mech-$node-%j.out" \
      --wrap "bash -lc '$ENVSET && ${EXPORT}cd $HERE && python probe_mechanism.py --out $MECH $PIX $TAG $EMB'")
    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
      echo "FAILED to submit $node (empty job id) -- NOT LAUNCHED" >&2; exit 1
    fi
    echo "submitted mechanism $node${VTS_MECH_TAG:-} ($type) -> job $jid"
  done
  ;;

status)
  squeue -u "$USER" -o "%.10i %.20j %.9T %.11M %R" | grep -E 'census|mech|JOBID' || echo "(no census jobs queued)"
  echo "nodes done: $(ls "$CENSUS"/*/device.json 2>/dev/null | wc -l)"
  ;;

analyze)
  source "$WT/gridenv.sh"
  cd "$HERE" && python analyze_census.py --census "$CENSUS" "${@:2}"
  ;;

*)
  echo "usage: $0 {submit [nodes...]|mechanism <nodes...>|status|analyze}" >&2; exit 1 ;;
esac
