#!/usr/bin/env bash
# #5591 People cut: dump, sweep and score. Work dir is owner-only (FHIBE subject ids, celebrity photos).
#
#   bash run.sh dump     # FHIBE cells -> dumps/fhibe_{1024,640}.npz (srun)
#   bash run.sh demos    # vggface2_faces_s + celebrity_together_s through image2face -> dumps/ (srun)
#   bash run.sh sweep    # one sbatch job per dump x population x linkage -> sweep/<job>/arms.csv
#   bash run.sh post     # summarize -> sweep/all/, Celebrities in Context -> cic/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${PEOPLE_WORK:-/expscratch/$USER/fhibe/people-5591}"
DERIVED=/expscratch/$USER/fhibe/fhibe-full-resolution-674a7dcf-derived
LIVE_DATA=/expscratch/$USER/vtsearch-data
DEMO_CACHE=/exp/scale26/datasets/external/vtsearch-demos
PY="python $HERE/people_cut_5591.py"
ENV="source $REPO/scripts/grid/gridenv.sh >/dev/null 2>&1; cd $WORK"
mkdir -p "$WORK"/{dumps,sweep,logs}
chmod 700 "$WORK"

case "${1:-}" in
dump)
    srun --ntasks=1 --partition=cpu --mem=8G -c 2 --time=00:30:00 bash -lc "$ENV
      for s in 1024 640; do $PY dump-fhibe --cell $DERIVED/cells/fhibe_faces_ik_\${s}__face.pkl \
        --identities $DERIVED/runs/2026-10-09-k4/identities.txt --out $WORK/dumps/fhibe_\${s}.npz; done"
    ;;
demos)
    # A data dir of the study's own: the demos' photos and the face models linked in, nothing else, so the
    # image2face load (no cache) neither reads nor writes the live app's state.
    DD="$WORK/datadir"
    mkdir -p "$DD"
    ln -sfn "$LIVE_DATA/models" "$DD/models"
    ln -sfn "$DEMO_CACHE/vggface2" "$DD/vggface2"
    ln -sfn "$LIVE_DATA/celebrity_together" "$DD/celebrity_together"
    srun --ntasks=1 --partition=cpu --mem=24G -c 8 --time=02:00:00 bash -lc "$ENV; export VTSEARCH_DATA_DIR=$DD
      $PY dump-demo --demo vggface2_faces_s --out $WORK/dumps/vggface2_s.npz
      $PY dump-demo --demo celebrity_together_s --out $WORK/dumps/cic_s.npz"
    ;;
sweep)
    for s in 1024 640; do
        for pop in review all; do
            for lk in centroid average; do
                sbatch --parsable --job-name="people5591-$s-$pop-$lk" --partition=cpu --mem=24G -c 4 --time=04:00:00 \
                    --output="$WORK/logs/sweep-$s-$pop-$lk-%j.out" \
                    --wrap="$ENV; $PY sweep --dump $WORK/dumps/fhibe_$s.npz --population $pop --linkage $lk --out $WORK/sweep/$s-$pop-$lk"
            done
        done
    done
    if [ -f "$WORK/dumps/vggface2_s.npz" ]; then
        for lk in centroid average; do
            sbatch --parsable --job-name="people5591-vgg-$lk" --partition=cpu --mem=24G -c 4 --time=04:00:00 \
                --output="$WORK/logs/sweep-vgg-$lk-%j.out" \
                --wrap="$ENV; $PY sweep --dump $WORK/dumps/vggface2_s.npz --population all --linkage $lk --out $WORK/sweep/vgg-$lk"
        done
    fi
    ;;
post)
    srun --ntasks=1 --partition=cpu --mem=16G -c 4 --time=00:30:00 bash -lc "$ENV
      $PY summarize $(for f in "$WORK"/sweep/*/arms.csv; do [[ $f == */all/* ]] || printf -- '--arms %s ' "$f"; done) --out $WORK/sweep/all
      $PY cic --dump $WORK/dumps/cic_s.npz --out $WORK/cic --sheets --photo-root $LIVE_DATA/celebrity_together/images"
    ;;
*)
    sed -n 2,8p "$0"
    exit 2
    ;;
esac
