#!/usr/bin/env bash
# Threshold-stability study (#2790) full pipeline on the HLTCOE Grid.
#
# CPU-only: reuses the #2790 SigLIP 2 / whole embedding cache (point THRSTAB_CACHE_DIR
# at it), so no prepare/GPU stage is needed — the sweep reads vectors from the cache
# and trains only tiny MLPs. If the cache is incomplete, pre-populate it with one
# GPU sweep run first (see scripts/sod/README.md), then run this.
#
# Usage:
#   export VTS_REPO=/exp/$USER/projects/vts-evalfw
#   export THRSTAB_CACHE_DIR=/exp/$USER/threshold-stability/cache   # the reused #2790 cache
#   bash launch_all.sh
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-evalfw}"
exec bash "$WT/scripts/experiments/threshold_stability/launch_cells.sh"
