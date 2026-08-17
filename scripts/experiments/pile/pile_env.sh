#!/usr/bin/env bash
# Source this to point a study at the shared pre-embedded pile.
#
#   source scripts/experiments/pile/pile_env.sh
#
# After this, load_demo_dataset / the calibration harness read the per-pair
# pickles in place instead of re-embedding, and model weights resolve to the
# pile's models dir rather than being re-downloaded per study.
#
# HF_HOME is pinned here on purpose: the grid shell points it at /exp, where a
# single model download fills the 50G quota.

VTS_PILE="${VTS_PILE:-/expscratch/${USER}/vts-cache}"
export VTS_PILE
export VTSEARCH_DATA_DIR="$VTS_PILE/datadir"
export VTSEARCH_MODELS_DIR="$VTS_PILE/models"
export HF_HOME="$VTS_PILE/models"

if [[ ! -d "$VTSEARCH_DATA_DIR/embeddings" ]]; then
  echo "warning: no pile at $VTS_PILE (build it with build_pile.py)" >&2
fi
