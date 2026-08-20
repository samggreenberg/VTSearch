#!/usr/bin/env bash
# link-demo-cache.sh — share downloaded demo datasets between VTSearch data dirs.
#
# On a multi-user server (or any machine with several VTSearch checkouts), the
# multi-GB demo-dataset sources only need to exist once. The demo downloaders
# skip a download whenever the dataset's extraction path already exists under
# the data dir (vtscore/datasets/downloader/*.py), so a communal cache
# directory whose entries mirror that extraction layout can be symlinked into
# each user's data dir and every linked demo is treated as already downloaded.
#
# Usage:
#   scripts/link-demo-cache.sh <cache-dir> <data-dir>            # link cached demos in
#   scripts/link-demo-cache.sh <cache-dir> <data-dir> --harvest  # first move demos this
#         data dir already downloaded INTO the cache, then link them back
#
# Rules:
#   * Only POPULATED cache entries are ever linked. Never hand-create an empty
#     dataset dir in the cache: the downloaders treat an existing extraction
#     path as "download complete", so an empty dir makes that demo silently
#     load zero items.
#   * A demo downloaded through an existing symlink writes into the cache
#     (intended). A demo whose dir didn't exist yet appears as a real dir in
#     the data dir instead; run --harvest periodically to donate those.
#   * Cache entries are made group-writable (best effort) so every user of a
#     shared cache can harvest into it. Run with a cooperative umask (e.g.
#     `umask 002`) on multi-user hosts.
set -euo pipefail

CACHE="${1:?usage: link-demo-cache.sh <cache-dir> <data-dir> [--harvest]}"
DATADIR="${2:?usage: link-demo-cache.sh <cache-dir> <data-dir> [--harvest]}"
MODE="${3:-}"
CACHE="$(cd "$CACHE" && pwd)"
DATADIR="$(cd "$DATADIR" && pwd)"

# Every demo extraction directory the downloaders create under DATA_DIR, per
# vtscore/datasets/downloader/*.py. Deliberately excludes the per-user dirs
# (embeddings, saved_datasets, models, detectors, staging, settings) and the
# derived images/ + video/ media dirs. Kept in sync with the source by
# tests_lib/downloads/test_demo_cache_script.py.
DEMO_DIRS=(
  aclImdb apollo11_audio bbc-fulltext birdvox_full_night caltech-101 caltech-256
  cifar-10-batches-py clotho dbpedia_csv enrico ESC-50-master EuroSAT_RGB
  food-101 gtzan hmdb51 nixon_tapes openlogo oxford_flowers places365
  reuters21578 rico_screen2words roxford5k rvl_cdip speech_commands_v2
  tut_sound_events_2017 UCF-101 UCF101_subset ucsf_documents UrbanSound8K
  vggface2 visual_genome
)

for name in "${DEMO_DIRS[@]}"; do
  src="$CACHE/$name"
  dst="$DATADIR/$name"

  if [[ "$MODE" == "--harvest" && -d "$dst" && ! -L "$dst" ]]; then
    if [[ -e "$src" ]]; then
      echo "SKIP harvest $name: already exists in cache (resolve by hand)"
    else
      echo "HARVEST $name -> cache"
      mv "$dst" "$src"
      chmod -R g+w "$src" 2>/dev/null || true
    fi
  fi

  # Link only populated cache entries (see the empty-dir rule above).
  if [[ -d "$src" ]] && [[ -n "$(ls -A "$src" 2>/dev/null)" ]]; then
    if [[ -L "$dst" ]]; then
      : # already linked, leave it
    elif [[ -e "$dst" ]]; then
      echo "SKIP link $name: real copy exists in data dir (remove it to use the cache)"
    else
      echo "LINK $name"
      ln -s "$src" "$dst"
    fi
  fi
done
echo "done."
