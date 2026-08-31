#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --job-name=acq2877-throttle
# When the cheap half drains, give its memory quota to the expensive one.
#
# The two halves are throttled so they finish together at the memory split they
# START with (bin 7x4x3G = 84G, reg 7x9x12G = 756G).  But `bin` finishes hours
# first, and its 84G then sits idle while `reg` -- the critical path -- runs at a
# throttle sized around a quota it no longer has to share.
#
# `ArrayTaskThrottle` is the right lever: `MinMemoryNode` cannot retarget tasks
# the scheduler has already dispatched, so raising the throttle frees capacity as
# tasks finish rather than fighting them.
#
# GRID-side on purpose.  A local watcher doing this gets culled -- one did, ten
# minutes in -- and the whole point of the bump is that it happens unattended,
# hours from now, whether or not anything is still connected.
set -uo pipefail

BIN_NAMES=acq2877-bin-prod,acq2877-bin-acq_m1,acq2877-bin-acq_m2,acq2877-bin-acq_m3,acq2877-bin-acq_m4,acq2877-bin-acq_p2,acq2877-bin-rank_pin
REG_IDS="590289 590307 590318 590334 590347 590359 590379"
NEW_THROTTLE="${NEW_THROTTLE:-11}"

echo "waiting for the bin half to drain ($(date))"
while [ "$(squeue -u "$USER" -h -n "$BIN_NAMES" -o %i | wc -l)" -ne 0 ]; do
  sleep 120
done
echo "bin half drained at $(date)"

# Re-check the reg half is still alive before touching it: if it finished first
# (a top-up, a resubmission, a cancelled run) there is nothing to bump and
# `scontrol` would report an error that reads like a failure.
alive=0
for j in $REG_IDS; do
  squeue -h -j "$j" -o %i >/dev/null 2>&1 && [ -n "$(squeue -h -j "$j" -o %i 2>/dev/null)" ] && alive=$((alive + 1))
done
if [ "$alive" -eq 0 ]; then
  echo "no reg arrays left in the queue; nothing to bump"
  exit 0
fi

# 7 arms x 11 x 12G = 924G of the 1074G quota.  Sized against the quota, not
# against what looks generous: memory is what binds here, and it binds against
# my OWN jobs -- an array claiming the whole allowance parks everything I submit
# afterwards behind it, which is indistinguishable from a busy cluster.
for j in $REG_IDS; do
  scontrol update "JobId=$j" "ArrayTaskThrottle=$NEW_THROTTLE" \
    && echo "  $j -> %$NEW_THROTTLE" \
    || echo "  $j: throttle update refused (already finished?)"
done

echo "done at $(date)"
squeue -u "$USER" -h -O "Name:26,State:12" | sort | uniq -c
