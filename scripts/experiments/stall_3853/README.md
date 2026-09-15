# Stall reproduction kit for #3853

Issue #3853: a rare 5-20 s stall after a vote in which the centre panel goes
black and every in-flight request finishes at the same instant. The fifth
comment on the issue captured one (a 4.9 s vote POST with `/api/votes`,
`/api/labeling-status` and the lock-free `/api/jobs/active` all frozen with
it) and narrowed it to *something holding the whole worker*, but a per-request
timer cannot say what. The app now carries the instruments that can
(`vtscore/concurrency/stalls.py`, see `docs/DEPLOYMENT.md` → "Diagnosing a
stall"); this directory holds what drives and reads them.

| File | Job |
|---|---|
| `serve_synthetic.py` | Run the real app over an in-memory synthetic patch dataset (N images, `dinov3_patch` CLS + `14x14x768` grids), for reproducing offline. Patch grids are never pickled, so a VG slate cannot be loaded without the DINOv3 weights and a GPU; this builds the post-load shape directly. |
| `drive_labeling.py` | Replay the SPA's per-vote request chain against any running VTSearch (a tunnel to the GRID included), timing each request client-side and reporting keypress→panel, sort wait, per-endpoint percentiles, and head-of-line clusters. Stdlib only. |
| `analyze_app_log.py` | Read the app's JSON log and print the ±15 s window around every `stall:` line, with the `faulthandler` thread dump that fired during it. |

## What the instruments say

On a stall the log carries, at the default WARNING level:

- `stall: heartbeat late by Nms; process cpu X of Y wall (r); top threads: …;
  majflt +n; gc gen2 pauses +n; rss …; cgroup mem …; thread dump armed …` —
  from the watchdog. `r ≈ 1` with one thread on top is a GIL hold; the
  thread dump written just before this line names its frame. `r ≈ 0` and no
  thread on top means the process was not scheduled (memory pressure,
  swapping, a paged-out cgroup): read `majflt` and the cgroup limit hits.
- `slow request: …` for each frozen request (`vtsearch/hooks.py`).
- `slow phase: learned_sort …`, `slow phase: train_and_score …`,
  `slow phase: label_sync …`, `slow phase: labeling_status_advance …`,
  `slow phase: rehydrate …` — which part of the suspect paths grew.
- `lock wait: <lock> waited Nms` — a vote, the sort thread or the status
  poll queued behind another holder of `_state_lock`, `_progress_lock` or
  `label_sync_write_lock`.
- `gc pause: generation 2 took Nms`.

At `VTSEARCH_LOG_LEVEL=INFO` two more: `rehydrating votes … : <reason>` /
`rehydrate for detector …: already fresh | restored N labels` and
`progress cache truncated at step i of n … k steps to replay`.

## On the GRID (the careful experiment)

1. Pull `dev` and start the app with the launcher as usual. It now sets
   `VTSEARCH_LOG_FILE` to `data/logs/app-<node>-<timestamp>.log`, which is
   also where the thread dump lands. Add `VTSEARCH_LOG_LEVEL=INFO` to get
   the rehydrate and truncation lines; optionally `VTSEARCH_SLOW_REQUEST_MS=400
   VTSEARCH_SLOW_PHASE_MS=300` to lower the bars.
2. Label a slate as normal. Note the wall-clock time of any felt stall.
3. `python scripts/experiments/stall_3853/analyze_app_log.py data/logs/app-*.log`.
   Every stall the watchdog saw is printed with what surrounded it; if the
   reviewer felt a pause the watchdog did not report, the stall was not in
   the server process (browser or network) — the earlier client-side probe
   in the issue covers that side.
4. Record RSS and the cgroup memory counters over the session too (the
   watchdog prints them on a stall; `cat /sys/fs/cgroup/memory.current`
   from the job shell gives the baseline). The datasets loaded that day
   each hold a `7.7k × 197 × 768` region matrix, and a process near its
   48 GB allocation stalls in exactly the observed shape.

Optionally, drive a synthetic session through the tunnel instead of hand
labeling, to get a stall on demand: `drive_labeling.py --base-url
http://localhost:<port> --dataset-id auto --detector-id <id> --votes 400
--region --flip-every 8`. Flips (relabeling an earlier item) are the client
action that truncates the labeling-status cache and makes its worker replay
the history under `_progress_lock`.

**Match the shape the reviewer labels.** The VG slates are ~500-image
`server_folder` imports embedded with SigLIP and labeled through a plain
*semantic* text-query detector — no patch grids, no region boxes. To
reproduce that against a fresh instance, import the slate folder over the
API and let the driver register the detector the way the SPA does:
`drive_labeling.py --create-detector "knife (repro)" --embedder-type semantic
--text-query knife --votes 600 --cadence 1.0`. `--region` and the default
`patch_semantic` type exercise a different code path (the region-vote
retrain), which is what the offline run below measures.

**Read the app log even when there is no `stall:` line.** The watchdog only
fires when the *interpreter* cannot run. A vote held by its own `fsync` to a
slow filesystem, or a request holding `_state_lock` across a `stat()` of the
detector file, never trips it; those show up only as `slow phase:
label_sync … write=NNNms`, `lock wait: _state_lock/…` and the requests
queued behind them, which `analyze_app_log.py` now lists after the stall
windows. The 2026-09-15 run on the GRID (issue thread) found exactly that
shape in the reviewer's own trace: every slow vote but one was slow *alone*,
and the one that froze everything did so through the lock every request
takes in `get_context()`.

## Offline (this kit's own reproduction)

```bash
S=/tmp/stall-3853
VTSEARCH_DATA_DIR=$S/data VTSEARCH_LOG_FILE=$S/data/app.log VTSEARCH_LOG_LEVEL=INFO \
VTSEARCH_SLOW_REQUEST_MS=400 VTSEARCH_SLOW_PHASE_MS=300 VTSEARCH_DEVICE=cpu HF_HUB_OFFLINE=1 \
python scripts/experiments/stall_3853/serve_synthetic.py --medias 2000 --port 5077 &
python scripts/experiments/stall_3853/drive_labeling.py --base-url http://127.0.0.1:5077 \
    --create-detector knife --votes 400 --region --out $S/reqlog.jsonl
python scripts/experiments/stall_3853/analyze_app_log.py $S/data/app.log
```

What the offline run measured is recorded in the issue thread, not here:
the numbers depend on the box.
