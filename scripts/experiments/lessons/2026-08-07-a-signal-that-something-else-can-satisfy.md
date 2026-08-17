# 2026-08-07 — a signal that something else can satisfy (#2897)

**Cost:** ~40 min of run time, and one wrong "it's finished" told to the owner.

**What broke.** Three times in one study I waited on a signal that something
*other than the thing I was waiting for* could satisfy.

1. **An empty job id read as a launch.** `launch_all.sh`'s prepare stage
   hardcoded `--partition=gpu` and passed `--gres="$GRES"` through literally.
   Every recent study sets `CALIB_GRES=none` for CPU-only cells — and
   `--gres=none` is not ignored here: the submit filter rewrites it into a
   `--gpus-per-task` form and rejects the job for having no `-n`. `--parsable`
   returns an **empty string** on refusal, so `--dependency=afterok:` got a blank
   id and the dependent launcher failed too, burying the real error. **Both live
   A/B arms silently failed to launch** into a log nobody was reading. Latent for
   every study using the full chain with `CALIB_GRES=none`; the screen escaped
   only because its prepare had been run by hand.
2. **A stale file read as completion.** Completion was armed as "does
   `STATUS.md` exist" — but an earlier stage had already written one. The watcher
   fired 18 seconds later and I reported the study complete about an hour early,
   with 126 jobs still queued.
3. **A degenerate arm read as a broken one.** The A/B builds
   `CALIB_FOLD_COUNTS="2,$K"`, which at K=2 collapses to a single value, so the
   *control* arm's analyze died with `KeyError: 'voting'` — a traceback where the
   honest answer was "nothing to contrast".

**Now prevented (code):**

1. `launch_all.sh` drops `--gres` when it is `none`/empty (the `GRES_ARG` shape
   `launch_cells.sh` already used) and honours `CALIB_PREP_PARTITION` /
   `CALIB_PARTITION`; `require_jobid` after **every** `sbatch` aborts on a
   non-numeric id, naming the stage.
2. `analyze_folds_2897.py` returns the empty shape for a baseline-only arm.

**Still advice — the shape of the mistake, which no single check covers.**
Prefer a signal that *cannot* pre-exist: a terminal job state, or a marker
written only by the stage you are waiting for. Before arming any wait, ask what
else could satisfy it. And note the launchers already printed "confirm each arm
came back with a numeric job id": that instruction was correct and was ignored,
because the thing reading the output was a batch job. **A check whose only
enforcement is a human reading a log does not survive being chained** — if a
stage can be chained, its checks have to be code.

*(This study also re-hit the #2877 premise trap above: `visual_genome_m ×
siglip` carries no `patch_grid`, so its `region_voting=True` is a silent no-op.
The one-line assertion recommended there is what caught it.)*
