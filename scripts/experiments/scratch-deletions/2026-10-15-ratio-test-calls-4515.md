# 2026-10-15: 6.1 GB, the recorded ratio-test calls from #4515 (owner-approved)

- `/expscratch/sgreenberg/ratio-4515/rec-s/calls/` (8 `call_<k>.npz`): real 1,000-page
  `ratio_test_matches` calls (template, uint8 candidates, output pairs), recorded on 2026-10-15 to
  prove #4933's rewrite returns dev's pairs. #4933 is merged and #4515 is `solved`; only the
  bench scripts beside them read these files.
- **Rebuild:** `sbatch --export=ALL,W=<a worktree of dev> /expscratch/sgreenberg/ratio-4515/record.sbatch`
  re-records them from `profile_retrain.py`'s tier-s CPU loop in ~6 min. The rest of `ratio-4515/` is
  kept: the scripts, `calls.json` (per-call shapes and timings) and the `prof/` step logs.
