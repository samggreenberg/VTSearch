# VTSBrowser QA follow-ups (2026-06-07 drive)

Findings from a full end-to-end QA drive of the browse flow (ESC-50 train →
UrbanSound8K Find → browse positives → region-select wrong items → Remove from
Good → back to Find) that were **not** fixed in the accompanying PR, with the
reasoning. The fixed findings (sidebar flex collapse, selection cleared on
Re-project, stale extent after a cull, autopilot boundary-phase copy,
dashboard Train/Find hint copy, mailto-subject typo) live in that PR.

## Deferred

### Server wedges during startup re-init; accept loop starves (grid, 2026-06-07)

After a restart of `python app.py` on the grid (rack4n02, inside the SLURM
allocation), the server became unreachable — port 5000 LISTENing with the
accept backlog pegged at 129/128, connections timing out even from
`localhost` on the node itself — and stayed that way for 30+ minutes. A
py-spy dump showed the main thread parked in `select()` inside
`serve_forever`, while the two startup load tasks were stuck, not slow
(identical stacks across samples 5 s apart):

- `ds-load-…`: `build_diversity_tree_for_context` → sklearn kmeans →
  `threadpoolctl` → `posixpath.realpath` (frozen mid-`_joinrealpath`,
  smells like a hung NFS `lstat`)
- `det-load-…`: `train_from_labelset` → re-embedding label media → CLAP
  `_np_extract_fbank_features` → `power_to_db` (frozen at the same line
  across samples)

Total process CPU ~1.6 %, RSS 2.1 GB. The previous app instance likely
wedged the same way (it was restarted by hand right before this one).

**Why deferred:** root cause is not yet clear — frozen stacks at low CPU
point at the node's filesystem (NFS) rather than app logic, but the accept
starvation while two registry `load_task` threads hang is an app-side
fragility regardless. Worth: (a) a watchdog/timeout around the startup
registry load tasks, (b) confirming whether the dev server should accept
(and 503) while load tasks run, (c) checking grid NFS health when it
recurs. Evidence above is the full diagnostic so far.

### Browser tab crash during rapid autopilot voting (unreproduced)

Mid-drive, around the 16th vote of an autopilot labeling run with the audio
player churning between items, the tab died ("Session closed. Most likely the
page has been closed."). Backend state was intact — all 16 labels survived —
and a fresh tab resumed cleanly. Not reproduced since; no console output was
captured before the crash.

**Why deferred:** one occurrence, no repro, no stack. Suspect the audio
element churn (rapid create/play/destroy while voting quickly) but there is
nothing actionable yet.

**If it recurs:** drive the same flow with DevTools attached and
`chrome://crashes` enabled; watch the media element count and JS heap while
voting fast in audio mode.

## Skipped (deliberate)

### Top-left Back/Re-project overlay can sit over canvas bins

The `.browse-tools-left` cluster floats over the canvas, so a bin can land
under a button and a drag started there hits the button instead of the canvas.

**Why skipped:** this is the standard map-UI overlay pattern, used consistently
by this view (zoom/hex pills top-right, info chip bottom-left). The canvas is
pannable, so any bin under a control can be moved out from under it in one
drag. Reserving a non-canvas gutter for the controls would shrink the plot for
every session to remove an occasional one-drag annoyance — a bad trade. Revisit
only if real users report fighting the buttons.
