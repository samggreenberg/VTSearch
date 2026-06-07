# VTSBrowser QA follow-ups (2026-06-07 drive)

Findings from a full end-to-end QA drive of the browse flow (ESC-50 train →
UrbanSound8K Find → browse positives → region-select wrong items → Remove from
Good → back to Find) that were **not** fixed in the accompanying PR, with the
reasoning. The fixed findings (sidebar flex collapse, selection cleared on
Re-project, stale extent after a cull, autopilot boundary-phase copy,
dashboard Train/Find hint copy, mailto-subject typo) live in that PR.

## Deferred

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
