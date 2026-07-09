# Screenshot reshoot queue

A running list of doc screenshots that are **known stale** — the GUI they
capture has changed, but the PNGs under `docs/user/assets/` haven't been
re-rendered yet.

**Why this file exists.** Screenshots are captured by a Playwright harness
(`scripts/screenshots/refresh.sh`) that needs a real browser. The standard
cloud container has **no chromium** (see `CLAUDE.md` → "No Chrome/Chromium
available"), so a session that changes the GUI usually *can't* reshoot the
affected shots itself. Instead of letting that drift go silently unrecorded,
the changing session adds the affected shot id(s) here. A later
browser-capable session drains the queue.

See `docs/plans/user-docs-screenshots.md` for the full screenshot system
(manifest, harness, determinism knobs, embedding convention).

## How to use it

**When you change the GUI and can't reshoot (no browser):** for every shot
whose framed UI your change alters, add a row to the table below. The shot
`id` must match an entry in `docs/user/screenshots.manifest.ts` — the wiring
check (`scripts/screenshots/wiring-check.py`, gated in `run-tests.sh`)
enforces this, so a typo'd or renamed id fails the suite.

**When you have a browser and want to drain the queue:**

1. Run `scripts/screenshots/refresh.sh` to re-render the stale shots (or all).
2. Review `git diff --stat docs/user/assets/` like any diff.
3. Commit the regenerated PNGs and **delete the drained rows** from the table
   below (leave the queue empty, not stale).

An empty table means "no known-stale shots" — the desired resting state.

## Queue

| Shot id | Embedded in | Why it's stale | Flagged |
|---------|-------------|----------------|---------|
| `importer-picker` | `docs/user/USER_GUIDE.md#loading-a-dataset` | The shot's caption calls out "per-row readiness badges" (`.badge-ready`/`.badge-embedding`); the 2026-07-09 UI style review Phase 1 fix swapped their unreadable white text for a new `--badge-text-dark` token, changing the badge's rendered text color in both themes. | 2026-07-09 |
| `dashboard-loaded` | USER_GUIDE.md#what-vtsearch-does | §2 contrast fix: dark-theme table row dividers (--border-subtle) now perceptible; dim sub-text (--text-dim) lightened | 2026-07-09 |
| `dashboard-manage` | USER_GUIDE.md#dashboard--managing-datasets-and-detectors | §2 contrast fix: dark-theme table row dividers (--border-subtle) now perceptible; dim sub-text (--text-dim) lightened | 2026-07-09 |
| `three-panel` | USER_GUIDE.md#the-three-panel-layout | (1) Panels now snap tight to their grid columns on load (not just on divider drag), so the left/right panel widths — and the centre boundary between them — shift from the old restored-width layout. (2) The collapsed "Metadata" toggle moved from a full-width divider row above the Good/Bad buttons to a docked strip below them (center-panel tray now sits under the voting overlay). | 2026-07-09 |
| `results-grid` | USER_GUIDE.md#view-options | Left panel snaps tight to its grid columns on load, so the `.panel-left` clip is now narrower (no trailing gap past the last thumbnail column). | 2026-07-09 |
| `autopilot-vote` | USER_GUIDE.md#autopilot--the-guided-workflow | The collapsed "Metadata" toggle now sits below the Good/Bad buttons rather than above them. | 2026-07-09 |
