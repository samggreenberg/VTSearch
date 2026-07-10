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

Every shot below is captured in **both** themes (`themes: BOTH` in the
manifest). The 2026-07-09 light-mode neutral-ramp change (see below) alters the
**light** variant of *all* of them, so the whole shot set is listed here; a few
rows also carry an earlier, unrelated staleness reason.

**Light-mode neutral ramp (2026-07-09).** The light theme moved its gray off the
mostly-hidden desk and onto the surfaces the user actually looks at: `--bg-panel`
(all three layout columns) `#f1f3f8 → #e9ecf3`, `--bg-subtle` `#edeff4 → #e3e7f0`,
`--bg-body` `#e4e7ef → #dbdfea`, `--bg-hover` `#dbe0ec → #d3d8e6`,
`--bg-secondary-btn` `#dde1ec → #d8dce9` (cards/`--bg-surface` stay `#fff`). Every
light-theme frame's panel/background tone shifts, so each `.light.png` is stale.
This session had a browser but could not drain the queue: the repo pins
Playwright 1.60 (Chromium build 1223) while the container ships build 1194, and
the dev app server would not stay alive across the capture in this session's
process model — both environment blockers, not recipe problems.

| Shot id | Embedded in | Why it's stale | Flagged |
|---------|-------------|----------------|---------|
| `importer-picker` | `docs/user/USER_GUIDE.md#loading-a-dataset` | The shot's caption calls out "per-row readiness badges" (`.badge-ready`/`.badge-embedding`); the 2026-07-09 UI style review Phase 1 fix swapped their unreadable white text for a new `--badge-text-dark` token, changing the badge's rendered text color in both themes. Also: 2026-07-09 light-mode neutral-ramp change (see note above) shifts the light variant's background tone. | 2026-07-09 |
| `dashboard-loaded` | USER_GUIDE.md#what-vtsearch-does | §2 contrast fix: dark-theme table row dividers (--border-subtle) now perceptible; dim sub-text (--text-dim) lightened. Also: 2026-07-09 light-mode neutral-ramp change shifts the light variant's background tone. | 2026-07-09 |
| `dashboard-manage` | USER_GUIDE.md#dashboard--managing-datasets-and-detectors | §2 contrast fix: dark-theme table row dividers (--border-subtle) now perceptible; dim sub-text (--text-dim) lightened. Also: 2026-07-09 light-mode neutral-ramp change shifts the light variant's background tone. | 2026-07-09 |
| `three-panel` | USER_GUIDE.md#the-three-panel-layout | (1) Panels now snap tight to their grid columns on load (not just on divider drag), so the left/right panel widths — and the centre boundary between them — shift from the old restored-width layout. (2) The collapsed "Metadata" toggle moved from a full-width divider row above the Good/Bad buttons to a docked strip below them (center-panel tray now sits under the voting overlay). Also: 2026-07-09 light-mode neutral-ramp change shifts the light variant's panel tone. | 2026-07-09 |
| `results-grid` | USER_GUIDE.md#view-options | Left panel snaps tight to its grid columns on load, so the `.panel-left` clip is now narrower (no trailing gap past the last thumbnail column). Also: 2026-07-09 light-mode neutral-ramp change shifts the light variant's panel tone. | 2026-07-09 |
| `dataset-panel` | USER_GUIDE.md#loading-a-dataset | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts (see note above). | 2026-07-09 |
| `importer-form` | USER_GUIDE.md#loading-a-dataset | 2026-07-09 light-mode neutral-ramp change: light variant's modal/background tone shifts. | 2026-07-09 |
| `autopilot-vote` | USER_GUIDE.md#autopilot--the-guided-workflow | The collapsed "Metadata" toggle now sits below the Good/Bad buttons rather than above them. Also: 2026-07-09 light-mode neutral-ramp change shifts the light variant's panel tone. | 2026-07-09 |
| `autopilot-progress` | USER_GUIDE.md#the-collapsed-bar | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
| `manual-controls` | USER_GUIDE.md#manual-mode--for-power-users | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
| `region-voting` | USER_GUIDE.md#region-voting-on-images | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
| `view-options` | USER_GUIDE.md#view-options | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
| `settings-appearance` | USER_GUIDE.md#solo-media-type--streamline-for-one-media-type | 2026-07-09 light-mode neutral-ramp change: light variant's modal/background tone shifts. | 2026-07-09 |
| `browse-view` | USER_GUIDE.md#browse--exploring-a-dataset-spatially | The bottom-left canvas overlay (redundant dataset name + item count) was removed, and the item count now renders as a floater pinned to the minimap's bottom-left. Both themes' frames change. Also: 2026-07-09 light-mode neutral-ramp change shifts the light variant's panel/background tone. | 2026-07-10 |
| `export-picker` | USER_GUIDE.md#exporting-your-work | 2026-07-09 light-mode neutral-ramp change: light variant's modal/background tone shifts. | 2026-07-09 |
| `import-detector` | USER_GUIDE.md#importing-pre-trained-detectors | 2026-07-09 light-mode neutral-ramp change: light variant's modal/background tone shifts. | 2026-07-09 |
| `new-detector` | USER_GUIDE.md#creating-a-detector | 2026-07-09 light-mode neutral-ramp change: light variant's modal/background tone shifts. | 2026-07-09 |
| `find-view` | USER_GUIDE.md#find--scoring-and-verifying | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
| `find-stats` | USER_GUIDE.md#find--scoring-and-verifying | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
| `achievements` | USER_GUIDE.md#achievements | 2026-07-09 light-mode neutral-ramp change: light variant's panel/background tone shifts. | 2026-07-09 |
