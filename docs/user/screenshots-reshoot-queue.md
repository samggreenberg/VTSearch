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
| `new-detector` | USER_GUIDE → "Creating a detector" | The New Detector modal's blank-detector example input changed from a side-by-side Text / media two-column layout to a Text / `<MediaType>` tabbed layout. Also: the Detector Embedder Type picker moved into a new always-visible, collapsed "Advanced ▾" section (previously an inline field shown only on multi-embedder datasets). | 2026-06-28 · PR #2107; 2026-06-30 · detector-embedder-selection |
| `browse-view` | USER_GUIDE → "Browse — exploring a dataset spatially" | The hex/square bin-shape toggle was removed from the canvas toolbar (shape is now fixed by media type), and the framed image dataset now renders **square** tiles instead of hexagons. Also: the Selection panel header's text **Clear** button was replaced by a tri-state select-all checkbox ([ ] none / [-] some / [x] all-in-view). | 2026-06-29 · square-hex-thumbnails; 2026-07-01 · selection-tristate-checkbox |
| `settings-appearance` | USER_GUIDE → Settings → Appearance | (1) A new **HuggingFace** tab was added to the Settings tab strip, so the tab row framed alongside the Appearance pane shows an extra tab. (2) A new always-on "Browser motion: Allowed / Blocked" status line now sits directly under the Show Animations control, reporting whether the browser/OS prefers-reduced-motion gate is suppressing animations. (3) The **Show Animations** control changed from a checkbox toggle to a three-way pulldown (Show / Hide / OS Setting). | 2026-06-30 · gated-dataset-download-ux + vtsearch-animations-missing + animations-setting-pulldown |
