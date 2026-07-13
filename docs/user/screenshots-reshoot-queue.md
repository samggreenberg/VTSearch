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

*(empty - no known-stale shots. Queue drained 2026-07-12: the full shot set
was re-rendered on the GRID after the light-ramp, zoom-removal/type-scale,
responsive side-track, and icon-unification changes.)*

| Shot id | Embedded in | Why it's stale | Flagged |
|---------|-------------|----------------|---------|
