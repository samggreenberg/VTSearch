# Screenshot reshoot queue

A running list of doc screenshots that someone has **flagged as stale** — the GUI they
capture has changed, but the images under `docs/user/assets/` haven't been
re-rendered yet.

**Why this file exists.** Screenshots are captured by a Playwright harness
(`scripts/screenshots/refresh.sh`) that needs a real browser and a running app.
A session that changes the GUI can't always supply both, so instead of letting
that drift go silently unrecorded, the changing session adds the affected shot
id(s) here and a later session drains the queue.

**This is now the exception, not the rule.** The cloud container *does* ship a
chromium (see `CLAUDE.md` → "Environment Notes"), so the default is to reshoot
in the same session and never add a row at all. Queue a shot only when you
genuinely can't render it — no browser, or the shot needs a fixture
`ensure-fixtures.mjs` doesn't build.

See `docs/plans/user-docs-screenshots.md` for the full screenshot system
(manifest, harness, determinism knobs, embedding convention).

## How to use it

**When you change the GUI and genuinely can't reshoot:** for every shot
whose framed UI your change alters, add a row to the table below. The shot
`id` must match an entry in `docs/user/screenshots.manifest.ts` — the wiring
check (`scripts/screenshots/wiring-check.py`, gated in `run-tests.sh`)
enforces this, so a typo'd or renamed id fails the suite.

**When you have a browser and want to drain the queue:**

1. Run `scripts/screenshots/refresh.sh` to re-render the stale shots (or all).
2. Review `git diff --stat docs/user/assets/` like any diff.
3. Commit the regenerated images and **delete the drained rows** from the table
   below (leave the queue empty, not stale).

An empty table means "nothing is *queued*" — the desired resting state, but
not a guarantee that every screenshot is current. This file only records drift
that a session noticed and wrote down; a GUI change whose session neither
reshot nor queued leaves no trace here. To actually verify the committed
images, run `scripts/screenshots/check.sh` (a manual, browser-needing
pixel-diff against `docs/user/assets/` — not a `run-tests.sh` gate).

## Queue

| Shot id | Embedded in | Why it's stale | Flagged |
|---------|-------------|----------------|---------|
