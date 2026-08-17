# Screenshot reshoot queue

A running list of doc screenshots that are **known stale** — the GUI they
capture has changed, but the PNGs under `docs/user/assets/` haven't been
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
`ensure-fixtures.mjs` doesn't build (the standing `find-stats` case below).

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
3. Commit the regenerated PNGs and **delete the drained rows** from the table
   below (leave the queue empty, not stale).

An empty table means "no known-stale shots" — the desired resting state.

## Queue

The full shot set (20 shots × 2 themes) was re-rendered on 2026-07-29, so
every row flagged before then is drained. One row survives, because a
re-render alone cannot fix it: `find-stats` needs a *fixture* the harness
doesn't build.

| Shot id | Embedded in | Why it's stale | Flagged |
|---------|-------------|----------------|---------|
| `export-picker` | USER_GUIDE.md#exporting-your-work | The export tab bar gained an **Open in Website** tab (the `open_url` exporter, issue #2855), so the framed tab row no longer matches the captured PNG. | 2026-08-06 |
| `find-stats` | USER_GUIDE.md#find-scoring-and-verifying | The Detector Stats modal gained a **Training-domain overlap** section (coverage-atlas domain-shift): when another dataset with a matching embedder is loaded, a reference picker + a chip reading "N% of this dataset looks atypical vs &lt;dataset&gt; — likely domain shift / largely in-domain" now render above "Detector Accuracy". The 2026-07-29 reshoot did **not** drain this: the section only renders when a second dataset sharing the shot dataset's embedder is loaded, and `ensure-fixtures.mjs` builds `syn-imgs` (SigLIP) and `syn-patch` (DINOv2-patch) — different embedders, so no reference is offered. Draining this row needs a second SigLIP image dataset in the fixture set first. | 2026-07-20 |
