# Screenshot reshoot queue

Screenshots of the app that a GUI change has made stale, waiting to be re-rendered
at the next release. Each file in this directory other than this README is one
**entry**: the shots one change moved.

**Why a queue rather than a reshoot.** Two Playwright harnesses capture the app:
`scripts/screenshots/refresh.sh` for the user docs (`docs/user/assets/`) and
`slides/figs/src/shoot-ui-figs.mjs` for the slide deck's UI figures
(`slides/figs/ui-*.webp`). Either one boots the app, builds its fixtures and
re-renders, which costs a GUI-changing session far more than the change usually
did, and the next GUI change moves the same shots again anyway. So a session that
changes the GUI writes an entry here instead, and the **Dev2Main** release run
drains the whole queue in one pass ([`docs/RELEASE.md`](../RELEASE.md#4b-drain-the-screenshot-reshoot-queue)),
before the release PR opens, so `main` never ships a stale screenshot. Between
releases `dev`'s screenshots can lag its GUI; that is the trade.

**The exception is a brand-new shot.** A change that adds an entry to
`docs/user/screenshots.manifest.ts` captures that shot in the same session
(`scripts/screenshots/refresh.sh <new-id>`): a doc cannot embed an image that does
not exist yet, and the wiring check requires both theme files on disk. Only
*re*shoots are deferred.

## Adding an entry

Add **one new file per change**, named after the issue it carries and a short
slug: `4311-required-field-marks.md` (or just a slug when there is no issue).
One file per change, never a row in a shared table, is what keeps two parallel
PRs from conflicting here: two branches appending to the same table collide on
the same lines every time, while two new files never do.

Inside, a heading naming the change and one bullet per shot, the shot id as the
first backticked token:

```markdown
# #4311: required-field marks on New Detector and Add Dataset

- `new-detector` — the Name field now carries a required mark
- `importer-form` — the Path field does too
- `slides:steps` — the New Detector step frames the same field
```

- **A docs shot** is named by its manifest id. To find the shots your change
  touches, scan the manifest's `embeddedIn` / `caption` fields for the surface
  you modified (a modal, a panel, a toolbar).
- **A slide figure** is named `slides:<group>`, where the group is one
  `shoot-ui-figs.mjs` takes on its command line: `steps`, `make-detector`,
  `train-loop`, `find` or `region-voting` (its header comment maps each to its
  files).

Don't agonise over precision. The drain re-renders the whole docs set, so a shot
you missed is still caught and one you listed needlessly just doesn't move; the
entry's job is to say the drain is owed and why.

`scripts/screenshots/wiring-check.py` (a `run-tests.sh` gate) fails on an entry
naming an id that is no manifest shot and no slide group, so an entry cannot
outlive a renamed or deleted shot.

## Draining the queue

That is Dev2Main's job, in [`docs/RELEASE.md`](../RELEASE.md#4b-drain-the-screenshot-reshoot-queue):
re-render, commit the images, and delete the drained entries in the same commit.
An empty directory (this README alone) means nothing is *queued*, which is not a
guarantee that every screenshot is current: a GUI change whose session neither
reshot nor queued leaves no trace here. `scripts/screenshots/check.sh` is the
browser-needing pixel-diff that actually verifies the committed docs images.
