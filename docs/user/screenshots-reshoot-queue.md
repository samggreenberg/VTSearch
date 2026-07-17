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

The full shot set was re-rendered on the GRID on 2026-07-12 (after the
light-ramp, zoom-removal/type-scale, responsive side-track, and
icon-unification changes), so every earlier row is drained. One shot is stale
again: the Achievements intro-copy simplification (cefadd7c, PR #2347) landed
*after* that reshoot, so `achievements` needs a re-render for that copy line
alone.

| Shot id | Embedded in | Why it's stale | Flagged |
|---------|-------------|----------------|---------|
| `achievements` | USER_GUIDE.md#achievements | Intro paragraph copy simplified (cefadd7c, 2026-07-12): the old "settings source configured / round-trip between sessions" line now reads "Your progress is saved with your settings, so it carries over between sessions". Landed after the 2026-07-12 full reshoot, so this copy line is the only difference. **Also (issue #2486):** the achievement progress bars now use the red→yellow→green `high-good` gradient (greener as they fill) instead of the old flat `var(--accent)` fill. | 2026-07-12 |
| `dashboard-loaded` | USER_GUIDE.md#what-vtsearch-does | The RAM / Disk usage bars now redden as they fill via the shared bar's `high-bad` polarity (green→yellow→red continuous gradient) instead of the old accent→warning(≥80%)→bad(≥95%) threshold fill (issue #2486). | 2026-07-14 |
| `dashboard-manage` | USER_GUIDE.md#dashboard--managing-datasets-and-detectors | Same RAM / Disk usage-bar recolor as `dashboard-loaded`: the bars in the framed action-bar row now use the `high-bad` green→yellow→red gradient instead of the old threshold fill (issue #2486). | 2026-07-14 |
| `importer-picker` | USER_GUIDE.md (demo importer) | The demo-dataset catalogue's "# Media" column (`demo.num_files`) now shows measured exact counts instead of the old per-category-average estimate for caltech256, places365, ucsf_documents, gtzan, urbansound8k, speech_commands_v2, ucf101_full, kth, hmdb51, and wikipedia_topics (issue #2355). The visible rows for the shown media type advertise different numbers than the captured PNG. **Also (#2358, half-media-types):** the media-type selector now includes a **Document** option (the UCSF demo moved off the Image list into a Document tab), and selecting it shows a "Convert to" selector for the convert-on-load target (image/text). **Also (faces demo):** the Image catalogue now lists two new rows, `vggface2_faces_s` and `vggface2_faces_m` (VGGFace2 in-the-wild faces). | 2026-07-15 |
| `new-detector` | USER_GUIDE.md#creating-a-detector | The Blank tab's "Example" label now carries a required `*` marker (issue #2377), matching the Trained-tab required fields. The captured PNG shows the bare "Example" label without the marker. | 2026-07-14 |
| `three-panel` | USER_GUIDE.md#the-three-panel-layout | Left-panel selection feedback changed (issue #2350): the active media item now renders a full inset accent ring (`box-shadow`) instead of just a solid accent left edge. The served/active item in the framed `.panel-left` shows the old left-edge treatment in the captured PNG. | 2026-07-14 |
| `browse-view` | USER_GUIDE.md#browse--exploring-a-dataset-spatially | The top-bar `Data:` / `Detector:` context pulldowns are now display-only on the browse view: the caret (▾) is gone and the triggers are disabled (the pair can only be switched from the Dashboard). The captured full-page PNG still shows the old interactive pulldowns with carets. | 2026-07-17 |
| `find-view` | USER_GUIDE.md#find--scoring-and-verifying | Same top-bar change as `browse-view`: the `Data:` / `Detector:` context pulldowns are now display-only on the find view (no caret, disabled trigger). The captured full-page PNG still shows the old interactive pulldowns. | 2026-07-17 |
