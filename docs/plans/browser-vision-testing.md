# Browser-Vision Testing

**Status:** Proposed. Drafted 2026-05-27. No runs executed yet.

**Scope:** Exercise VTSearch through the Claude Chrome extension on a
CPU-only Linux laptop, using Claude's ability to *see* the rendered
browser. The point is to catch issues that static analysis cannot:
visual regressions, layout overflow, dark-mode contrast, empty/edge
states, multi-step UX friction, and the observability of long-running
operations. Perf hunting via screenshots is explicitly out of scope:
use Chrome devtools / profilers for that.

The static [`style-check`](../../.claude/skills/style-check/SKILL.md)
skill is the complementary static-only pass; this plan covers what it
cannot see.

## How to use this doc

Each task below is a self-contained session. **Run them in separate
sessions** (separate transcripts, fresh context); the prompts assume
no prior shared state. Each task lists:

- **Goal**: what we want to learn.
- **Setup**: how to bring the app up to the right state before the
  vision pass.
- **Deliverable**: a markdown report under `docs/reviews/<date>-<scope>.md`
  with screenshots saved alongside (in
  `docs/reviews/assets/<date>-<scope>/`).
- **Draft prompt**: copy-paste into the laptop session.

Order suggestion (fastest → deepest):

1. Rendered style audit
2. Empty / edge-state sweep
3. End-to-end flow walkthrough
4. Long-running-op observability

## Shared setup notes

- **Branch**: any branch off `dev`. The vision pass is read-only on
  the codebase aside from the report file, so a throwaway branch is
  fine.
- **Server**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --local`.
  CPU-only is fine; embedder cold-loads will be slow, which is actually
  desirable for task 4.
- **Themes**: every visual task captures *both* light and dark themes.
  Toggle via the settings panel.
- **Viewport**: desktop only. VTSearch is desktop-only by policy
  (see CLAUDE.md "Frontend Scope: Desktop Only"); do not test mobile
  or narrow viewports.
- **Screenshots**: PNG, full-page where the extension supports it,
  otherwise viewport. Name `<view>-<theme>-<state>.png`
  (e.g. `dashboard-dark-empty.png`).
- **Findings format**: every finding gets a stable ID
  (`V#` Visual, `U#` UX, `O#` Observability), a one-line summary, a
  screenshot reference, and a severity guess (Low / Med / High).

## Task 1: Rendered style audit

**Goal.** Establish a baseline of how every major view actually
renders, in both themes, and produce a punch list of visual bugs the
static `style-check` skill cannot see: overflow, misalignment,
contrast, inconsistent spacing, Back-vs-Cancel placement, focus
rings, hover states, modal stacking.

**Setup.**

1. Boot the app with one demo dataset loaded and one trained detector
   present, so views aren't empty. `python app.py --local`, then
   import a demo dataset from the UI and vote on ~10 items to train
   a detector.
2. Have `docs/style-guide.md` open in another tab as the reference.

**Views to capture (each in light and dark):**

- Dashboard (dataset/detector pickers, settings entry)
- Import-dataset modal: picker view, then each importer's form view
- New-detector modal: picker view, then form view, then media-picker
  inner view
- Results grid (sorted view, with a few items voted)
- Settings modal (each tab)
- Help window
- Any modal that has a `← Back` button (verify chevron placement per
  CLAUDE.md "Nested-modal back buttons")

**Deliverable.** `docs/reviews/<date>-style-audit.md` with:

- Findings list (V1, V2, ...).
- Each finding: summary, screenshot path, suspected SCSS/component
  file, severity.
- Cross-reference any finding that the static `style-check` skill
  *also* flagged (so we know overlap vs. gaps).

**Draft prompt:**

```
You have browser vision via the Claude Chrome extension. The VTSearch
dev server is running at http://localhost:5000 with one demo dataset
loaded and a trained detector. Your job is a rendered style audit.

Read docs/style-guide.md and CLAUDE.md ("Nested-modal back buttons"
section) first; those are the rules.

Then drive the app through each view listed in
docs/plans/browser-vision-testing.md Task 1 ("Views to capture"),
in BOTH light and dark themes. For each view:

1. Take a full-page screenshot. Save under
   docs/reviews/assets/<today>-style-audit/ with the naming
   convention <view>-<theme>-<state>.png.
2. Compare to docs/style-guide.md. Note any visible violation:
   overflow, misaligned controls, inconsistent spacing, low-contrast
   text in dark mode, mis-styled focus rings, wrong button placement
   (especially Back vs Cancel per CLAUDE.md), modal-stacking glitches.

Produce docs/reviews/<today>-style-audit.md with:

- One section per view.
- Each finding: stable ID (V1, V2, ...), one-line summary, screenshot
  path, suspected SCSS or component file (use grep to confirm),
  severity (Low / Med / High).
- A short "Coverage" section at the top listing every view+theme pair
  you actually captured.

Out of scope: mobile/narrow viewports, perf, code changes. Do not
edit any source file other than the report and the screenshots.
Commit the report and screenshots to the branch when done.
```

## Task 2: Empty / edge-state sweep

**Goal.** Catch the UI states that are easy to overlook because dev
usually happens with realistic data loaded. Most regressions live
here: zero datasets, zero detectors, one media, very long names,
in-progress loads, error toasts.

**Setup.** Fresh container: delete `data/` (or run in a new clone),
so registries are empty. No login provider configured.

**States to capture:**

- Cold boot: no datasets, no detectors. Dashboard, both pickers.
- One dataset loaded, no detectors trained.
- One detector trained, zero votes since last train.
- Dataset with a single media item.
- Dataset name and detector name set to a 200-character string.
- Origin name / media URL set to a 500-character string (one media).
- Settings modal with `solo_media_type` set, then unset.
- Error states: trigger an import failure (point a server-folder
  importer at a path that doesn't exist), check the toast and modal.
- Mid-load: kick off a slow import, screenshot the partial state.

**Deliverable.** `docs/reviews/<date>-edge-states.md` with one section
per state and a findings list (V# / U# IDs). Highlight any state
where the UI silently shows a confusing empty box instead of a useful
"nothing here yet, do X" affordance.

**Draft prompt:**

```
You have browser vision via the Claude Chrome extension. The VTSearch
dev server is running at http://localhost:5000 with an empty data
directory (no datasets, no detectors). Your job is an
empty/edge-state sweep.

Drive the app through each state listed in
docs/plans/browser-vision-testing.md Task 2 ("States to capture"),
in light theme only (dark-mode coverage is Task 1's job). For each
state:

1. Set up the state (you may need to import a small demo dataset,
   rename things, or trigger failures).
2. Screenshot the relevant view(s). Save under
   docs/reviews/assets/<today>-edge-states/.
3. Note any UI failure: silent empty box where there should be a
   "get started" affordance, text overflow with long names, broken
   layout when a list has exactly one item, missing error messaging,
   stale state after a failed load.

Produce docs/reviews/<today>-edge-states.md with one section per
state, findings IDs (V1, V2, U1, U2, ...), screenshot paths,
suspected source files, and severity. Add a short "Coverage" section
listing every state you actually exercised (and any you tried to set
up but couldn't reach).

Out of scope: code changes, dark theme, perf. Commit the report and
screenshots when done.
```

## Task 3: End-to-end flow walkthrough

**Goal.** Drive the canonical user journeys and narrate UX friction.
Catches dead-end modals, missing back chevrons, unclear progress,
confusing terminology, journeys that *technically* work but feel
broken.

**Setup.** Empty data dir, server running. The walkthrough will
populate it.

**Flows:**

1. **Train-a-new-detector flow.** Import a demo dataset → vote on
   ~15 items (mix of good/bad) → save the detector → re-load it →
   run autodetect on the same dataset → export labels via a server
   JSON file exporter.
2. **Use-an-existing-detector flow.** Import a *different* compatible
   dataset → load the saved detector from flow 1 → autodetect → spot
   check a few high-score and low-score items → export.
3. **Multi-media import flow.** Use the multi-media importer with at
   least two source rows (e.g. an `image` output dataset with a
   `video → video2image` row and a `document → document2image` row).
   Verify the picker, the converter forms, and the final ingestion.
4. **Settings round-trip.** Open settings → change a few values
   (theme, volume, view mode) → export settings via local JSON file
   → reset → import the JSON back → verify everything restored.

**Deliverable.** `docs/reviews/<date>-e2e-flows.md` with one section
per flow. For each: a brief narration of the journey (step-by-step,
with screenshots at decision points), then a findings list focused
on UX issues, not pure styling. Look for:

- Dead-end modals (no obvious next action).
- Missing or mis-labeled back chevrons (CLAUDE.md Back vs Cancel).
- Unclear progress: did the user know the import was working?
- Terminology mismatches between the UI and what actually happened.
- Steps that required reading the docs to figure out.

**Draft prompt:**

```
You have browser vision via the Claude Chrome extension. The VTSearch
dev server is running at http://localhost:5000 with an empty data
directory. Your job is to drive the canonical user journeys
end-to-end and narrate UX friction.

Read CLAUDE.md ("Nested-modal back buttons") first.

Execute each flow listed in docs/plans/browser-vision-testing.md
Task 3 ("Flows"), in light theme. For each flow:

1. Walk through it step by step. Take a screenshot at each decision
   point (importer picker → importer form, modal transitions, the
   "did this work?" moments after each click).
2. Save screenshots under docs/reviews/assets/<today>-e2e-flows/
   with a sequence number prefix so they read in order
   (e.g. 01-flow1-importer-picker.png).
3. As you go, log every moment of UX friction: anything where you
   had to pause and look around for the next action, anything where
   the UI's word for something didn't match what it actually did,
   anything where you'd expect a back chevron and didn't find one,
   anything where progress was opaque.

Produce docs/reviews/<today>-e2e-flows.md with:

- One section per flow, with a numbered step list and inline
  screenshot references.
- A findings list per flow (U1, U2, ...), each with the step number
  it occurred at, a one-line summary, screenshot path, suspected
  source file, and severity.
- A short "Cross-flow patterns" section at the bottom calling out
  any friction that recurred across multiple flows.

Out of scope: dark theme, deep styling nits (Task 1's job), code
changes. Commit the report and screenshots when done.
```

## Task 4: Long-running-op observability

**Goal.** Watch what the UI actually communicates while long
operations are in flight. The CPU-only laptop is a feature here:
embedder cold-loads and full-dataset re-embeddings will be slow
enough that the progress UX gets a real workout. Catches: stalled
spinners with no text, progress bars that finish before the work
does, missing cancel buttons, UI that goes stale after a
background job completes.

**Setup.** Empty data dir. Server running. Be prepared to wait
several minutes per operation.

**Operations to observe:**

- First-ever demo dataset import (cold embedder load + ingestion).
- Re-import of the same dataset (warm embedders; should be faster).
- Train-and-rerank cycle after voting on ~20 items.
- Concurrent imports: kick off a second dataset import while the
  first is still embedding. Verify the `_download_gate` /
  `_embed_gate` behaviour from CLAUDE.md (second can download while
  first embeds, but not embed concurrently with default 1/1 limits).
- Cancel mid-load: kick off an import, hit cancel, verify the UI
  recovers cleanly and the partial dataset is not left registered.
- Autodetect on a freshly loaded large-ish demo dataset.

**For each operation, capture:**

- Screenshot at 0% (just kicked off).
- Screenshot at ~50% (or whenever the UI changes meaningfully).
- Screenshot at 100% (just finished).
- Any *intermediate* screenshot where the UI looks misleading
  (frozen, no text, stale).
- Wall-clock duration (rough; we're not benchmarking, just
  contextualising the screenshots).

**Deliverable.** `docs/reviews/<date>-longops.md` with one section per
operation, findings list (O# IDs), and a short "Patterns" section
at the bottom calling out cross-cutting observability gaps
(e.g. "every long op shows a spinner with no text for the first 20
seconds while embedders cold-load").

**Draft prompt:**

```
You have browser vision via the Claude Chrome extension. The VTSearch
dev server is running at http://localhost:5000 on a CPU-only laptop
(embedder cold-loads will be slow; that's the point). Your job is to
observe what the UI communicates during long-running operations.

Read CLAUDE.md ("Concurrent dataset loading") for context on the
download/embed gates.

For each operation listed in docs/plans/browser-vision-testing.md
Task 4 ("Operations to observe"):

1. Kick it off. Note wall-clock start.
2. Screenshot at 0% (just started), at any meaningful intermediate
   state (every ~30 seconds is fine), and at 100%. Save under
   docs/reviews/assets/<today>-longops/ with sequence prefixes
   (01-import1-0pct.png, 02-import1-30s.png, ...).
3. Specifically watch for: spinners with no text, progress bars
   that finish before the work does, UI that goes stale after a
   background job completes (e.g. the dashboard still shows
   "loading" after the dataset registered), missing cancel
   affordance, cancel that doesn't actually stop the work.

Produce docs/reviews/<today>-longops.md with:

- One section per operation. Step-by-step screenshots with
  timestamps and brief narration ("at 45s the spinner is still
  showing but progress text says 'embedding 230/500'").
- A findings list per operation (O1, O2, ...), each with a one-line
  summary, screenshot path, suspected source file, severity.
- A "Patterns" section at the bottom for cross-cutting gaps.
- A "Concurrency gate behaviour" subsection confirming whether the
  download/embed gate split actually does what CLAUDE.md describes
  (second import can download while first embeds, etc.) — or
  reporting the divergence.

Out of scope: hard perf measurement (we have profilers for that),
dark theme, code changes. Commit the report and screenshots when
done.
```

## Open follow-ups

- Pair Task 1 with a parallel `/style-check` run and cross-reference
  the findings; produces a "static caught / vision caught / both"
  breakdown that tells us what the static skill actually misses.
- If Task 4 surfaces consistent observability gaps, fold them into a
  follow-up `progress-ux.md` plan rather than fixing piecemeal.
- The first run of any task is also a test of *the testing setup*
  itself (extension reliability, screenshot ergonomics, prompt
  clarity); expect to refine the draft prompts after run 1.
