# Empty / edge-state sweep — 2026-05-28

Task 2 of [docs/plans/browser-vision-testing.md](../plans/browser-vision-testing.md).
Driven via the Playwright MCP browser at 1440×900, light theme only. Server
was started with an empty `data/` (existing `data/` moved aside to
`data.task1-backup/`; embedder models and the synthetic generator cache were
symlinked back in so imports could run without re-downloading SigLIP). The
only persisted state at the start of the run was the user-settings file the
app creates on first boot.

Findings are tagged `V#` (visual), `U#` (UX), `O#` (observability), or
`B#` (functional bug). Severities are Low / Med / High.

## Coverage

| State | Screenshots | Captured |
|---|---|---|
| Cold boot — dashboard | `01-dashboard-cold-boot.png` | ✓ |
| Cold boot — data picker (empty) | `02-data-picker-cold-boot.png` | ✓ |
| Cold boot — detector picker (empty) | `03-detector-picker-cold-boot.png` | ✓ |
| Cold boot — settings modal (Appearance) | `04-settings-cold-boot.png` | ✓ |
| Cold boot — Add Dataset picker | `05-import-picker-cold-boot.png` | ✓ |
| Mid-load — embedder cold-load | `06-import-in-flight.png` | ✓ |
| Mid-load — embedding files 20/20 | `07-dashboard-one-dataset.png` (transient row) | ✓ |
| One dataset loaded, no detectors — dashboard | `07-dashboard-one-dataset.png` | ✓ |
| One dataset loaded, no detectors — data picker | `08-data-picker-one-dataset.png` | ✓ |
| New-detector modal (blank) | `09-new-detector-modal.png` | ✓ |
| Zero-vote detector — dashboard | `10-dashboard-zero-vote-detector.png` | ✓ |
| Detector picker with one detector | `11-detector-picker-one-detector.png` | ✓ |
| Train view, zero votes — autopilot | `12-train-view-zero-votes.png` | ✓ |
| Single-item dataset — dashboard | `13-dashboard-single-item-dataset.png` | ✓ |
| Single-item dataset — train view | `14-train-single-item-dataset.png` | ✓ |
| Single-item dataset — after voting the only item | `15-train-single-after-vote.png` | ✓ |
| Detector rename to 200 chars — server-error toast | `16-rename-detector-200chars-error.png` | ✓ |
| Detector row with 200-char name | `17-detector-200char-name-row.png` | ✓ |
| Dataset rename to 200 chars + detector 200 chars | `18-dashboard-200char-names.png` | ✓ |
| Data picker with 200-char dataset name | `19-data-picker-200char-name.png` | ✓ |
| Dataset name extended to 500 chars | `20-dashboard-500char-dataset-name.png` | ✓ |
| Settings — `solo_media_type` set to Image | `21-settings-solo-image.png` | ✓ |
| Dashboard with `solo_media_type` set | `22-dashboard-solo-image.png` | ✓ |
| Add Dataset picker with `solo_media_type` set | `23-import-picker-solo-image.png` | ✓ |
| Settings — `solo_media_type` reset to unset | `24-settings-solo-unset.png` | ✓ |
| Server-folder importer — bad path (nonexistent) | `25-server-folder-bad-path-form.png` | ✓ |
| Server-folder importer — Import disabled, no message | `26-server-folder-bad-path-button-disabled.png`, `27-server-folder-path-real-but-button-disabled.png` | ✓ |

Not exercised:
- Per-media `origin` / URL set to a 500-character string. The plan asks for
  this on one media item, but reaching it requires either a source file with
  a 500-char on-disk path (>filesystem limits in most layouts) or a backdoor
  through the API. Approximated by extending the **dataset** name to 500
  chars (`20-dashboard-500char-dataset-name.png`), which surfaces the same
  table-overflow class of bug. Capturing the per-media variant should follow
  up either via a Files importer with a renamed source or via a new
  fixture-loading helper.
- A real successful submit through the server-folder importer to observe the
  error-toast path on a path that fails server-side rather than client-side
  (see U2). The 200-char rename incident in V4/B1 stands in as a sample of
  the toast UI, but the importer-specific toast wasn't reached.

## Cold-boot state (no datasets, no detectors)

The empty-registry messaging is reasonably clear: each section shows
"No datasets yet. Click + to add one." / "No detectors yet. Click + to add
one.", the bottom action area says "Select a dataset" / "Select a dataset
and a detector", and both nav-bar pickers expose a `+ Add New …` action
inside their empty dropdowns. Nothing here looked broken.

### Findings

- **V1 — Low — Add-button CTA is in the section header, not where the
  empty-state text directs the eye.** The "No datasets yet. Click + to add
  one." text is centered, but the `+` is tucked in the top-right of the
  section header. New users have to hunt for it.
  Screenshot: `01-dashboard-cold-boot.png`. SCSS: most likely the dataset
  section template / `frontend/src/scss/_panels.scss`.

## One dataset loaded, no detectors

After a small synthetic-image import the dashboard shows a populated
Datasets table and the empty detectors section. The Detectors empty state
("No detectors yet. Click + to add one.") still reads, the bottom Train
button enables, and Find correctly stays disabled with the hint "Select a
detector".

### Findings

- **U1 — Med — Nav-picker placeholder stays "Select a dataset / detector"
  even when one row has been implicitly selected.** With one dataset
  imported, the bottom Train button enables (so the system clearly considers
  the dataset selected), but the nav-bar still reads "Data: Select a
  dataset". Same for the detector once you create one — the nav says
  "Select a detector" while the row is selected in the table. The nav only
  catches up after the user navigates away from the dashboard (e.g. into
  Train). The selection is ambiguous from the user's POV: are they
  pointed at something, or not?
  Screenshots: `07-dashboard-one-dataset.png`, `08-data-picker-one-dataset.png`,
  `10-dashboard-zero-vote-detector.png`. Likely culprit: the dashboard sets
  the in-table selection but doesn't push it into the global
  active-dataset/active-detector observable until a downstream view reads it.

- **V2 — Low — Picker dropdown's "active" affordance is a tiny dot vs a
  green checkmark, hard to read on hover.** When two datasets exist, the
  active one shows a green ✓ next to a bold name and the inactive one shows
  a gray ● next to a regular name (`19-data-picker-200char-name.png`); but
  when only one dataset exists, the row shows the *gray* dot even though
  it's the only / implicitly active option (`08-data-picker-one-dataset.png`).
  Together with U1 this makes "what's active?" unclear in the one-dataset
  case.

## Trained-detector, zero votes

Creating a new blank detector with a text seed lands the user back on the
dashboard with the detector row showing `# Training = 0`, `Last Trained = -`,
`Loaded? = ✗`, and the Find hint reading "Selected detector has no training
labels". Clicking Train opens the labeling view, which correctly starts on
"Find Initial Goods" stage with `0/3 good labels`. Both the Goods and Bads
panels show `(0)`.

### Findings

- **U3 — Low — Newly-created detector shows `Loaded? = ✗`.** The user just
  created it and the row is highlighted, but the Loaded column reads false.
  This is probably technically correct (no MLP exists yet because there are
  no labels), but it reads as "broken" rather than "freshly created". A
  state of "Empty" or "Awaiting labels" would be clearer than `✗`.
  Screenshot: `10-dashboard-zero-vote-detector.png`.

## Single-media dataset

Imported a synthetic image dataset with `size=1` named "Single". The
dashboard view (`13-dashboard-single-item-dataset.png`) renders the single
row correctly — `# Items = 1`, Loaded ✓. Train view loads the only image
fine. The behavior gets bad after the user votes.

### Findings

- **B1 — High — Autopilot demands more labels than the dataset contains,
  with no way out.** "Find Initial Goods" stage requires 3 good labels; in a
  1-item dataset that's impossible by construction. After voting Good on the
  only item, the autopilot UI does *not* advance, does *not* show "dataset
  exhausted" messaging, and does *not* unlock the next stage. The center
  pane goes completely blank but the bottom Bad/Good buttons remain present
  (clicking them does nothing). The metadata strip still shows the
  previously-voted item's name, leaving the user staring at metadata for an
  image that is no longer rendered.
  Screenshot: `15-train-single-after-vote.png`.
  Suspected source: the autopilot stage controller doesn't consider the
  dataset size when defining "Find Initial Goods" targets. Either cap the
  target at `min(3, dataset.size)` or render a clear "dataset exhausted"
  end-state.

- **O1 — Med — Cross-dataset media-id leak: 404 on `/api/medias/6/image`
  when switching from a 20-item dataset to a 1-item dataset.** As soon as
  the Train view opens against the new 1-item dataset, the client requests
  `/api/medias/6/image?dataset_id=<new>&detector_id=<x>` — id 6 was a valid
  index on the 20-item dataset but not on the new 1-item one. The browser
  console logs a 404 but the UI doesn't surface the error. Suggests the
  view state isn't fully reset when the active dataset changes.
  Screenshot: `14-train-single-item-dataset.png` (console errors visible
  in the captured page metadata).

## 200-char dataset / detector name overflow

Renamed the test detector to a 200-character string via the inline rename
input, then renamed "Single" to a 200-character string. Both attempts
revealed serious problems.

### Findings

- **B2 — High — Renaming a detector to a 200-character name causes an
  uncaught `FileNotFoundError` in the backend and a 500 response, but the
  in-memory rename succeeds on the client.** The backend uses the user-
  supplied name as the on-disk filename for the detector JSON
  (`<sanitized name>.json` and a `.tmp` swap file). At 200 characters the
  resulting filename exceeds the ext4 `NAME_MAX` (255 bytes) once the path
  prefix and `.tmp` suffix are added, so the atomic rename fails. The error
  toast that fires contains the *full filesystem path of both files*
  (`/home/samiam/Code/VTSearch/data/detectors/detector_with_a_very_long_name…json.tmp -> …json`).
  The UI optimistically updates the detector name in the in-memory registry,
  so after the failure the dashboard shows a detector that has no
  corresponding file on disk — a persistent divergence between UI state
  and persisted state.
  Screenshot: `16-rename-detector-200chars-error.png`,
  `17-detector-200char-name-row.png`.
  Suspected source: detector rename endpoint should either truncate /
  hash the on-disk name or surface a validation error before attempting
  the rename. The UI should not commit the rename optimistically.

- **V3 — High — Long names in either table collapse the entire data
  grid to just the `Name` column.** A single overflowing row pushes its
  cell wide enough that the other columns (Type, # Items, Created,
  Loaded?, Actions for datasets; Type, # Training, Autorun?, Last Trained,
  Created, Loaded?, Actions for detectors) get squashed off the right edge
  of the viewport. Critically, the row's *action buttons* (Rename / Stats /
  Delete or Rename / Import Labels / Export / Delete) become inaccessible,
  so the user can no longer revert or delete the offending row from the
  UI. Affects every row in that table, not just the overflowing one.
  Screenshots: `17-detector-200char-name-row.png`,
  `18-dashboard-200char-names.png`, `20-dashboard-500char-dataset-name.png`.
  Suspected source: the dataset / detector table uses an auto-layout that
  lets the Name cell grow without bound. Needs `max-width` + ellipsis on
  the Name cell, or a `table-layout: fixed` grid with a defined Name column
  width. SCSS: likely `frontend/src/scss/_tables.scss` or the table-row
  component styles.

- **V4 — Med — Error toast overlaps the nav bar and is large enough to
  block both nav-bar pickers.** When the 200-char detector rename failed,
  the error toast rendered near the top-right and visually overlapped the
  Data and Detector pickers (`16-rename-detector-200chars-error.png`). On
  the dashboard it also overlaps the column headers of the Datasets table
  below. Toast position should clear the nav-bar buttons.

- **V5 — Med — Error toast leaks absolute server filesystem paths.** The
  toast body includes the literal path
  `/home/samiam/Code/VTSearch/data/detectors/…`. In a multi-user / shared
  deployment that's sensitive; in any deployment it's not actionable for the
  user. The visible error should be the user-facing message ("Couldn't
  rename detector — name too long.") with the path moved behind the
  Details button or stripped from the toast entirely.

## 500-char dataset name

Extending the dataset rename from 200 to 500 chars did not produce a new
server error — the dataset rename succeeded — but reproduced the same V3
table-overflow with even more dramatic horizontal scroll. The nav-bar
button still truncates cleanly with ellipsis at 500 chars. This is a useful
proof point for V3: the table layout has no graceful behavior at any large
length, while the nav button's `text-overflow: ellipsis` does.

Screenshot: `20-dashboard-500char-dataset-name.png`.

(Per-media `origin` / URL with 500 chars was not exercised — see the
Coverage section. The expected failure shape, by analogy with V3, is the
same overflow class.)

## `solo_media_type` set vs unset

Setting Appearance → Solo media type to `Image` made the Scroll Style tabs
in Settings highlight `Image` and surface Left/Right Side configurations.
Resetting to `Show everything` collapsed the per-side configuration. No
breakage observed in Settings.

### Findings

- **U4 — Low — Setting `solo_media_type` does not visibly affect the Add
  Dataset picker.** The plan implies the solo type should narrow the UI to
  one media type. With `solo_media_type = Image` set and a fresh import
  modal opened, the Services / Server / Local / Demo tabs and their
  sub-types are unchanged from the unset state
  (`23-import-picker-solo-image.png` vs `05-import-picker-cold-boot.png`).
  If the setting is supposed to filter incompatible importers (e.g. hide
  audio-only sources when Image is solo), it's not doing so. If it's only
  meant to affect the labeling/sort surface, the wording in Settings
  ("Solo media type") doesn't make that scope clear.

## Import-failure / error state

Could not actually submit the server-folder importer against a nonexistent
path because the Import button stays disabled. Tried two paths
(`/tmp/nonexistent-vtsearch-edge-states-folder` and a real but empty
`/tmp/edge-states-empty-folder`) — both leave Import disabled with no
inline validation message. The input is `ng-valid` and the form has no
visible blocker, so the user has no signal of what's missing.

### Findings

- **U2 — Med — Server-folder importer disables Import with no
  explanation.** With Folder set to a path the user typed, Dataset Name
  blank (the placeholder says blank is OK), Include subfolders checked,
  MediaType Image — Import stays disabled. There's no inline error, no
  asynchronous validator status, no hover help text. The user is stuck
  with a disabled primary button and no idea what to fix. Suspected
  source: probably an async path-exists check that doesn't surface its
  pending / failed state into the UI. Either expose the validation status
  ("Path not found", "No image files found", "Checking…") or let the
  client submit and surface the server error in the toast like other
  failures do.
  Screenshot: `27-server-folder-path-real-but-button-disabled.png`.

- **U5 (related) — Low — Error toast is the only "real" error surface
  exercised in this sweep.** Because U2 blocked the importer's own error
  path, the 200-char rename incident (V4 / V5 / B2) is the only sample of
  the toast UI we have. The toast does render (with the issues called out
  in V4 / V5) and offers Details + Copy debug info + Dismiss buttons,
  which is the right shape; the per-importer error path wasn't reached.

## Mid-load partial state

Captured incidentally as part of importing the first dataset. Two distinct
in-flight states render cleanly:

1. Embedder cold-load (`06-import-in-flight.png`): row shows the dataset
   name, status "Loading dataset · embedding model", sub-status "Loading
   SigLIP weights. First-time only; cached on disk afterwards.", an
   indeterminate progress bar, and a Cancel button. All other table
   columns (Type, # Items, Created, Loaded?, Actions) are empty during
   load.
2. Per-file embedding (`07-dashboard-one-dataset.png`, captured mid-frame):
   status "Loading dataset · embedding files", sub-status "(20/20) Embedding
   SigLIP 20/20…", determinate progress bar near full, Cancel still
   present.

No findings here — this is the rare case where the UI actually does the
right thing on a long-running op. Worth re-using as a positive reference
when designing other progress states.

### Findings

- **O2 — Low — "First-time only" copy is mildly misleading on subsequent
  runs.** The "Loading SigLIP weights. First-time only; cached on disk
  afterwards." sub-status fires even when the embedder is already cached
  (my session's `data/models` was symlinked from a prior run, so the
  weights were on disk). The user sees the "first-time only" message
  every time, which undermines its informational value. Possibly a check
  for whether the snapshot already exists could swap the message to
  something like "Warming up SigLIP weights."
  Screenshot: `06-import-in-flight.png`.

## Patterns

- **Long-name overflow is the most consequential single class of bugs**
  found in this sweep. It surfaces as a backend filesystem error
  (B2 / detector-rename), a table-collapse that hides every other column
  and all action buttons (V3 / both tables, both 200-char and 500-char),
  an optimistic UI commit that diverges from disk (B2), and a sensitive
  path leak in the error toast (V5). The whole class needs an upstream
  fix: input length limits + filesystem-safe filename derivation + non-
  optimistic UI updates that wait for the server to confirm.
- **The nav-bar pickers and the dashboard's table selection do not share
  the same notion of "active"** (U1, V2). The nav picker only catches up
  once the user navigates into a downstream view, leaving a window where
  the bottom CTAs claim a dataset is selected but the picker disagrees.
- **Empty / impossible end-states in autopilot are not handled** (B1).
  Whenever a stage's labeling target exceeds the dataset's reach, the
  view goes blank and the user is stranded with no error and no
  affordance to advance.
- **Async validation states are invisible** (U2). Forms whose submit
  enables on a server-side check don't surface "checking…" / "failed
  because…" status, which makes a disabled primary button indistinguishable
  from a never-tried one.
- **The progress UX during long ops is genuinely good** — clear two-line
  status, a Cancel affordance, determinate vs indeterminate bars where
  appropriate. Worth treating as the reference for fixing the empty /
  error UX above.
