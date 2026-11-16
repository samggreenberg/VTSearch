# Blurring and unblurring sensitive images

Tracking issue: #5689. Nothing here is built yet; this file is the whole plan.

## Goal

Shield users from images an outside ingest service has ruled sensitive (NSFW,
gore, …). Three behaviours:

- **Ingest rulings decide the starting state.** Per-instance server settings
  name the ingest metadata fields that mark an image as worth blurring, and how
  to read each one (a flag, or a numeric score against a threshold).
- **A Blur / Unblur button on every focused image** flips that image's state for
  good, in either direction: it overrules a wrong "sensitive" ruling and a wrong
  "safe" one alike.
- **Hover to peek.** Resting the pointer on any blurred image for more than a
  second gradually removes the blur; leaving puts it straight back. This holds
  everywhere an image is drawn: thumbnails, lists, the Browse canvas, the
  focused viewer.

## What already exists

- **The ingest channel.** Per-item metadata from an importer already lands in
  `media["custom_metadata"]`, survives the dataset pickle, and is served by
  `POST /api/medias/batch`. Converters copy it onto their outputs, so a face
  crop inherits its photo's fields. No new ingest path is needed for the
  rulings themselves; see the open question on how the outside service's
  fields reach an importer.
- **A structured server-tier setting to copy**: `docs_links` (a normalised list
  with a rejected-entries report at startup), in `vtsearch/settings/models.py`.
- **A per-item state service to copy**: `VoteStateService`
  (`frontend/src/app/services/vote-state.service.ts`), signal-backed, optimistic
  write, polled for other clients.
- **A CSS filter hook on the focused viewer**: `imageFilter` in
  `frontend/src/app/components/center-panel/image-viewer/image-viewer.component.ts`.

What does **not** exist: any per-item user state other than votes (which are
per detector, so the wrong home), any shared thumbnail component (about twenty
sites each write their own `<img>`), and any blur or moderation code.

## Design

### The state model: rules plus overrides

An image's **effective state** is

```
override, if a user ever clicked Blur/Unblur on it
else: does any blur rule match its custom_metadata?
```

Only the **overrides** are stored. The ruling is never copied into a stored
"blurred" bit, so changing the rules later (a new field, a different threshold)
immediately re-decides every image nobody has overruled, and leaves every
image somebody has overruled alone. An override is a plain boolean per item,
so both directions of "the ingest got it wrong" are the same write.

### The rules: `blur_rules`, a server-tier setting

A list in `data/settings.json`; an image is blurred when **any** rule matches:

```json
"blur_rules": [
  {"field": "Porn", "op": ">=", "value": 0.8},
  {"field": "Gore", "op": "truthy"},
  {"field": "Rating", "op": "in", "value": ["R", "X"]}
]
```

- Operators: `truthy`, `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`.
- `field` matches a `custom_metadata` key after trimming whitespace and a
  trailing colon, case-insensitively (the ingest side's `"Porn:"` should not
  silently miss).
- Values are coerced leniently: a numeric rule reads `"0.93"` as a number;
  `truthy` treats `"false"`, `"no"`, `"0"` and empty as false.
- A missing field, or a value that cannot be compared, does **not** match.
- A malformed rule is dropped and reported at startup, as `docs_links` does.
- Evaluated server-side against the **raw** media dict, not the display-formatted
  metadata the API returns.

Server-tier only: not writable through `PUT /api/settings`, shown read-only on
the settings modal's Server tab.

### Where overrides live

Per **dataset**, shared by every reader of that dataset, in a small JSON
sidecar beside the dataset container (`<stem>.blur.json`, `{media_id: bool}`).

- Not in the detector JSON: blur has nothing to do with a detector.
- Not a container entry: replacing an entry rewrites the whole ZIP, far too
  heavy for a click.
- A sidecar sharing the pickle's stem is already swept by
  `registry.unregister_dataset`.
- Written atomically, debounced like the detector writer.
- In memory: a non-derived slot on `DatasetContext` for the overrides, and a
  derived-cache slot for the computed blurred-id set (invalidated by an
  override, a media revision, or a rules change).

Media ids are per dataset file, so the sidecar is valid for exactly the file it
sits beside. The flows that mint a new dataset from an old one (subset,
promote, pickle export and re-import) must carry overrides across: the export
folds them into the container, and subset/promote remap them through the id
renumbering.

### The API

- `GET /api/medias/blurred` → the ids whose effective state is blurred, for the
  dataset in `X-Dataset-Id`. One compact list, because most display sites hold
  a bare id and no metadata (Browse cells, member grids, pickers).
- `PUT /api/medias/<id>/blur` with `{"blurred": bool}` → records the override.
- `blur_rules` appears, dump-only, in `GET /api/settings`.

### This is a shield, not access control

The blur is applied in the browser. The server keeps serving the real bytes,
because peeking needs them and because a blurred variant under the same URL
would collide with the day-long image cache. Anyone who opens the image URL
directly sees the image. That is the intended trade for this feature.

### Frontend: one service, one directive, one canvas path

- **`BlurStateService`** holds, per dataset id, the set of blurred ids as a
  signal; `isBlurred(datasetId, id)`; an optimistic `setBlurred(...)`; a poll
  for other clients' changes. Keyed by dataset because several sites show a
  sibling dataset's media (a face's source photo).
- **Fail closed while loading.** When the instance has blur rules and a
  dataset's set has not arrived yet, every image of that dataset counts as
  blurred. Otherwise the first paint leaks.
- **`vtBlur` directive** for every DOM site: `[vtBlur]="id"` plus an optional
  dataset id. It toggles a class; the peek is pure CSS:

  ```scss
  .vt-blurred       { filter: blur(var(--vt-blur-radius)); transition: none; }
  .vt-blurred:hover { filter: blur(0); transition: filter 1.5s ease-out 1s; }
  ```

  The one-second transition delay is the "more than a second"; `transition:
  none` on the resting rule is the instant snap back. The radius scales with
  the element's size (a 40px chip and a full-screen image need different
  radii), and each host clips its overflow so the blur does not bleed.
- **The focused viewer** adds a blur term to its existing `imageFilter` and
  gets the same hover rule.
- **The Browse canvas** is a 2D canvas, so CSS cannot reach a cell. `ThumbStore`
  caches a pre-blurred bitmap per blurred thumbnail (the way it already caches
  tinted audio masks); `drawHex` draws that instead. Peek reuses the existing
  `hoveredCell`: after one second on a blurred cell, cross-fade from the
  blurred bitmap to the sharp one over the same duration, and drop back at once
  when the hover ends. A blur-state change schedules a redraw.

### The button

In `vt-image-view-controls`, beside Zoom and Rotate: a text button reading
**Blur** when the image is sharp and **Unblur** when it is blurred. It calls
`BlurStateService.setBlurred`, so every other view of that image follows. The
control is mounted in the centre panel, the face-reference pane and the
source-photo modal; the modal passes the sibling dataset's id. Shortcut: `b`
(free today), listed in `utils/keyboard-shortcuts.ts`.

## Work items

<!-- item-sep -->

- **Blur rules setting** — `blur_rules` on `ServerSettings` with its
  normaliser, the rejected-rules startup report, the dump-only schema field,
  and a rule evaluator in the library tier (pure function: rules + metadata →
  bool) with tests for every operator and the lenient coercions. Docs:
  `docs/SETTINGS.md`, `docs/api/settings.md`. Regenerate the OpenAPI snapshot.

<!-- item-sep -->

- **Override store and API** — the `DatasetContext` slots (and their place in
  the derived/non-derived partition), the sidecar read/write, the two routes,
  the blurred-set cache and its invalidation. Tests: effective-state truth
  table, persistence across a reload, sweep on unregister. Docs:
  `docs/api/medias.md`, `docs/ARCHITECTURE.md` (the first per-dataset,
  per-item user state).

<!-- item-sep -->

- **Carry overrides across dataset copies** — fold into the pickle export and
  read back on import; remap through subset and promote.

<!-- item-sep -->

- **`BlurStateService` and the `vtBlur` directive** — the service, the
  fail-closed loading rule, the directive, the shared SCSS. Verify the peek
  timing in real chromium, not only jsdom: a CSS transition cannot be asserted
  in the unit suite.

<!-- item-sep -->

- **Focused viewer and the Blur / Unblur button** — the `imageFilter` term, the
  button, the `b` shortcut, the three mount points. Queue the toolbar
  screenshots under `docs/reshoot-queue/`.

<!-- item-sep -->

- **Apply `vtBlur` at every DOM image site** — `media-item`, `photo-grid`,
  `vote-grid` (session votes and labelset piles), `browse-bin-popup`,
  `browse-bin-member-grid`, `browse-selection-panel`, `people-view` (five
  sites), `autodetect-results-modal`, `dataset-example-picker`, and the crop
  modal when it was opened on a dataset image. `VoteGridEntry` needs the media
  id (and dataset id) added, since it carries only a precomputed URL today.
  Add a spec that fails when a template binds a `/api/medias/…/thumbnail` or
  `/image` URL without the directive, so a future site cannot forget it.

<!-- item-sep -->

- **Browse canvas blur and peek** — the pre-blurred bitmap cache in
  `ThumbStore`, the `drawHex` / `drawHoveredHex` switch, the hover timer and
  cross-fade, the full-resolution tier, redraw on state change.

<!-- item-sep -->

- **User docs and changelog** — a user-guide section on blurring and peeking,
  the operator's how-to for `blur_rules`, a `CHANGELOG.md` entry.

<!-- item-sep -->

## Open questions

- **How do the outside service's fields reach `custom_metadata`?** No in-tree
  importer reads a per-item metadata sidecar today; importer plugins supply
  metadata in code. If the ingest service arrives as its own importer plugin,
  nothing is owed here. If it writes a JSON or CSV beside the images for the
  stock folder importers to pick up, that reader is one more work item.
- **Shared or per-user overrides?** The plan stores one state per dataset,
  seen by every reader, which is how the issue reads ("update the state"). A
  per-user layer would let one user unblur without unblurring for everyone.
- **Which media types?** The plan covers every still-image display, which
  includes face crops and the poster thumbnails of videos and documents. The
  video *player* and the document viewer have no toolbar and are left sharp;
  blurring playback is a separate piece of work.
- **Labelset thumbnails with no dataset item.** A detector's stored label can
  point at media outside the loaded dataset, so there is no override and no
  dataset metadata to consult. Proposed: evaluate the rules against the label
  element's own `metadata`, and leave it sharp when that is empty.
- **Faces and their source photo.** Both inherit the same ingest fields, so
  both start blurred, but an override on one does not move the other. Linking
  them is possible through the source-item link if that proves annoying.
- **Blur state as an export column.** Results exporters could offer the
  effective state as a column; nothing has asked for it yet.
