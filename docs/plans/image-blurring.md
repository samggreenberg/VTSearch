# Blurring and unblurring sensitive images

Planned in #5689. The work is split into the issues under "Work items"; this
file holds the design they share. Built so far: the rules setting and its
evaluator (#5691), the override store and its routes (#5692), carrying
overrides across dataset copies (#5693), the frontend state service and
`vtBlur` directive (#5694), the Blur / Unblur button with the focused
viewer's blur (#5695), the directive on every DOM image site (#5696), the
Browse canvas's blur and peek (#5697), and video playback's blur and its
button (#5698).

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
  crop inherits its photo's fields. Each outside ingest service brings its own
  field names through the existing importers, so no new ingest path is needed:
  `blur_rules` is where an instance names the fields it receives.
- **A structured server-tier setting to copy**: `docs_links` (a normalised list
  with a rejected-entries report at startup), in `vtsearch/settings/models.py`.
- **A per-item state service to copy**: `VoteStateService`
  (`frontend/src/app/services/vote-state.service.ts`), signal-backed, optimistic
  write, polled for other clients.
- **A CSS filter hook on the focused viewer**: `imageFilter` in
  `frontend/src/app/components/center-panel/image-viewer/image-viewer.component.ts`.

What does **not** exist: any shared thumbnail component (about twenty sites
each write their own `<img>`). Every DOM image site carries `vtBlur`, the
focused viewer blurs itself, and the Browse canvas draws pre-blurred cells
(see "Blurred images" in `docs/FRONTEND.md`).

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

Built: the setting is documented in `docs/SETTINGS.md` and evaluated by
`vtscore.datasets.blur_rules.matches_blur_rules`. What the remaining work needs
from it: evaluate it server-side against the **raw** `custom_metadata` of the
media dict, not the display-formatted metadata the API returns.

### Where overrides live

Built: per dataset, in a `<stem>.blur.json` sidecar beside the container
(`vtscore/datasets/blur_state.py`), and carried across the flows that make a
new dataset from an old one (export and re-import, promote, combine). The
reasons and the mechanism are under "Blur state" in `docs/ARCHITECTURE.md`.

### The API

Built: `GET /api/medias/blurred` → `{"ids": [...]}` for the dataset in
`X-Dataset-Id`, and `PUT /api/medias/<id>/blur` with `{"blurred": bool}`, both
in `docs/api/medias.md`; `blur_rules` is dump-only in `GET /api/settings`.

### This is a shield, not access control

The blur is applied in the browser. The server keeps serving the real bytes,
because peeking needs them and because a blurred variant under the same URL
would collide with the day-long image cache. Anyone who opens the image URL
directly sees the image. That is the intended trade for this feature.

### Frontend: one service, one directive, one canvas path

Built: `BlurStateService` (per-dataset blurred sets, optimistic
`setBlurred`, a poll, fail-closed while loading), the `vtBlur` directive
with the CSS peek, the directive on every DOM site, and the Browse canvas's
pre-blurred copies in `ThumbStore` with a cross-fading peek, all described
under "Blurred images" in `docs/FRONTEND.md`. The canvas's hover preview
shows no image for a still.

### The button

Built: Blur / Unblur ends the image viewer's toolbar, sits in the centre
panel's item tools under a video, and `b` toggles either; a blurred video
plays under the app's own control bar. All of it is described under
"Blurred images" in `docs/FRONTEND.md`.

## Work items

### Dependencies

An arrow points from an issue to one that needs it merged first.

```text
#5691 rules setting
  └─> #5692 override store and routes
        ├─> #5693 carry overrides across dataset copies
        └─> #5694 state service and vtBlur directive
              ├─> #5695 button and focused viewer ──> #5698 video playback
              ├─> #5696 every DOM image site
              └─> #5697 Browse canvas

#5691 … #5698 (all eight) ──> #5699 docs and changelog
```

- #5698 needs both #5694 and #5695: it reuses the button #5695 adds.
- #5693, #5695, #5696 and #5697 do not depend on one another and can run in
  parallel once their parent has merged.
- The longest chain is #5691 → #5692 → #5694 → #5695 → #5698 → #5699.

<!-- item-sep -->

- [ ] #5691 — Blur rules setting and the rule evaluator (Sonnet)

<!-- item-sep -->

- [ ] #5692 — Override store, blurred-id set and the two routes (Opus)

<!-- item-sep -->

- [ ] #5693 — Carry overrides across dataset export, subset and promote (Sonnet)

<!-- item-sep -->

- [ ] #5694 — `BlurStateService` and the `vtBlur` directive (Opus)

<!-- item-sep -->

- [ ] #5695 — The Blur / Unblur button and the focused image viewer (Sonnet)

<!-- item-sep -->

- [ ] #5696 — Apply `vtBlur` at every DOM image site (Sonnet)

<!-- item-sep -->

- [ ] #5697 — Browse canvas blur and peek (Opus)

<!-- item-sep -->

- [ ] #5698 — Video playback blur and its button (Sonnet)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #5699 — User guide, operator how-to and changelog (Sonnet)

<!-- item-sep -->

## Open questions

- **Faces and their source photo.** Both inherit the same ingest fields, so
  both start blurred, but an override on one does not move the other. Linking
  them is possible through the source-item link if that proves annoying.
- **The document viewer.** A document's poster thumbnail is blurred with the
  other stills; the viewer itself (an embedded `<object>`) is left sharp.
- **Blur state as an export column.** Results exporters could offer the
  effective state as a column; nothing has asked for it yet.
