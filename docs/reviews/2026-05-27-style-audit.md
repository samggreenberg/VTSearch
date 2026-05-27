# Rendered Style Audit - 2026-05-27

**Status:** V7 / V10 / V11 / V13 shipped, plus the underlying systemic
fix (`--opacity-disabled` token, see §1.9 of `docs/style-guide.md`).
V1 / V2 / V3 (tab-bar duplication) and the remaining Low findings (V4 /
V5 / V6 / V8 / V9 / V12 / V14 / V15) are still open.



**Scope.** Task 1 of `docs/plans/browser-vision-testing.md`: a rendered
style audit of every major view in light and dark themes, comparing
against `docs/style-guide.md` and the CLAUDE.md "Nested-modal back
buttons" rule. Driven via the Playwright MCP at viewport 1440x900 on
`http://localhost:5000`.

**Setup note.** The prompt stated the dev server would have "one demo
dataset loaded and a trained detector"; the registry was actually empty
at audit start. I imported a 20-item synthetic-image demo dataset via
the UI and trained a `red shapes` detector with four votes (2 good, 2
bad) so the data-dependent views (Train/Find, dataset row, detector
row) could be captured. Two consequences:
- The "media-picker inner view" called out under New-detector in the
  task plan was not reached: the Blank flow uses inline text /
  Browse-files / drop-zone inputs, not a separate media-picker view.
  The Trained flow's "Server JSON File" inner view is captured instead.
- The Local-Files importer form was not opened separately; the
  Server-Folder form is captured as its peer and uses the same shared
  form chrome.

## Coverage

| View                                            | Light | Dark |
| ----------------------------------------------- | ----- | ---- |
| Dashboard (empty)                               | x     |      |
| Dashboard (loading - mid-import)                | x     |      |
| Dashboard (loaded, dataset + detector)          | x     | x    |
| Header context dropdown (Data:)                 |       | x    |
| Import-dataset modal: picker (no tab)           | x     | x    |
| Import-dataset modal: Services tab              | x     |      |
| Import-dataset modal: Server sub-tab strip      | x     |      |
| Import-dataset modal: Server-Folder form        | x     | x    |
| Import-dataset modal: Local sub-tab strip       | x     |      |
| Import-dataset modal: Demo sub-tab strip        | x     |      |
| Import-dataset modal: Demo-Downloaded table     | x     | x    |
| Import-dataset modal: Demo-Synthetic form       | x     |      |
| New-detector modal: Blank picker (no dataset)   | x     |      |
| New-detector modal: Blank w/ Image media locked | x     | x    |
| New-detector modal: Trained tab (source cards)  | x     | x    |
| New-detector modal: Trained > Server JSON inner | x     | x    |
| Settings: Appearance                            | x     | x    |
| Settings: Autopilot                             | x     | x    |
| Settings: Sorting                               | x     | x    |
| Settings: Data Imports                          | x     | x    |
| Help: Keyboard shortcuts                        | x     | x    |
| Help: User guide                                | x     | x    |
| Train (results grid) - empty                    | x     |      |
| Train (results grid) - 4 votes                  | x     | x    |

All assets are under `docs/reviews/assets/2026-05-27-style-audit/`.

## Findings summary

| ID  | View(s)                  | Severity | One-line                                                                  |
| --- | ------------------------ | -------- | ------------------------------------------------------------------------- |
| V1  | New-detector modal       | Med      | Custom `.tab-bar`/`.tab-btn` duplicate the shared `.importer-tab-bar`     |
| V2  | Help modal               | Med      | Custom `.help-tabs`/`.help-tab` duplicate the shared `.importer-tab-bar`  |
| V3  | Settings modal           | Med      | Custom `.settings-tabs`/`.settings-tab` (sidebar nav) - shared baseline?  |
| V4  | New-detector / Settings  | Low      | Modal-close (x) intentionally hidden, diverges from style-guide template |
| V5  | New-detector modal       | Low      | `Esc` does not dismiss the dialog from any inner focus                    |
| V6  | Import-dataset picker    | Low      | Top-level tabs serve as picker - no canonical "Back" chevron pattern      |
| V7  | Import-dataset / Demo    | Low      | "Needs Download" badge text reads very dim in dark mode                   |
| V8  | Settings (all tabs)      | Low      | Footer "Close" is styled primary even though there is no Save action     |
| V9  | Dashboard (empty)        | Low      | `.empty-state` panel reserves a large fixed height in empty card         |
| V10 | Dashboard / table header | Low      | Disabled "Combine selected" / "Delete selected" icon buttons very faint  |
| V11 | Header (always)          | Low      | Disabled "Dashboard" button text low contrast on accent-blue header bg   |
| V12 | Settings Appearance      | Low      | List/Grid + Click/Hover toggles are custom button styles, not `.btn`    |
| V13 | Help modal (dark)        | Low      | Scrollbar in modal body unstyled - default browser thumb on dark bg     |
| V14 | New-detector modal       | Low      | `.new-detector-form { width: 480px; }` is a hardcoded raw px value      |
| V15 | Import-dataset picker    | Low      | Picker-tab `<i>` icons appear muted/grey in light mode (only Demo flask reads colour) |

No findings were promoted to High. The audit did not surface a
broken-layout / unreadable-text bug in either theme; all issues are
style-guide compliance or minor-contrast.

### Systemic finding (added during fix-up)

The four contrast Lows above (V7, V10, V11, V13) all traced back to one
underlying habit: components reach for `opacity` to "tone something
down" without picking a shared value, and without checking whether
opacity is the right tool against the surface they're sitting on. A
grep across `frontend/src` turned up **seven different opacity values**
used to express the `:disabled` / `.disabled` state across button-like
elements: `0.35`, `0.4`, `0.45`, `0.5`, `0.55`, `0.6`, `0.75`. The
dashboard icon buttons sat at `0.35` (V10), the header "you are here"
button at `0.4` against the accent-blue header (V11), the importer
sub-tabs at `0.45`, and the global `.btn` baseline at `0.5`.

Resolved by introducing a single `--opacity-disabled: 0.5` token (see
`_variables.scss`; documented at §1.9 of `docs/style-guide.md`) and
collapsing the standard `:disabled { opacity: ... }` sites onto it
(`_components.scss`, `_picker-shared.scss`, `dashboard.component.scss`,
`dataset-card.component.scss`, `app.component.scss`,
`view-controls.component.scss`, `right-panel.component.scss`,
`export-modal`, `media-crop-modal`, `audio-crop-overlay`,
`label-importer-modal`). The few non-standard sites that mean
"something different" (`folder-browser` at `0.55` is the wait-cursor
state, `login` at `0.6`, achievement-tier opacities) are deliberately
left alone - they convey distinct semantics and shouldn't be conflated
with disabled.

The V11 fix also makes the broader point: **opacity is the wrong tool
for "disabled" when the surface is saturated**. White-on-accent-blue
dimmed via opacity cannot hold contrast, so the header's Dashboard
button now uses a `color: var(--header-text-dim)` shift instead. The
style guide now calls this out explicitly.

---

## Dashboard

**Screenshots.**
- `assets/2026-05-27-style-audit/dashboard-light-empty.png`
- `assets/2026-05-27-style-audit/dashboard-light-loading.png`
- `assets/2026-05-27-style-audit/dashboard-light-loaded.png`
- `assets/2026-05-27-style-audit/dashboard-dark-loaded.png`
- `assets/2026-05-27-style-audit/data-dropdown-dark.png`

Two stacked panels (Datasets / Detectors) over a footer action row
(Train / Find + RAM + Disk usage bars). Header has hamburger, logo,
Dashboard button, two context dropdowns (Data, Detector), and three
icon buttons (Achievements, Help, Settings).

### V9 - Empty-state panel reserves large fixed height

The Datasets and Detectors cards render at the same height even when
their body is just "No datasets yet. Click + to add one." (see
`dashboard-light-empty.png` - both panels are ~360px tall with a tiny
centred italic line of text floating in the middle). The fixed height
makes the empty state feel like a layout bug rather than an intentional
"nothing here yet" state. Severity: Low - desktop-only, plenty of
space, but the affordance is weak.

Suspected source: `frontend/src/app/components/dashboard/dashboard.component.scss`
(the section wrappers in `dashboard.component.html:46` and `:210` carry
the `.empty-state` element).

### V10 - Disabled icon buttons (Combine, Delete) very faint

In light mode, the disabled "Combine selected datasets" (chevron-right)
and "Delete selected" (trash) icons sit just above the panel border at
`opacity: 0.5` against a white card. The result is borderline
invisible. Once items are checked (see `dashboard-light-loaded.png`)
the icons darken to readable. The empty state would benefit from
hiding the icons or rendering them at slightly higher contrast.

Suspected source: shared `.btn:disabled { opacity: 0.5 }` rule in
`frontend/src/scss/_components.scss` combined with the icon-button
chrome in `frontend/src/app/components/dashboard/dashboard.component.scss`
(panel header right-aligned action group). Severity: Low.

**Fixed.** `.side-action-btn:disabled` and `.select-checkbox:disabled`
in `dashboard.component.scss` were sitting at `opacity: 0.35`; both now
use `var(--opacity-disabled)` (0.5). Combined with the systemic token
roll-out, every standard "this button is disabled" affordance now sits
at the same opacity.

### V11 - "Dashboard" disabled button vs. accent header bg

The persistent header bar uses the accent-purple background in both
themes. The "Dashboard" button is rendered in-place (you're on it,
hence disabled). The disabled text (`opacity: 0.5` x light text on
accent-blue) sits at ~3:1 contrast in light and is similarly muted in
dark. It is technically readable but reads as a layout artefact, not
"you are here." Severity: Low.

Suspected source: header layout in
`frontend/src/app/components/header/` (or wherever the top app bar
lives) combined with `.btn:disabled { opacity: 0.5 }` in
`frontend/src/scss/_components.scss`.

**Fixed.** Actual source was `.top-bar-btn:disabled` in
`app.component.scss` reaching for `opacity: 0.4`. Replaced with a
color shift: `color: var(--header-text-dim)`,
`background: transparent`, no opacity change. This is now the canonical
example in §1.9 of the style guide: opacity cannot hold contrast on a
saturated background; reach for a theme-aware dim color token instead.

---

## Import-dataset modal

**Screenshots.**
- Picker (no tab): `import-dataset-picker-light-empty.png`,
  `import-dataset-picker-dark.png`
- Services: `import-dataset-services-light.png`
- Server (sub-tabs): `import-dataset-server-light.png`
- Server > Folder (form): `import-dataset-server-folder-form-light.png`,
  `import-dataset-server-folder-form-dark.png`
- Local (sub-tabs): `import-dataset-local-light.png`
- Demo (sub-tabs): `import-dataset-demo-light.png`
- Demo > Downloaded Media (table): `import-dataset-demo-downloaded-light.png`,
  `import-dataset-demo-downloaded-dark.png`
- Demo > Synthetic Media (form): `import-dataset-demo-synthetic-light.png`

Two-tier tabbed picker: top tabs (Services / Server / Local / Demo),
then media-source sub-tabs (Folder / Files for Server+Local;
Downloaded Media / Synthetic Media for Demo). The selected sub-tab
swaps in either a form or a demo-table.

### V6 - Tab-driven picker has no canonical "Back" affordance

CLAUDE.md "Nested-modal back buttons" says "Any modal that switches
between an outer view and an inner view (importer picker -> importer
form ...) must render a left-aligned back chevron at the top of the
inner view." The Add Dataset modal sidesteps this by keeping both tab
bars persistently visible above the form, so "going back" is "click a
different tab." That works in practice but means there is no canonical
`.back-btn` chevron in this modal at all. The Server-JSON detector
flow follows the canonical pattern (V picker -> ← Back -> form) so
the inconsistency stands out.

Suspected source: `frontend/src/app/components/dashboard/dataset-importer-modal/source-picker/source-picker.component.html`.
Severity: Low (UX-debatable; calling out for design alignment, not
fixing as-is).

### V7 - "Needs Download" badge very dim in dark mode

The `.badge-download` pill on the Demo > Downloaded Media table reads
as ~light-grey text on near-same-shade pill background in dark mode
(see `import-dataset-demo-downloaded-dark.png`). The description
column ("Animals, nature, cities, & homes", "30sec music excerpts")
also reads as quite muted grey - acceptable for a secondary column,
but readable contrast suffers when the badge sits adjacent.

Suspected source: `frontend/src/scss/_picker-shared.scss`
`.badge-download` block + the `--bg-subtle`/`--text-muted` resolution
on `[data-theme="dark"]` in `frontend/src/scss/_variables.scss`.
Severity: Low.

**Fixed.** `.badge-download` had been pairing `background: var(--border)`
with `color: var(--text-muted)` - two near-equal shades of grey that
flattened against the dark table row. Now uses
`background: var(--bg-secondary-btn)` + `color: var(--text-secondary)`
in `_picker-shared.scss`. Light mode also benefits (a hair more contrast
on white).

### V15 - Picker-tab `<i>` icons mostly grey in light mode

In `import-dataset-picker-light-empty.png` the Services lightning bolt,
Server cabinet, Local home, Demo flask icons all render grey-on-white;
the flask glyph (Demo) is slightly darker. The picker-tab `<i>`
icons read as a uniformly muted strip that does not match the
descriptive copy below. Severity: Low (icons recover their
accent colour once their tab is active).

Suspected source: `frontend/src/scss/_picker-shared.scss`
`.importer-tab i { color: var(--text-muted); }` style on the inactive
state.

---

## New-detector modal

**Screenshots.**
- Blank picker (no dataset): `new-detector-picker-light.png`
- Blank w/ Image media locked: `new-detector-blank-image-light.png`,
  `new-detector-picker-dark.png`
- Trained tab (source cards): `new-detector-trained-light.png`,
  `new-detector-trained-dark.png`
- Trained > Server JSON File inner: `new-detector-trained-jsonform-light.png`,
  `new-detector-trained-jsonform-dark.png`

This dialog hosts a `Blank / Trained` tab pair; the Trained tab does
follow the canonical picker -> inner-view (← Back) pattern. Blank is
a single-page form with inline media-example inputs.

### V1 - Custom `.tab-bar`/`.tab-btn` duplicate the shared importer tabs

Style guide §2.6 and anti-pattern #15: "Tab strips inside modals use
the shared classes in `_picker-shared.scss`. Subclasses
(`.importer-subtab`, `.demo-tab`, etc.) extend the base via SCSS
`@extend`. ... Redeclaring shared utility classes locally" is listed
as a violation.

`frontend/src/app/components/dashboard/new-detector-modal/new-detector-modal.component.scss:21-46`
defines a private `.tab-bar` (display:flex, border-bottom) and
`.tab-btn` (padding/font/color/hover/active) that duplicate
`.importer-tab-bar` / `.importer-tab` from `_picker-shared.scss:12-46`.
The duplicate uses `--font-lg` and `--space-md var(--space-xl)`
padding, which is visibly *smaller* and *tighter* than the shared
importer tabs (compare `new-detector-picker-light.png` against
`import-dataset-picker-light-empty.png` - the two tab strips inside
sibling dialogs look subtly different).

Fix: replace the local class with `@extend .importer-tab-bar`
(structure) plus a `.importer-tab` markup change in the template, or
introduce a `.tab-bar--compact` variant in `_picker-shared.scss` if a
denser style is intentional. Severity: Medium - active drift between
two sibling modals.

### V2 - Help modal also uses its own `.help-tabs` / `.help-tab`

Same family of violation as V1.
`frontend/src/app/components/modals/keyboard-help-modal/keyboard-help-modal.component.scss:9-32`
defines `.help-tabs`, `.help-tab`, `.help-tab--active` instead of
extending `.importer-tab-bar` / `.importer-tab`. The visual difference
is small but the help dialog's active tab uses a darker underline
weight than the importer's.

Suspected source: as above. Severity: Medium.

### V3 - Settings modal sidebar uses its own `.settings-tabs` / `.settings-tab`

Same family of violation again.
`frontend/src/app/components/modals/settings-modal/settings-modal.component.scss:13-83`
defines `.settings-tabs` and `.settings-tab`. The settings nav is
*vertical* rather than the horizontal `.importer-tab-bar`, so a
straight `@extend` is not the right call; this one wants a new shared
class (e.g. `.side-tab-bar`) that the settings modal can extend. Worth
calling out as part of the V1 / V2 / V3 cluster: three modals, three
private tab implementations, no shared baseline for the new shape.

Suspected source: as above. Severity: Medium (needs design alignment
before a fix).

### V4 - Modal-close (×) intentionally hidden on multiple dialogs

`vt-modal` exposes `[showCloseButton]="false"` and several dialogs
opt in to hiding it:
`new-detector-modal`, `clipper-chooser`, `combine-detectors-modal`,
`resort-prompt-modal`, `dialog-host`. Style-guide §2.4's canonical
template includes `<button class="modal-close">×</button>` without a
toggle. Either the style-guide should mention "intentional close-less
flows" or the dialogs should pick up the close glyph for consistency.
Severity: Low (intentional, but undocumented divergence).

### V5 - Esc does not always dismiss the New Detector dialog

Reproduced during the audit: after navigating into Trained > Server
JSON File and pressing Escape, the modal stayed open and the click
that follows targeted the backdrop instead of the underlying
"Settings" button. The shared `<vt-modal>` has an Esc keydown handler
on `.modal-backdrop[tabindex=-1]` (`modal.component.ts:25-29`), so Esc
*should* bubble. My best guess is that focus had moved into one of
the form inputs that suppresses bubbling, or that Angular's view
encapsulation interferes with the focus path on this specific dialog.
The Add Dataset dialog dismissed cleanly on Escape. Severity: Low
(workaround: click Cancel or the backdrop), but worth follow-up since
the inconsistency is user-visible.

### V14 - Hardcoded `width: 480px` on `.new-detector-form`

`frontend/src/app/components/dashboard/new-detector-modal/new-detector-modal.component.scss:64`
sets `.new-detector-form { width: 480px; max-width: 100%; }`. Raw `px`
values for layout sizing aren't strictly forbidden by the style guide
(which targets spacing/font/radius tokens), but this width pins the
form to a different size than the surrounding modal-content, which
the modal scss already widens to 900px when the media picker is on
screen (`scss:15-18`). Result: the Blank form floats inside a wider
modal with empty side gutters. Severity: Low.

---

## Settings modal

**Screenshots.**
- Appearance: `settings-light-default.png`, `settings-dark-appearance.png`
- Autopilot: `settings-light-autopilot.png`, `settings-dark-autopilot.png`
- Sorting: `settings-light-sorting.png`, `settings-dark-sorting.png`
- Data Imports: `settings-light-dataimports.png`, `settings-dark-dataimports.png`

Two-column layout: vertical sidebar on the left (Appearance / Autopilot
/ Sorting / Data Imports) with content on the right. Footer carries
version stamp + Default / Import / Export / Close buttons.

### V8 - Footer "Close" is the primary accent button without a Save

The footer right-aligns `Default`, `Import`, `Export`, `Close`. Only
`Close` carries the accent-primary styling (purple-fill) in both
themes. Settings auto-save (there is no Save / Apply), so styling
Close as the primary action is OK in isolation, but the visual reads
as "Close = commit," which can confuse first-time users into
expecting a hidden Save action. Consider `Close` as default outline
or a verbal cue ("Close" -> "Done"). Severity: Low.

Suspected source: `frontend/src/app/components/modals/settings-modal/settings-modal.component.html`
footer area + the `.btn--primary` modifier on the Close button.

### V12 - Custom View Mode / Focus Mode toggle buttons

The Appearance tab's per-side View Mode (List | Grid) and Focus Mode
(Click | Hover) toggles render as a 50/50 split chip with the active
half filled with `--accent-light` and the inactive half with a faint
outline. They don't appear to use the shared `.btn` taxonomy
(no `.btn--primary` / `.btn--secondary`); the styling visibly
disagrees with the rest of the dialog (compare against the footer
`.btn` buttons in the same screenshot). Severity: Low.

Suspected source: `frontend/src/app/components/modals/settings-modal/settings-modal.component.scss`
(`.view-mode-toggle`, `.toggle-btn`, or similar) - look for `--accent-light`
backgrounds on `button` elements there.

---

## Help modal

**Screenshots.**
- Keyboard shortcuts: `help-light.png`, `help-dark-shortcuts.png`
- User guide: `help-light-userguide.png`, `help-dark-userguide.png`

Two-tab dialog (Keyboard shortcuts | User guide). The shortcuts tab
shows grouped `<kbd>` chips; the User guide tab renders a TOC + body
from the markdown copied in at build time.

V2 already covers the bespoke tab implementation.

### V13 - Scrollbar in modal body unstyled in dark mode

In both Help tabs the body is taller than the modal so a vertical
scrollbar appears on the right side of the modal-content. In dark
mode that scrollbar is the default browser thumb on a near-white
track (`help-dark-shortcuts.png`, right edge) - jarring against the
otherwise dark surface. The rest of the app uses a `--scrollbar-*`
custom-property treatment (visible on the Demo table inside the
import dataset modal in `import-dataset-demo-downloaded-dark.png`).
Severity: Low.

Suspected source: the markdown-body container in
`frontend/src/app/components/modals/keyboard-help-modal/keyboard-help-modal.component.scss`
likely overrides `overflow` without re-applying the project's
`scrollbar-color` / webkit thumb rule that lives in
`frontend/src/scss/_globals.scss` (or wherever the default scrollbar
treatment is declared - grep for `::-webkit-scrollbar`).

**Fixed.** The global `* { scrollbar-color: ... }` rule in
`_components.scss` should reach `.guide`, but the screenshot proves it
doesn't (likely a Chromium-on-Linux scrollbar-color quirk - the rule
holds zero specificity and competes with whatever the engine falls
back to per-overflow-context). Reasserted the theme scrollbar
explicitly on `.guide` (both `scrollbar-color` and the
`::-webkit-scrollbar*` variants), scoped to the Help modal body only.

---

## Train / results grid (label view)

**Screenshots.**
- Empty (just created detector): `results-grid-light.png`
- 4 votes (2 good / 2 bad): `results-grid-light-voted.png`,
  `results-grid-dark-voted.png`

Three-panel layout: left (Manual / Autopilot tabs, stage list), centre
(image preview + viewer controls + metadata + Bad / Good vote
buttons), right (Goods / Bads thumbnails with sort + view-mode
controls).

No new findings beyond the ones already captured above. The Bad /
Good action buttons use semantic red / green colour, which matches
`--color-bad` / `--color-good`; the panel headers ("Goods (2)" / "Bads
(2)") echo the same colours. The detector-hue bullet in the header
dropdown is correctly reflected.

One *positive* observation: the synthetic dataset's thumbnail strip
(right panel) stays bright against the dark background; the Goods /
Bads sectional dividers separate them cleanly. The autopilot stage
list on the left uses `--accent-bg` for the active stage chip in both
themes and reads correctly.

---

## Modals with `← Back` (CLAUDE.md verification)

Two flows in the captured set have an inner-view that should follow
the Back-not-Cancel rule:

1. **New-detector > Trained > Server JSON File** -
   `new-detector-trained-jsonform-light.png` /
   `new-detector-trained-jsonform-dark.png`: top-left `← Back`
   chevron rendered alongside the inner-view title "Server JSON File",
   *and* a footer `Cancel` (dismisses the whole dialog) +
   `Create & Import` (primary). Matches CLAUDE.md exactly. ✓
2. **Import-dataset > Server > Folder** -
   `import-dataset-server-folder-form-light.png` /
   `import-dataset-server-folder-form-dark.png`: footer `Cancel` +
   `Import`. There is *no* back chevron - the user steps back by
   clicking a different top-level tab. This is the V6 deviation
   already called out: a tabbed-picker design instead of the canonical
   picker -> Back flow. The dataset importer doesn't violate
   "Cancel as a Back button" (the Cancel here truly dismisses the
   whole dialog), it just doesn't use a Back chevron at all.

No "Cancel where it should be Back" or "Back where it should be
Cancel" violations found in the captured views.

---

## Out-of-scope items observed

- The `Esc` divergence (V5) is behavioural, not visual; flagged for
  follow-up.
- The synthetic-dataset import progress UX (visible in
  `dashboard-light-loading.png`) - inline "Loading SigLIP weights" copy
  is good UX. That belongs to Task 4 (long-op observability), not
  here.
- No mobile / narrow-viewport considerations were tested; per CLAUDE.md
  "Frontend Scope: Desktop Only," intentionally so.
- Performance, accessibility, and keyboard-focus traversal beyond what
  came up incidentally were not in scope.
