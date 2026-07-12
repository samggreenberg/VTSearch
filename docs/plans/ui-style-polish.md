# UI Style Polish

**What this is:** The open work distilled from two rendered/static UI style
reviews — a Playwright-driven visual audit (2026-05-27, `V#` ids) and a
comprehensive static SCSS/token/a11y audit of all 85 component styles
(2026-07-09, `§` sections). Their shipped findings (the `--opacity-disabled`
roll-out, the broken-token bugs, the Back-vs-Cancel fixes, the light-theme
contrast ramp) are already in `dev` and are **not** repeated here. What remains
is a set of independently shippable polish slices, each with component/file
pointers and a concrete approach.

Items are named (stable labels, never renumbered) and separated by
`<!-- item-sep -->` sentinels; when you ship a slice, delete only your item's
own lines and leave the sentinels intact (see the plan-file policy in
`CLAUDE.md`). Several items touch the shared SCSS in `frontend/src/scss/` and
the token file `frontend/src/scss/_variables.scss`, so two people editing those
at once should coordinate via the "Files" line on each item.

**Reference:** all of this audits against `docs/style-guide.md` and the token
system in `_variables.scss`. The static half is enforceable — extend
`.claude/scripts/style-check.py` (see the token item) so regressions get caught
without a browser.

**Screenshot note:** the type-scale item changes essentially every rendered
surface. When you ship it, add the affected shot ids to
`docs/user/screenshots-reshoot-queue.md` (see CLAUDE.md "Screenshot reshoots").

---

<!-- item-sep -->

- **Delete `zoom: 1.1` and rebuild the type scale** — the single
  highest-leverage polish item. `frontend/src/styles.scss:44` applies
  `html { zoom: 1.1 }`, which distorts every `px`/`vh` value in the app and
  forces the `100%`-vs-`100vh` workaround in `app.component.scss`. The scale
  itself is over-compressed: `_variables.scss:20-27` packs `--font-2xs` 0.7 /
  `--font-xs` 0.75 / `--font-sm` 0.8 / `--font-md` 0.85 / `--font-lg` 0.9rem —
  five steps inside a 0.2rem band, i.e. sub-pixel differences that don't read.
  **Fix:** rebase the scale to a ~1.12–1.2 ratio with a ~0.875rem base, remove
  the `zoom`, and re-tune the handful of surfaces that were compensating for it
  (the `100vh` workaround, any fixed heights that assumed the zoom). Do this
  first — most "text feels cramped" nits downstream dissolve once the scale is
  real. **Files:** `styles.scss`, `_variables.scss`, `app.component.scss`.
  **Reshoot:** queue the doc screenshots (this touches every view).

<!-- item-sep -->

- **Consolidate design tokens and extend the style scanner** — six alias
  tokens still live in all three theme blocks (`_variables.scss:156-161`,
  `:248-253`, `:322-327`): `--border-color`, `--bg-secondary`, `--bg-primary`
  (confusingly maps to `--bg-panel`), `--accent-color`, `--color-accent` (the
  same word order-swapped), and `--error`. Alongside them: 11 raw
  letter-spacing values with no `--tracking-wide` token, 11 ad-hoc opacity
  values, off-scale durations/radii, hand-copied box-shadows, reinvented
  selected-tile tints, and raw z-indexes on the browse overlays.
  **Fix:** codemod each alias to its canonical name and delete the alias; add
  `--tracking-wide`, a decorative-dim opacity token, and a canvas-overlay
  `--z-*` token; round stray durations/radii onto the scale. Then extend
  `.claude/scripts/style-check.py` to flag (a) `var(--x)` names that resolve to
  nothing and (b) use of the deleted aliases, so this can't regress without a
  browser. **Files:** `_variables.scss`, component SCSS using the aliases,
  `.claude/scripts/style-check.py`.

<!-- item-sep -->

- **Promote shared component primitives** — "one role, N implementations" is
  the biggest structural debt. Bespoke buttons bypass the `.btn` taxonomy:
  `.ivc-btn` (`center-panel/...:88`), `.panel-btn` (`right-panel/...:60`),
  `.vc-btn` (`view-controls/...:17`), plus the `.toggle-group`/`.toggle-btn`
  View/Focus segmented control (`settings-modal.component.scss:154-198`, custom
  padding/font/`--accent-color` fill). Tab strips are re-implemented ~7 times
  instead of extending `.importer-tab-bar`/`.importer-tab`
  (`_picker-shared.scss:12-45`): `.tab-bar`/`.tab-btn`
  (`new-detector-modal.component.scss:21-46`), `.help-tabs`/`.help-tab`
  (`keyboard-help-modal.component.scss:9-32`), and the *vertical*
  `.settings-tabs`/`.settings-tab` (`settings-modal.component.scss:13-47`,
  which can't `@extend` a horizontal bar — it wants a new shared
  `.side-tab-bar`). Also duplicated: 6 data tables, 6 picker-card copies (with
  3 different title sizes), 6 empty states, 3 progress bars, 3 pane dividers,
  verbatim `dataset-card`/`detector-card` and `goods-actions` SCSS, and
  hand-built inputs that skip `.form-input`/`.form-select`.
  **Fix:** extract sanctioned shared classes — a `.btn--toolbar` variant for
  the icon buttons, a `.segmented-toggle`, a shared vertical `.side-tab-bar`,
  and shared picker-card / table / empty-state / divider primitives — then fold
  the bespoke copies onto them via markup + `@extend`. Ship incrementally
  (one primitive per PR) to keep diffs reviewable. **Files:** `_components.scss`,
  `_picker-shared.scss`, and the listed component SCSS/templates.

<!-- item-sep -->

- **Modal polish package** — modal chrome is inconsistent across ~24 dialogs.
  Widths are ad-hoc (720/680/480/900/34rem, no shared scale); the "Close"
  button has no fixed identity (styled `.btn--primary` in some, secondary in
  others — e.g. `settings-modal.component.html:472` renders a purple-fill
  "Close" even though settings auto-save, reading as a commit action);
  `media-crop-modal` puts its actions *inside* the scrolling body with
  non-`.btn` classes in reversed order; `autodetect-progress` double-pads; and
  ~9 hard-coded nested `max-height` scroll regions live inside `.modal-body`.
  The `.new-detector-form` also pins `width: 480px`
  (`new-detector-modal.component.scss:64`) while the modal widens to 900px when
  the media picker mounts, leaving empty side gutters on the Blank form.
  **Fix:** add `--modal-w-sm/md/lg` tokens and adopt them; enforce "Close is
  never primary" (demote/relabel the settings Close to "Done"); move the
  media-crop actions into a real `.modal-footer`; replace the nested
  `max-height`s with `flex: 1; min-height: 0`; make `.new-detector-form` track
  the modal width. Also document the intentional close-less dialogs
  (`[showCloseButton]="false"` on new-detector, clipper-chooser,
  combine-detectors, resort-prompt, dialog-host) in `docs/style-guide.md §2.4`,
  or restore the `×` glyph. **Files:** the listed modal SCSS/templates,
  `_variables.scss`, `docs/style-guide.md`.

<!-- item-sep -->

- **Verify the New Detector Esc dismissal (V5)** — the shared `vt-modal` now
  has full focus management (`cdkTrapFocus` + auto-capture; initial focus, Tab
  trap, restore-to-trigger on close). What remains is the reported "Esc doesn't
  always dismiss the New Detector dialog" (V5): `modal.component.ts` has an Esc
  keydown handler (topmost-modal-only, unit-tested), but confirming the repro
  needs a live browser this cloud container lacks. Re-check once a browser is
  available; if it reproduces, the fix likely lives in how the New Detector
  flow's nested modals register on the open-modal stack. **Files:**
  `modal.component.ts`.

<!-- item-sep -->

- **Unify the icon system** — several inline SVGs are pasted 2–4× (eye/export/
  trash/combine, the tri-state checkbox), the center-panel toolbar uses raw
  Unicode glyphs (`⟲ ⟳ − +`) and a text "Reset", and there are three different
  success-check renderings. **Fix:** move the duplicated SVGs into the `vt-icon`
  registry and replace the glyphs/text with registry icons. Deep-verify each
  duplicate against the current `vt-icon` registry when planning (some may
  already be registered). **Files:** `vt-icon` registry + the listed templates.

<!-- item-sep -->

- **Simplify header and layout IA** — the 3-panel grid is declared *twice* with
  conflicting borders (`scss/_layout.scss:5` and `app.component.scss:296`, both
  `300px 1fr 300px`) — a dead, conflicting source of truth. The fixed 300px
  side panels don't scale on wide monitors; the burger menu and the top bar
  duplicate the same four destinations; the logo is a `mailto:` link
  (`app.component.html:98`) instead of navigating to the Dashboard; and there
  are three near-identical header icon-button classes (`.help-btn`,
  `.achievements-btn`, `.settings-btn`). **Fix:** delete one grid declaration;
  switch side tracks to `minmax(280px, 20%)`; drop the burger-vs-topbar
  duplication; point the logo at Dashboard and move "email us" into Help;
  collapse the three header button classes into one `.header-icon-btn`.
  **Files:** `_layout.scss`, `app.component.{scss,html}`. **Note:** the empty
  `mailto:` recipient on the separate "Email us" affordance is tracked in
  `standard-workflow-polish.md` (`mailto-recipient`) — same file, different
  line; coordinate.

<!-- item-sep -->

- **Accessibility foundation sweep** — verified still open: dashboard section
  titles are `<span class="dashboard-section-title">`, not headings
  (`dashboard.component.html:6,167`; only the login screen has an `<h1>`);
  `vt-dataset-card`/`vt-detector-card` use element selectors
  (`dataset-card.component.ts:19`) rendered `display: table-row`, which breaks
  table semantics; sortable `<th>` headers have no `aria-sort` and no keyboard
  path; radio groups lack `<fieldset>`/`<legend>`; settings-modal `<label>`s
  lack `for=` associations; some dashboard "+" buttons are `title`-only; and
  the Find wait message has no `aria-live`. **Fix:** promote section titles to
  real headings; switch the cards to attribute selectors
  (`selector: 'tr[vt-dataset-card]'`); add `aria-sort` + keyboard sort;
  add fieldsets/legends and `for=` associations; add `aria-label` on the "+"
  buttons and `aria-live` on the Find wait region. **Files:** the listed
  templates/components. (The three deleted focus outlines were restored in the
  2026-07-09 report's Phase 1 — exclude those.)

<!-- item-sep -->

- **Copy-style guide + sweep** — microcopy drifts three ways: ellipsis is
  written `...`, `…`, and `&hellip;`; Title Case and sentence case are mixed;
  one concept has several names (Train/Label, Find/Autodetect; "model" leaks
  into UI where the product word is "detector"); placeholder casing is
  inconsistent. **Fix:** add a copy-style section to `docs/style-guide.md`
  (sentence-vs-Title-case decision, `…` only, placeholder format, canonical
  product vocabulary), then sweep the templates to match. **Files:**
  `docs/style-guide.md` + template text.

<!-- item-sep -->

- **Dataset importer picker Back affordance (V6)** — the Add-Dataset picker
  (`source-picker.component.html`) uses persistent `.importer-tab-bar` tabs with
  no `.back-btn`; "going back" means clicking a different tab. The sibling
  New-detector › Trained › Server-JSON flow uses the canonical picker → `← Back`
  pattern, so the two diverge. This is a design-alignment call, not a mechanical
  fix (the footer `Cancel` there legitimately dismisses the whole dialog per the
  Back-vs-Cancel rule). Decide whether the picker should adopt the
  form-with-`← Back` shape or stay tabbed, then align. **Files:**
  `source-picker.component.{html,ts}`; cross-check CLAUDE.md "Nested-modal back
  buttons".
