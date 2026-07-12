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

---

<!-- item-sep -->

<!-- item-sep -->

- **Consolidate the remaining ad-hoc token values** — the six alias tokens,
  the `--tracking-wide` letter-spacing token, and the two style-scanner checks
  (broken `var()` refs + deleted-alias usage) have shipped. What's still owed
  is the rest of the value-drift consolidation: ~11 ad-hoc opacity values that
  want a decorative-dim opacity token (distinct from the existing
  `--opacity-disabled`); raw z-indexes on the browse overlays
  (`browse-view.component.scss` `z-index: 1/2/3`) that want a canvas-overlay
  `--z-*` token; off-scale transition/animation durations and stray radii that
  should round onto the existing scales; hand-copied box-shadows that should
  resolve through `--shadow-sm/md/lg`; and reinvented selected-tile tints that
  should share one token. Each is judgment-heavy (which opacity sites are
  "decorative dim" vs. animation vs. disabled; which durations are interactive
  vs. keyframe timing that would change feel if rounded), so scope each before
  applying. **Files:** `_variables.scss`, the listed component SCSS. **Note:**
  the scanner (`.claude/scripts/style-check.py`) already flags any resurrected
  alias and any broken `var()` — extend it similarly if a new token added here
  is meant to be enforced.

<!-- item-sep -->

- **Promote shared component primitives** — "one role, N implementations" is
  the biggest structural debt. Each remaining primitive is tracked as its own
  GitHub issue and shipped one-per-PR; the umbrella stays here until the last
  is done. (Shipped: `.segmented-toggle`, folding the settings View/Focus
  control and the view-controls size/focus toolbar onto one primitive.) Still
  owed: a `.btn--toolbar` variant for the `.ivc-btn`/`.panel-btn` icon buttons
  (#2300); folding the horizontal `.tab-bar`/`.help-tabs` strips onto
  `.importer-tab-bar` (#2301); a vertical `.side-tab-bar` for the settings tabs
  (#2302); shared picker-card (#2303), data-table (#2304), empty-state (#2305),
  and pane-divider (#2307) primitives; unifying the 3 progress bars (#2306);
  deduping the verbatim `dataset-card`/`detector-card` + `goods-actions` SCSS
  (#2308); and folding hand-built inputs onto `.form-input`/`.form-select`
  (#2309). **Fix pattern:** extract a sanctioned shared class into
  `_components.scss` / `_picker-shared.scss`, then fold the bespoke copies onto
  it via markup + `@extend`. Ship incrementally (one primitive per PR) to keep
  diffs reviewable. **Files:** `_components.scss`, `_picker-shared.scss`, and
  the listed component SCSS/templates.

<!-- item-sep -->

<!-- item-sep -->

- **Add focus management to the shared `vt-modal`** — `app/components/modal/`
  has no focus handling at all: no initial focus into the dialog, no Tab trap,
  no focus restore on close, across all ~24 modals (grep finds no
  `cdkTrapFocus`/`FocusTrap`/`focus(` in the component or template). Keyboard
  and screen-reader users can tab out of an open modal into the page behind it.
  **Fix:** add `cdkTrapFocus` (Angular CDK a11y), move initial focus to the
  first interactive control (or the heading), and restore focus to the trigger
  on close — one change in the shared component fixes it app-wide. While here,
  verify the reported "Esc doesn't always dismiss the New Detector dialog"
  (V5): `modal.component.ts` still has an Esc keydown handler but the repro
  needs a live browser to confirm. **Files:** `modal.component.ts`,
  `modal.component.html`.

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
