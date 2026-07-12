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
  is the rest of the value-drift consolidation, tracked as one GitHub issue per
  sub-slice (shipped one-per-PR; the umbrella stays here until the last is
  done): a canvas-overlay
  `--z-*` token for the raw z-indexes on the browse overlays
  (`browse-view.component.scss` `z-index: 1/2/3`) (#2321); and rounding
  off-scale transition/animation durations and stray radii onto the existing
  scales (#2322). Each is
  judgment-heavy (which opacity sites are "decorative dim" vs. animation vs.
  disabled; which durations are interactive vs. keyframe timing that would
  change feel if rounded), so scope each before applying. **Files:**
  `_variables.scss`, the listed component SCSS. **Note:** the scanner
  (`.claude/scripts/style-check.py`) already flags any resurrected alias and any
  broken `var()` — extend it similarly if a new token added here is meant to be
  enforced.

<!-- item-sep -->

- **Promote shared component primitives** — "one role, N implementations" is
  the biggest structural debt. Each remaining primitive is tracked as its own
  GitHub issue and shipped one-per-PR; the umbrella stays here until the last
  is done. (Shipped: `.segmented-toggle`, folding the settings View/Focus
  control and the view-controls size/focus toolbar onto one primitive.) Still
  owed: a `.btn--toolbar` variant for the `.ivc-btn`/`.panel-btn` icon buttons
  (#2300); and the shared data-table (#2304) primitive. **Fix pattern:** extract a sanctioned shared class into
  `_components.scss` / `_picker-shared.scss`, then fold the bespoke copies onto
  it via markup + `@extend`. Ship incrementally (one primitive per PR) to keep
  diffs reviewable. **Files:** `_components.scss`, `_picker-shared.scss`, and
  the listed component SCSS/templates.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Unify the icon system** (#2326) — several inline SVGs are pasted 2–4× (eye/export/
  trash/combine, the tri-state checkbox), the center-panel toolbar uses raw
  Unicode glyphs (`⟲ ⟳ − +`) and a text "Reset", and there are three different
  success-check renderings. **Fix:** move the duplicated SVGs into the `vt-icon`
  registry and replace the glyphs/text with registry icons. Deep-verify each
  duplicate against the current `vt-icon` registry when planning (some may
  already be registered). **Files:** `vt-icon` registry + the listed templates.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

