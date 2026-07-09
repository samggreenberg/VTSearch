# UI Style Review — 2026-07-09

**Status: Phase 1 (bug fixes) shipped; Phases 2–4 not started.** Comprehensive
style/structure/layout review of the VTSearch frontend: fonts, sizing,
organization, colors, contrast, component consistency, accessibility, and
information architecture. Findings are grouped by severity and theme, with
file:line references, and end with larger rework proposals plus a suggested
phasing.

**What shipped (Phase 1).** All of §1's verified broken/ghost tokens (incl.
`--font-mono` and a real `--z-tooltip` defined once, and the find-stats-modal
table styling gap); the two badge-contrast FAILs from §2
(`badge-ready`/`badge-embedding` white-on-fill in dark theme) via a new
`--badge-text-dark` token; the 3 focus-outline removals in §5/§8 (restored
after confirming none were wired to a hotkey/programmatic-focus path — the
app's `KeyboardService` already blurs `activeElement` on every shortcut); and
the 3 Back-vs-Cancel violations in §6 (footer Cancel now calls `close()`
instead of `back()`).

**Open follow-ups (Phases 2–4, not started).** The remaining §2 light-theme
contrast ramp (still uses the dark theme's pastel status palette — needs its
own darkened ramp), the `zoom: 1.1` / type-scale rework (§4), token
consolidation and alias cleanup (§3), the "one role, N implementations"
primitives and modal polish package (§5/§6/§10.4-5), the icon system
unification (§10.6), header/IA simplification (§7/§10.7), the remaining a11y
sweep items (§8: modal focus trapping, heading structure, table semantics,
`aria-sort`, fieldsets, `for=` associations), and the copy-style guide
(§9/§10.9) are all still open — see the phasing table below for grouping.

**Method.** Static audit of all 85 component SCSS files + 86 templates against
`docs/style-guide.md` and the token system in `frontend/src/scss/_variables.scss`;
the `/style-check` scanner (50 raw hits, curated below); computed WCAG contrast
ratios for every meaningful foreground/background token pair in the dark and light
themes; and four parallel deep-reads covering (a) the shell + 3 panels, (b) all
modals, (c) dashboard + browse surfaces, (d) template semantics/a11y/content style.

**What's already good** (so this reads fair): the token system is real and mostly
respected — zero hex literals in component SCSS, zero `font-weight: 700`, zero
`transition: all <raw-duration>`; every modal goes through the shared `vt-modal`
so `role="dialog"`/`aria-modal`/Escape/backdrop-close are uniform; icon-only
buttons almost always carry `title` + `aria-label`; there are no static
`style=""` attributes anywhere; and `prefers-reduced-motion` handling (OS +
app-level toggle, with a functional-progress exemption) is genuinely well done.

---

## 1. Verified bugs (broken references — fix first)

These are not style drift; they reference tokens that **do not exist** in
`_variables.scss`, so the declared value silently never applies.

1. **`var(--accent-text, var(--bg))` — both undefined.**
   `toast-container.component.scss:129` and `offline-banner.component.scss:48`
   set the text color of accent-filled buttons (toast primary action, offline
   Retry) to two undefined tokens. The `color` declaration is invalid at
   computed-value time, so the button inherits the surrounding text color — in
   the light theme that's near-black `--text-primary` on accent blue
   (unreadable); in dark it happens to be light grey and merely wrong. Should be
   `var(--btn-primary-text)`.
2. **`var(--weight-normal)` — undefined** (tokens are `--weight-regular/medium/semibold`).
   `browse-bin-popup.component.scss:443`: the grid item name label gets UA
   default weight instead of the intended 400.
3. **Find-stats tables render unstyled.** `find-stats-modal.component.html:14,32,54`
   use `class="stats-table"`, but no global `.stats-table` exists — the rule
   lives only as *local copies* inside `dataset-stats-modal.component.scss:11`
   and `detector-stats-modal.component.scss:8`, which Angular view encapsulation
   scopes to those components. The find-stats SCSS's header comment
   (`find-stats-modal.component.scss:1`) wrongly claims the class is shared.
4. **`var(--success-text, var(--text-primary))` — undefined.**
   `settings-modal.component.scss:311`: the HuggingFace "signed in" check icon
   is meant to be green and always renders `--text-primary` instead.
5. **`var(--color-success-bg, …)` / `var(--color-error-bg, …)` — undefined.**
   `autodetect-results-modal.component.scss:78,82` permanently fall back to raw
   `rgba(0,128,0,0.12)` / `rgba(200,0,0,0.12)` — hardcoded colors that ignore
   all three themes (`--good-bg`/`--bad-bg` exist for exactly this).
6. **`var(--danger, …)` — undefined** (`keyboard-help-modal.component.scss:168`);
   the real token is `--color-bad`.
7. **`var(--z-tooltip, 1000)` — undefined**, and the fallback collides exactly
   with `--z-modal-backdrop` (1000): `browse-hover-preview.component.scss:3`.
   A hover preview should not sit at the modal-backdrop layer.
8. **More ghost tokens leaning on fallbacks:** `--bg-elevated`
   (`dashboard/usage-bar.component.scss:36`, falls back to a non-theme grey
   rgba), `--bg-disabled`/`--bg-input`/`--bg-active`
   (`new-detector-modal.component.scss:279,290,338`), and two that are *only
   ever* fallbacks and never defined at all: `--font-mono` (8 call sites) and
   `--bg-selected` (`_picker-shared.scss:206,208`). Either define these tokens
   or reference the real ones; fallback-only tokens hide drift and confuse
   grep-based audits.
9. **Wrong-value fallback:** `var(--space-2xs, 4px)` at
   `find-stats-modal.component.scss:97` — `--space-2xs` is 2px; the fallback
   contradicts the token it names.

---

## 2. Color & contrast (measured)

WCAG 2.1 ratios computed from the actual token values (normal text needs 4.5:1
AA; large text and non-text UI need 3:1).

### Light theme — the weak spot

| Pair | Ratio | Verdict |
|---|---|---|
| `--status-yellow-sub` #f9a825 on `--bg-surface` | **1.97:1** | FAIL |
| `--status-green-sub` #66bb6a on `--bg-surface` | **2.36:1** | FAIL |
| `--status-yellow-label` #f57f17 on `--bg-surface` | **2.65:1** | FAIL |
| `--badge-embedding` bg with white `--btn-filled-text` | **2.81:1** | FAIL |
| `--status-red-sub` #e57373 on `--bg-surface` | **2.99:1** | FAIL |
| `--accent-light` #7c8aff on `--bg-surface` | 3.03:1 | large-only |
| `--text-warning` #b87900 on `--bg-surface` | 3.64:1 | large-only |
| `--header-text` white on `--header-bg` #4a7ad8 | 4.14:1 | large-only |
| `--color-good` #388e3c on `--bg-surface` | 4.12:1 | large-only |
| Disabled `.btn--primary` (0.5 opacity, effective) | **1.45:1** | FAIL |

The status-row sub/label colors are used at `--font-2xs`–`--font-sm` sizes in
dashboard status rows — small text at 2:1–3:1 is genuinely hard to read. The
light theme reuses the *dark* theme's pastel status palette in several slots
(`--status-red-sub`/`--status-yellow-sub`/`--status-green-sub` are lighter than
their own labels); it needs its own darkened ramp.

### Dark theme — mostly solid, three trouble spots

| Pair | Ratio | Verdict |
|---|---|---|
| `--btn-filled-text` white on `--badge-embedding` #e0a020 | **2.28:1** | FAIL |
| `--btn-filled-text` white on `--color-good` (badge-ready) | **2.78:1** | FAIL |
| `--text-dim` #666 on `--bg-surface` | 2.93:1 | FAIL (used for real copy, e.g. burger recent-times) |
| `--btn-primary-text` white on `--accent` #7c8aff | 3.03:1 | large-only — primary buttons are `--font-md` (13.6px), i.e. *not* large |
| `--text-placeholder` on `--bg-subtle` | 2.16:1 | placeholder-only, tolerable |

Badge fixes are cheap: dark text on the green/amber badge fills (the highviz
theme already does `--btn-filled-text: #000`), or dimmer fills with colored text
like `--badge-download` already does. For `.btn--primary` in dark, either darken
the accent fill slightly or accept it as brand-priority (4.5:1 would require
`#5a67e6`-ish).

### Non-text contrast (borders/dividers)

`--border-subtle` vs `--bg-surface` is **1.04:1** in dark (`#1e2030` vs
`#1a1d27`) — table-row dividers are effectively invisible in dark mode; light is
1.36:1. `--border` vs panel/surface is ~1.2–1.3:1 in dark. Hairlines are allowed
to be subtle, but 1.04 is below perceptible; and `--border` is also the
*input/button boundary* (WCAG 1.4.11 asks 3:1 for control boundaries). Inputs
survive because they sit on the recessed `--bg-subtle`, but the border itself is
doing almost nothing. Worth a deliberate decision: either brighten the ramp one
step or lean fully on fill-contrast and drop the pretense of hairlines.

### Disabled states

`--opacity-disabled: 0.5` on outline buttons yields ~4.2:1 (dark) / ~3.3:1
(light) — acceptable for disabled. But 0.5 *on top of an accent fill* (disabled
`.btn--primary`) collapses to 3.08:1 (dark) and **1.45:1 (light)**. The style
guide already documents the right pattern for the header (`--header-text-dim`
color shift instead of opacity, `_variables.scss:55-64`); disabled *filled*
buttons need the same treatment: swap to a desaturated fill + dimmed text
instead of opacity.

---

## 3. Design tokens: gaps and drift

- **Alias sprawl.** `--border-color`, `--bg-secondary`, `--bg-primary`,
  `--accent-color`, `--color-accent`, `--error`, `--error-text`, `--error-bg`
  are all live aliases (`_variables.scss:135-155`) with **46 combined uses** in
  component SCSS, alongside the canonical names (105 uses of `--border`, 72 of
  `--accent`, …). Two of them are the *same word order swapped*
  (`--accent-color` vs `--color-accent`), and `--bg-primary` confusingly maps to
  `--bg-panel`, not the body. One canonical name per role; codemod the aliases
  away and delete them.
- **No warning surface tokens.** Three hand-mixed amber chips, in two different
  ambers: `rgba(255,196,0,…)` (`new-detector-modal.component.scss:349-350`) vs
  `rgba(255,180,0,…)` (`combine-datasets-modal.component.scss:101`), plus
  `import-advanced`. `--text-warning` exists but has no `--warning-bg`/
  `--warning-border` companions, so every warning invents its own literal — and
  none adapt to highviz. Add the pair to all three theme blocks.
- **11 distinct letter-spacing values**, mixing units for the same "uppercase
  micro-label" role: `0.05em` (×8), `0.5px` (×3), `0.03em`/`.03em`,
  `0.04em`/`.04em`, `0.02em`, `0.06em`. Pick one token (e.g.
  `--tracking-wide: 0.05em`) and use it everywhere.
- **11 distinct opacity values** (0.35, 0.4, 0.45, 0.5-raw, 0.55, 0.6, 0.7,
  0.75, 0.8, 0.85, 0.9) despite `--opacity-disabled` existing to kill exactly
  this. Notable: autopilot uses *four different* dim levels for the same two
  semantic states depending on collapsed/expanded
  (`autopilot-panel.component.scss:69,74,164,175`); `.dimmed` card rows use 0.35
  (`dataset-card.component.scss:20`); left-panel disabled sections use 0.45
  (`left-panel.component.scss:22`). Decorative dimming needs its own one or two
  tokens (e.g. `--opacity-muted: 0.7`), and disabled must be the token.
- **Off-scale durations:** 180ms (toast/offline slide-ins), 0.18s (vote-swipe,
  duplicated in two files), 0.6s (`progress-bar.component.scss:20`, no token
  even close), raw `0.3s` where `--transition-slow` is exactly 0.3s
  (`app.component.scss:216-221`, `dashboard.component.scss:90,94`).
- **Hand-copied shadows:** `box-shadow: 0 4px 12px var(--shadow-dropdown)` is
  `--shadow-md` re-typed (`app.component.scss:72`); raw `rgba(0,0,0,0.3)` shadow
  at `browse-hover-preview.component.scss:10`; ad-hoc insets at
  `view-controls.component.scss:58`, `browse-view.component.scss:303`,
  `settings-modal.component.scss:192`.
- **Off-scale radii:** `2px` slider tracks (`browse-view.component.scss:342,362`),
  `1px` legend swatch (`find-stats-modal.component.scss:103`).
- **Selected-tile tint reinvented:** `color-mix(in srgb, var(--accent) 18%, transparent)`
  (`browse-bin-popup.component.scss:115,372`) where `--accent-highlight-bg`
  already encodes "accent at 20%". Also `color-mix` at
  `keyboard-help-modal.component.scss:67` and
  `detector-portable-export-modal.component.scss:16`.
- **Raw z-indexes** on browse overlays (1/2/3 at
  `browse-view.component.scss:58,144,277,304,331`) with siblings jumping to
  `--z-burger-menu` (2000, bin popup) and 1000 (hover preview). The canvas
  overlay layer needs one or two real `--z-*` tokens. Also semantically wrong:
  the burger dropdown borrows `--z-modal-backdrop` (`app.component.scss:74`).

---

## 4. Typography

### The scale is too compressed to do its job

Five of the eight sizes live within a 0.2rem band: 0.7 / 0.75 / 0.8 / 0.85 /
0.9rem. Adjacent steps differ by **0.05rem = 0.8px** (0.88px after zoom) —
below what anyone can perceive, which means the scale can't express hierarchy;
authors pick sizes by vibe and reviewers can't tell which was used. It also
explains recurring confusion like `font-size: var(--font-xs, var(--font-sm))`
(`label-list.component.scss:43`) — the author literally didn't know which one
it was, and it doesn't visibly matter.

A professional scale needs perceptible steps (~1.12–1.2 ratio). Suggested
consolidation (roughly: merge 2xs+xs, merge sm+md, keep lg as the pivot):

| Role | Today | Proposed |
|---|---|---|
| badges / micro-caps | 0.7 / 0.75rem | 0.75rem |
| body UI default | 0.8 / 0.85rem | 0.875rem |
| emphasized body / card titles | 0.9rem | 1rem |
| h3 / section | 1rem | 1.125rem |
| h2 / modal title | 1.1rem | 1.3rem |
| h1 / page | 1.4rem | 1.6rem |

### `html { zoom: 1.1 }` is a workaround, not a design

`styles.scss:43-45` scales the whole app 110% because the type baseline
(0.85rem ≈ 13.6px) was too small. It works, but: it makes every px value in
the codebase a lie (a "1px" border renders at 1.1 device px and rounds
unevenly), it forced the `100%`-vs-`100vh` workaround documented at
`app.component.scss:4-9`, and any JS that mixes `getBoundingClientRect` with
viewport units inherits a permanent 10% trap. The honest fix is a type-scale
rework (above): raise the base to ~0.875–0.9rem, delete the zoom, and let 1px
be 1px again. This is the highest-leverage single polish item in the codebase.

### Other typography findings

- **`<h3>` restyled below `<h4>`:** `label-list.component.scss:48-55` renders
  h3 at `--font-sm`/regular — smaller and lighter than the global h4. Same
  override at `detector-portable-export-modal.component.scss:7-9` (0.95rem, an
  off-scale size). The "compact uppercase panel header" is a real role — give
  it a shared class instead of bending heading tags (see §5).
- **No body line-height.** Headings set 1.3 (`_components.scss:24`); body text
  inherits UA `normal`. Multi-line helper copy (`.info-text`, importer
  descriptions) sets `line-height` ad hoc (1.25, 1.35, 1.4 all appear). Set a
  base `line-height: 1.45` on `body` and a `--leading-*` pair.
- **Eight monospace stacks.** `font-family: monospace` (5 sites),
  `ui-monospace, SFMono-Regular, Menlo, monospace`, the same + Consolas, and
  `var(--font-mono, …)` with three different fallback chains. Define
  `--font-mono` once (it's already the assumed name) and use it everywhere.
- **No sans font token either.** The stack lives only on `body`
  (`styles.scss:29`); fine today, but a `--font-sans` token costs nothing and
  future-proofs a brand-font change.

---

## 5. Component-system drift

The shared classes in `_components.scss` are good; the problem is how much of
the app bypasses or re-implements them.

### Buttons — the taxonomy is bypassed across every major surface

Bespoke button classes that re-implement padding/font/radius instead of
`.btn` + variant + size: `.panel-btn` (`right-panel.component.scss:60`),
`.ivc-btn` (`center-panel.component.scss:84`), `.vc-btn`
(`view-controls.component.scss:17`), `.load-sort-add-btn`
(`sort-bar.component.scss:95`), `.collapse-toggle`
(`autopilot-panel.component.scss:24`), `.train-rename-btn`
(`detector-context-bar.component.scss:36`), `.toggle-btn`
(`settings-modal.component.scss:159`), `.advanced-toggle`/`.lock-toggle`
(`new-detector-modal.component.scss:358,254`), `.btn-add-good`/`.btn-add-bad`
(`label-importer-modal.component.scss:97,131`), `.remove-btn`
(`examples-editor-modal.component.scss:38`), the media-crop footer buttons
(`class="primary"/"secondary"/"cancel"` with no `.btn` at all,
`media-crop-modal.component.html:26-35`), the login button (bare `button {}`
element selector, `login.component.scss:62-80`), and the dashboard Find/Train
buttons which restyle `.btn` with `min-width:120px; font-size: var(--font-xl)`
(`dashboard.component.scss:314-317`). Some of these (compact panel toolbars)
genuinely need a smaller/denser look than `.btn--xs` — which argues for adding
one sanctioned `.btn--toolbar` variant, then migrating.

### Forms

Hand-built inputs/selects that skip `.form-input`/`.form-select` (and thus get
different borders, focus states, and fonts): `inclusion-slider` number input
(`inclusion-slider.component.scss:14-28`), `.label-sort-select`
(`label-sort.component.scss:27-34`), `.text-sort-input`
(`sort-bar.component.scss:57-62`), `.bsp-sort-select`
(`browse-selection-panel.component.scss:130-137`), `combine-datasets`'
`.name-input` (value renders `--font-xl`, *larger than its own label*,
`combine-datasets-modal.component.scss:36-48`), login inputs, and
`achievements-tab`'s `.docs-phrase-input`/`.docs-phrase-submit`
(`achievements-tab.component.scss:256-279`). Label-role drift too:
`new-detector-modal`/`resort-prompt` use a bespoke `.col-label`
(semibold/secondary) where everything else uses `.form-label` (medium/primary).

### One visual role, N implementations

| Role | Count | Sites |
|---|---|---|
| Tab strip | **7+** | shared `.importer-tab` + `.view-tab`, then bespoke `.export-tab`, `.help-tab`, `.kbd-context-tab` (pill-fill active!), `.settings-tab` (left-border active!), `.tab-btn` (off-by-one underline math), `.clipper-tab` |
| Data table | **6** | `.dash-table`, `.demo-table` (near-identical twins, acknowledged in `_data-table.scss:6-9`), then `.results-table`, `.export-table`, `.stats-table` ×2 local copies, `.combine-table` (cells `--font-lg`, headers `--font-sm` — both off-convention) |
| Picker card | **6 copies, 3 title sizes** | shared `.importer-card` redeclared in `label-importer`, `settings-importer`, `settings-exporter`, `load-sort`, `resort-prompt`, `new-detector`; `.importer-name` is `--font-lg`/semibold (shared) vs `--font-xl`/medium vs `--font-md`/medium depending on modal |
| Empty state | **6+** | shared `.empty-state` plus `.empty-list`, `.placeholder` (`margin-top: 40%` hack, `app.component.scss:316`), `.empty`, `.pulldown-empty`, `.placeholder-text`, `.video-error` — five bespoke look-alikes; `.empty-state` is also overloaded for loading states |
| Progress bar | **3** | `vt-progress-bar` (6px track, `--radius-sm`), `usage-bar` (8px, `--radius-md`), achievements track (6px, `--radius-pill`) |
| Pane divider | **3** | identical 8px-hit-target/4px-line block in `find-view.component.scss:17-38,52-73`, `label-view.component.scss:26-47,60-81`, `browse-view.component.scss:158-178` |
| Segmented size button | **3** | `.vc-btn--size`, `.browse-size-btn` (`browse-view.component.scss:247-294`), `.bin-popup-size-btn` (`browse-bin-popup.component.scss:47-87`) — comments admit they "mirror" each other |
| Compact panel header | **3** | `.images-header-title` (plain el), restyled `<h3>` (label-list), `.train-context-label` — two sizes, three mechanisms |

### Verbatim copy-paste pairs (drift already visible)

- **`dataset-card` vs `detector-card`** duplicate ~90% of their SCSS
  (`.edit-btn`, `.load-btn`, `.delete-btn`, `.overflow-btn`, `.inline-edit`,
  keyframes). Already diverged: td padding `0.625rem` vs `--space-md`; error
  font `--font-sm` vs `--font-md`.
- **`.goods-actions`/`.goods-action-btn`** copied between
  `left-panel.component.scss:150-179` and `right-panel.component.scss:86-115`
  (the comment admits it), *and* the corresponding HTML action cluster
  (Browse / To Dataset / Export with inline SVGs) is duplicated in both
  templates.
- **`.sort-label` means two different things:** `--font-sm`/medium/secondary in
  the left panel (`sort-bar.component.scss:14`) vs `--font-2xs`/dim in the
  right (`label-sort.component.scss:10`) — same class name, sibling panels,
  visibly different.
- Swipe keyframes duplicated (`center-panel.component.scss:44-60`,
  `audio-player.component.scss:15-31`); `.sr-only` redeclared locally twice
  (`browse-bin-popup.component.scss:461`,
  `browse-selection-panel.component.scss:143`); `.warn-text` duplicated in two
  combine modals; `.required` redeclared in 5 modals; `.importer-form` /
  `.server-folder-browser` add inner `--space-2xl` padding inside an
  already-padded modal.

### Interaction-state drift

- **Focus rings:** global is `outline: 2px solid var(--accent); offset 2px`.
  Variants in the wild: offset `-2px` (media-item, bin-popup), `1px`
  (browse-view), box-shadow-instead-of-outline (`folder-browser.component.scss:62`),
  and three outright removals with no replacement —
  `left-panel.component.scss:55-57` (`.left-tab:focus-visible { outline: none }`),
  `center-panel.component.scss:168` (`.metadata-toggle`),
  `inclusion-slider.component.scss:23`. The removals are keyboard-a11y
  regressions.
- **Hover:** most controls swap background/color tokens; toast, offline banner,
  and autopilot use `filter: brightness(1.05–1.1)` instead — a different
  visual language that also ignores theme. One dead hover:
  `achievements-tab.component.scss:130-132` sets hover bg to the resting bg.
- **Menus:** `context-menu` uses `--radius-md`, `context-pulldown` and the
  burger dropdown use `--radius-lg`; context-menu disables via color-only while
  everything else dims.
- **Toggles:** pressed segmented buttons use `--toggle-active-text` in
  view-controls but `--btn-filled-text` in center-panel
  (`view-controls.component.scss:54-56` vs `center-panel.component.scss:105-108`).

---

## 6. Modals

The shared `vt-modal` gives every dialog a consistent skeleton (all 24 modals
use `<h2>` titles — good). The drift is in projected content:

- **Width anarchy.** 720px (export, dataset-importer), 680px (settings), 480px
  (label-importer ×4 repeats, new-detector), 900px via `:has(.media-picker)`
  (new-detector), `min-width:360/max-width:720` (keyboard-help), `max-width:
  34rem` (portable-export — the only rem one), natural width (the rest).
  Nothing shares a scale. Two or three width tokens
  (`--modal-w-sm/md/lg`: e.g. 480/680/900px) would cover every case.
- **"Close" has no fixed identity:** primary-filled in achievements
  (`achievements-modal.component.html:4`) and settings
  (`settings-modal.component.html:461`), secondary in all four stats modals.
  Dismiss labels roam: Close / Cancel / "Keep Current…" / OK / ×-only (export,
  keyboard-help have no footer button at all). Convention worth fixing: *Close
  is always `.btn` (default), never primary; primary is reserved for commit
  actions.*
- **Back-vs-Cancel violations** (the rule CLAUDE.md calls mandatory): in
  `label-importer-modal.component.html:66,176`,
  `settings-importer-modal.component.html:30,106`, and
  `settings-exporter-modal.component.html:30,100`, the footer **Cancel calls
  `back()`** — it returns to the picker rather than abandoning the dialog,
  while a `← Back` already exists at the top. `dataset-importer` does it
  correctly (footer Cancel closes). The importer/exporter family is internally
  inconsistent about what Cancel does.
- **media-crop-modal** puts OK / "OK but crop" / Cancel *inside the body* (they
  scroll away; `media-crop-modal.component.html:25-36`), primary-first (reverse
  of the app convention), with non-`.btn` classes.
- **Double padding:** `autodetect-progress` wraps its body in a div re-adding
  `--space-2xl` (`autodetect-progress-modal.component.scss:2`) — anti-pattern
  #8.
- **Nested scrollbars:** ~9 hard-coded `max-height`s inside the
  already-scrolling `.modal-body` (autodetect-results 300px, export 280px,
  examples-editor 250px, load-sort 200/300px, resort-prompt 300px, auto-find
  220px, keyboard-help `70vh` with its own custom webkit scrollbar). The
  `.modal-body` flex-column design exists precisely to avoid this
  (`_components.scss:243-255`); these should claim space with `flex: 1` +
  `min-height: 0`.
- **Footer/close affordance table** (from the full inventory): 5 modals have no
  footer; 4 hide the × on inner views (fine per Back rules) but `resort-prompt`'s
  outer view offers only "Keep Current" with no explicit cancel affordance
  beyond Esc/backdrop.

---

## 7. Layout & shell

- **The 3-panel grid is declared twice, with conflicting borders.**
  `_layout.scss:1-24` (left panel: right border; right panel: left border) vs
  `app.component.scss:293-311` (all panels right-bordered, then unset on
  `.panel-right`), the latter commented "Keep … for future phases". Dead,
  conflicting second source of truth for the core layout — delete one.
- **Panels are a fixed `300px 1fr 300px`** (both copies). On wide monitors the
  side panels get proportionally tiny; `minmax(280px, 20%)`-style tracks (or
  the user-resizable dividers find/label views already have) would scale
  better. The context pulldowns hard-truncate at `width: 200px`
  (`context-pulldown.component.scss:5`) regardless of available header space.
- **Header IA duplication.** The burger menu (`app.component.html:7-97`) and
  the top bar duplicate the same four destinations (Dashboard / Help /
  Achievements / Settings) as two parallel implementations — SVGs inlined
  twice, `div role="menuitem"` items vs real buttons. One of them can go (or
  the burger can become the *only* home of low-frequency items).
- **The logo is a `mailto:` link** (`app.component.html:98-100`): clicking the
  app logo opens an email compose window. Users expect logo → home/dashboard;
  feedback belongs in the burger/help menu.
- **Three near-identical header icon-button blocks** — `.help-btn`,
  `.achievements-btn`, `.settings-btn` (`app.component.scss:193-285`) are ~30
  lines each, byte-similar; one `.header-icon-btn` class would do.
- Magic numbers worth tokens or comments: burger dropdown `top: 40px`,
  `.placeholder { margin-top: 40% }` (percent-of-*width* fake centering),
  browse grid `minmax(0,1fr) 8px var(--browse-panel-width, 300px)`, volume
  slider thumb `margin-top: -4.5px`
  (`browse-view.component.scss:316-373` — the most off-token block in the app),
  `progress-bar.component.scss:24` `width: 30% !important` (the codebase's only
  `!important`).

---

## 8. Accessibility

Beyond the focus-outline removals (§5) and modal notes (§6):

1. **No focus management in any modal.** `modal.component.ts` never moves focus
   into the dialog, never traps Tab, never restores focus on close — keyboard
   focus stays on the launch button *behind* the backdrop for all ~24 modals.
   One `cdkTrapFocus` (CDK a11y) in the shared component fixes the entire app.
2. **Heading structure is missing above h3.** The only `<h1>` is the login
   screen; the Dashboard's "Datasets"/"Detectors" titles are `<span>`s
   (`dashboard.component.html:6,167`); Label/Find views start at `<h3>`. Give
   each top-level surface an `<h1>` (visually styled as today's header) and
   promote the dashboard section titles to `<h2>`.
3. **Dashboard rows aren't real `<tr>`s.** `vt-dataset-card`/`vt-detector-card`
   are element-selector components rendered `display: table-row`
   (`dataset-card.component.ts:19` + template starting at `<td>`), so the
   accessibility tree sees a custom element between `<tbody>` and `<td>`.
   Switching to `selector: 'tr[vt-dataset-card]'` (attribute selector on a real
   `<tr>`) preserves the architecture and restores table semantics.
4. **Sortable headers have no `aria-sort`, no keyboard path** — `<th (click)>`
   with a glyph-only indicator (`dashboard.component.html:75-91,236-252`).
   Column drag-reorder and resize handles are `mousedown`-only divs; panel
   dividers likewise (`label-view.component.html:52,69`,
   `find-view.component.html:38-41,72-75`) — no `role="separator"`, no
   tabindex, no arrow-key handling.
5. **No `<fieldset>`/`<legend>` anywhere** (29 radios/checkboxes across 9
   files). The "Sort:" / "Select:" pseudo-legends are plain spans
   (`sort-bar.component.html:3`, `select-mode.component.html:2`) — screen
   readers hear the option labels with no group name.
6. **Settings modal: 29 `<label>`s, zero `for=` associations** (e.g. label at
   `settings-modal.component.html:300` vs its select at `:308`), while
   `dataset-importer` mostly *does* associate. Inconsistent and fixable
   mechanically.
7. **Dashboard "+" buttons announce as "plus"** — `title` only, no
   `aria-label` (`dashboard.component.html:25-30,186-191`), unlike the labeled
   `+` buttons in sort-bar/center-panel.
8. **The Find "scoring dataset" wait message has no `aria-live`**
   (`find-view.component.html:46`) while the equivalent autodetect progress
   modal announces correctly.
9. `auto-find-settings.component.html:4-9` uses a `tabindex="0"` span with only
   `title` as a help affordance — not reliably announced; make it a labeled
   button or add `aria-label` + `role="img"`.

---

## 9. Content & copy style

- **Ellipsis three ways**, colliding on near-identical strings: `"..."` (~25
  sites, e.g. `settings-modal.component.html:3` "Loading settings...") vs `…`
  (~30 sites, e.g. `load-sort-modal.component.html:3` "Loading detectors…") vs
  `&hellip;` (`auto-find-settings.component.html:12,43`). Standardize on the
  Unicode `…`.
- **Button/label capitalization mixes Title Case and sentence case** on the
  same surfaces: "Add Corrections to Detector" / "Create & Import" vs "Add
  media to Bad", "Download bundle", "Copy debug info", "Dismiss all". Pick one
  (sentence case is the modern default; Title Case matches the current
  majority) and sweep.
- **One concept, multiple names.** The dashboard's primary actions: **Train**
  opens the *label* view via `onLabel()`; **Find** runs components named
  *autodetect* (`autodetect-progress-modal`, `autodetect-results-modal`).
  Users see "Train/Label" and "Find/Autodetect" seams in tooltips, headings,
  and docs. Also "gated AI models" (`settings-modal.component.html:405`) leaks
  "model" where the product says "detector".
- **Placeholder style drift:** `"e.g. dog barking sounds"` vs `"e.g. Dog
  Barks"` vs `"e.g. All Dog Barks"` vs `"e.g. all systems nominal"` — same
  convention, four casings; plus mixed instruction-vs-example placeholders.
- Modal h3 sections mostly Title Case noun phrases, with sentence-y exceptions
  ("What you'll get", "Missed vs. wrong matches by inclusion").

---

## 10. Rework opportunities (not constrained by the current setup)

Ordered by leverage; each is independently shippable.

1. **Type-scale rework + delete `zoom: 1.1`.** Rebase the scale per §4 (base
   ~0.875rem, perceptible steps), remove the zoom, re-tune the handful of
   surfaces that relied on it. Kills the vh/px distortions, makes 1px real,
   and gives the app genuine typographic hierarchy. This is the single change
   most likely to make the whole product feel more polished.
2. **Light-theme contrast pass + status/warning tokens.** Darken the light
   theme's status sub/label ramp to ≥4.5:1, add `--warning-bg`/`--warning-border`
   to all three themes, switch filled badges to dark-text-on-fill (or tinted-bg
   + colored-text like `.badge-download`), and replace disabled-filled-button
   opacity with a desaturated-fill treatment. Fixes every FAIL in §2.
3. **Token consolidation.** Define `--font-mono`; delete the 9 alias names via
   codemod; fix the 9 broken/ghost references (§1); add `--tracking-wide`, a
   decorative-dim opacity token, a canvas-overlay `--z-*` token; round stray
   durations/radii onto the scales. Then extend `.claude/scripts/style-check.py`
   to flag (a) `var(--x)` names not defined in `_variables.scss` and (b) alias
   usage — both classes of bug in §1 would have been caught mechanically.
4. **Promote the eight "one role, N implementations" patterns** (§5 table) into
   shared primitives — in order of visible payoff: picker card (6 copies, 3
   title sizes), tab strip (7 variants), stats/data table (6), empty state
   (6), pane divider (3), progress bar (3), segmented size button (3), compact
   panel header (3, and it un-bends the `<h3>` hacks). Merge `dataset-card`/
   `detector-card` SCSS and the duplicated `goods-actions` cluster into shared
   files while there.
5. **Modal polish package:** width tokens (`--modal-w-sm/md/lg`); "Close is
   never primary" + Cancel-always-abandons (fix the three Back-vs-Cancel
   violations); move media-crop actions into a real footer; remove the ~9
   nested-scroll max-heights in favor of `flex: 1; min-height: 0`; add
   `cdkTrapFocus` + initial-focus + focus-restore to `vt-modal` (one component,
   app-wide fix).
6. **One icon system.** Move the duplicated inline SVGs (eye, export, trash,
   combine, tri-state checkbox — each pasted 2–4×) into the existing `vt-icon`
   registry; replace the center-panel toolbar's Unicode glyphs (`⟲ ⟳ − +`) and
   text "Reset" with registry icons; standardize the three success-check
   renderings. The mixed SVG/glyph/text toolbar is the most visible polish
   issue in the app.
7. **Header/IA simplification.** Drop the burger-vs-top-bar duplication (keep
   the top bar; burger becomes overflow-only, or vice versa); logo navigates
   to Dashboard; move "email us" into Help; collapse the three header
   icon-button classes into one; let the panels breathe with
   `minmax(280px, 20%)` side tracks.
8. **A11y foundation sweep** (mostly mechanical): headings per surface (§8.2),
   `tr[vt-dataset-card]` attribute selectors, `aria-sort` + keyboard sort on
   table headers, fieldsets/legends for the radio groups, `for=` in settings,
   `aria-label` on the dashboard "+" buttons, `aria-live` on the Find wait,
   restore the three deleted focus outlines.
9. **Copy style guide.** One page in `docs/style-guide.md`: sentence-vs-Title
   Case decision, `…` only, placeholder format, and the canonical names
   (detector, not model; Train/Find naming resolved). Cheap, and it stops the
   drift at review time.

### Suggested phasing

| Phase | Contents | Risk |
|---|---|---|
| 1 (bug fixes) | §1 broken tokens, find-stats table, focus-outline restores, badge/status contrast, Back-vs-Cancel ×3 | Low — small, local diffs |
| 2 (foundations) | Type scale + zoom removal; token consolidation + scanner extension; light-theme ramp | Medium — visual diff everywhere, wants screenshot review |
| 3 (consolidation) | Shared primitives (§10.4), modal package (§10.5), icon system (§10.6) | Medium — many files, mechanical |
| 4 (IA & copy) | Header simplification, headings/a11y sweep, copy sweep | Low-medium |

Phase 2 changes GUI surfaces framed by essentially every doc screenshot; plan a
full reshoot via `scripts/screenshots/refresh.sh` (or queue all affected ids in
`docs/user/screenshots-reshoot-queue.md`) when it lands.

---

## Appendix: curated `/style-check` scanner results (2026-07-09 run)

50 raw hits. Disposition after review:

- **§4.1-2 raw px/rem (20)** — real: `detector-portable-export-modal` (7 hits,
  the least token-compliant file), `auto-find-settings` (6),
  `find-stats-modal` `font-size: 8px`, four `gap/padding: 2px` sites (→
  `--space-2xs`). Acceptable with comments: the two `0.625rem` "off-scale kept
  exact" pads (though dataset-card vs detector-card disagree — §5) and
  `source-specs-picker`'s checkbox-alignment 28px.
- **§4.7 heading restyles (2)** — both real (§4).
- **§4.13 `font: inherit` (1)** — `folder-browser.component.scss:93`,
  intentional per the skill's curation note; `new-detector-modal:332`'s
  `font-size: inherit` is the same trap uncaught by the scanner.
- **§4.14 column-without-gap (26)** — mostly benign top-level panels; the
  form-like containers worth a look are `settings-modal:15`,
  `dataset-importer-modal:26`, `source-picker:13,27`, and
  `load-sort-modal:83`.
- **§4.15 redeclared utility (1)** — `auto-find-settings:24` `.subhead`; the
  scanner misses the bigger §5 redeclarations (`.importer-card` ×6,
  `.empty-state` ×6, `.required` ×5, `.sr-only` ×2) because they live under
  different top-level names or in files it exempts — worth extending it.
