# VTSearch Frontend Style Guide

This is the canonical reference for visual styling in the Angular frontend. **All component SCSS must use the tokens and classes defined here** - raw `px`/`rem` values for padding, margins, gaps, font sizes, border radii, and transition durations are not allowed. Hex color literals are not allowed. The shared classes in `frontend/src/scss/_components.scss` are the source of truth for buttons, form fields, modals, and typography; one-off restyling of these elements is not allowed.

The frontend is **desktop-only**. There are no responsive breakpoints, touch-targeted controls, or mobile layouts to design for.

---

## 1. Design tokens

Tokens live in `frontend/src/scss/_variables.scss` as CSS custom properties on `:root` (with overrides on `[data-theme="light"]` and `[data-theme="highviz"]`). Reference them with `var(--name)`. **Never hardcode a value that has a token.**

### 1.1 Spacing - `--space-*`

Use the spacing scale for `padding`, `margin`, `gap`, and similar layout offsets. The scale is sized so each step roughly doubles, giving clear visual rhythm without dozens of arbitrary increments.

| Token         | Value     | When to use |
|---------------|-----------|-------------|
| `--space-2xs` | 0.125rem (2px) | Badge inner padding, hairline offsets |
| `--space-xs`  | 0.25rem (4px)  | Tight icon/text gaps, list-item gaps |
| `--space-sm`  | 0.375rem (6px) | Default button gap, small padding |
| `--space-md`  | 0.5rem (8px)   | Standard toolbar/header gap, default control padding |
| `--space-lg`  | 0.75rem (12px) | Card padding, table cell padding |
| `--space-xl`  | 1rem (16px)    | Section padding, modal-body bottom margin |
| `--space-2xl` | 1.5rem (24px)  | Modal-content padding, page-section spacing |

**Rounding rule:** any hardcoded `5px`/`7px`/`10px`/`14px`/`20px` value not produced by a token is a smell. Round to the nearest token; tiny visual shifts (1–3px) are acceptable in exchange for a consistent rhythm.

### 1.2 Font sizes - `--font-*`

| Token         | Value           | When to use |
|---------------|-----------------|-------------|
| `--font-2xs`  | 0.6875rem (11px) | Badges, tiny meta text |
| `--font-xs`   | 0.75rem (12px)   | Table headers, captions, subtitle rows |
| `--font-sm`   | 0.8125rem (13px) | Form labels, secondary helper text |
| `--font-md`   | 0.875rem (14px)  | **Default** for body UI text, buttons, inputs, tables |
| `--font-lg`   | 1rem (16px)      | Card titles, prominent secondary headings |
| `--font-xl`   | 1.125rem (18px)  | `<h3>`, section headers |
| `--font-2xl`  | 1.25rem (20px)   | `<h2>`, modal titles |
| `--font-3xl`  | 1.5rem (24px)    | `<h1>`, page-level header, modal-close glyph |

The scale steps 1px at a time through the small sizes (11-14px, where a step
must stay subtle) and widens to a ~1.12-1.2 ratio from `--font-lg` up, so
adjacent sizes read as genuinely different. Every step lands on a whole pixel
at the default 16px root.

There are no other font sizes. `0.78rem`, `0.83rem`, `0.95rem`, `12px`, `13px`, etc. are all violations - they collapse into the table above.

**Monospace:** `--font-mono` is the one monospace stack (`ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`). Reach for it whenever a face needs to be fixed-width (`.form-hint`, `.form-textarea`, code-ish values); never write a bare `monospace` or a hand-rolled font stack.

### 1.3 Font weights - `--weight-*`

Use these named weights, not raw numbers. The semantic mapping is enforced so a casual reader can tell *what role* an element has just from its weight.

| Token              | Value | When to use |
|--------------------|-------|-------------|
| `--weight-regular` | 400   | Body copy, table cells, descriptions, info text |
| `--weight-medium`  | 500   | Form labels, table headers, subtitles |
| `--weight-semibold`| 600   | Card titles, active tab labels, names in lists, anything that should "pop" without screaming |

There is no `--weight-bold` (700). If you find yourself wanting `font-weight: 700`, you're either reaching for size (use a larger heading) or you're styling around the layout (fix the layout). The one exception is the `<strong>` tag itself, which the browser styles bold by default.

### 1.4 Border radius - `--radius-*`

| Token          | Value  | When to use |
|----------------|--------|-------------|
| `--radius-sm`  | 3px    | Badges, scroll thumbs, small chips |
| `--radius-md`  | 4px    | **Default** for buttons, inputs, panels |
| `--radius-lg`  | 6px    | Cards, importer/exporter tiles |
| `--radius-xl`  | 8px    | Modals, prominent surfaces |
| `--radius-pill`| 999px  | Pill-shaped progress tracks, status pills |

`50%` is fine for circular avatars. Anything else (`2px`, `10px`, `1px`) is a violation.

### 1.5 Colors

All colors are theme-aware CSS variables defined in `_variables.scss`. There are dark (default), light, and high-contrast themes. **Hex literals in component SCSS are not allowed** - using one means the component will not respond to theme changes, which is a bug.

Canonical roles:

- **Text:** `--text-primary` (default), `--text-secondary` (subtitles), `--text-muted` (hints), `--text-dim`, `--text-placeholder`
- **Surface:** `--bg-body`, `--bg-surface` (cards, modal), `--bg-panel` (side panels), `--bg-subtle` (input bg, recessed sections), `--bg-hover`
- **Borders:** `--border` (default), `--border-secondary` (emphasis), `--border-subtle` (table row dividers)
- **Accent:** `--accent` (primary), `--accent-hover`, `--accent-light`, `--accent-highlight-bg`, `--accent-highlight-border`
- **Status:** `--color-good` (success/green), `--color-bad` (error/red), `--text-warning`, `--badge-embedding`
- **Status surfaces:** `--good-bg`, `--bad-bg`, `--warning-bg` / `--warning-border` (amber chips: license notices, type-mismatch rows)
- **Status rows (red/yellow/green sets):** `--status-{color}-{border|dot|label|sub}`
- **Text on a filled surface:** `--btn-primary-text` (on `--accent` buttons), `--btn-filled-text`, `--toggle-active-text` (on an active segmented-toggle button), `--badge-text-dark` (on a saturated status badge). These flip per theme - never hardcode `#fff` on a fill.

If you need a color that does not exist, add it to all three theme blocks in `_variables.scss` - don't introduce a hex literal "just this once."

### 1.6 Transitions - `--transition-*`

| Token               | Value  | When to use |
|---------------------|--------|-------------|
| `--transition-fast` | 0.1s   | Hover backgrounds where the cursor moves quickly |
| `--transition-base` | 0.15s  | **Default** for color/border/background changes |
| `--transition-slow` | 0.3s   | Progress fills, width animations, opacity reveals |

`0.2s`, `0.4s`, and other ad-hoc durations should be rounded to one of these. `prefers-reduced-motion` overrides all three globally (see `styles.scss`), as does the app's own "Show Animations → Hide" setting (`html.animations-off`).

**Easings.** Two shared curves so new motion doesn't hand-roll a `cubic-bezier` per rule:

| Token          | Curve                          | When to use |
|----------------|--------------------------------|-------------|
| `--ease-out`   | `cubic-bezier(0.16, 1, 0.3, 1)` | Things arriving / settling (panel reveal, drawer enter) - fast start, gentle deceleration |
| `--ease-in-out`| `cubic-bezier(0.65, 0, 0.35, 1)`| Symmetric moves where both ends should feel eased (drawer leave) |

### 1.7 Shadows - `--shadow-*`

| Token         | When to use |
|---------------|-------------|
| `--shadow-sm` | Subtle lift (tooltip, popover) |
| `--shadow-md` | Toasts, floating buttons |
| `--shadow-lg` | Modals |

All three resolve through `--shadow-dropdown` so they auto-tint per theme.

### 1.8 Z-index - `--z-*`

| Token                | Value  | Used by |
|----------------------|--------|---------|
| `--z-base`           | 1      | Normal in-flow content |
| `--z-elevated`       | 10     | Sticky table headers, raised tiles |
| `--z-sticky`         | 100    | App header, sticky toolbars |
| `--z-modal-backdrop` | 1000   | Modal overlay |
| `--z-modal`          | 1001   | Modal content |
| `--z-tooltip`        | 1500   | Hover previews / tooltips that must clear a modal |
| `--z-burger-menu`    | 2000   | Dropdown menus that must clear modals |
| `--z-offline-banner` | 4500   | Offline / connection-lost banner |
| `--z-toast`          | 5000   | Toast notifications |
| `--z-login`          | 9999   | Login gate |

Never use a raw z-index. If you need a new layer, add a token.

**Canvas-overlay scale (`--z-canvas-*`).** The affordances painted over the VTSBrowse UMAP canvas live inside the `.browse-content` / `.browse-main` stacking contexts, so they need their own low integers rather than a slot on the global scale: `--z-canvas-base` (1, lift a hovered/active control just past its siblings), `--z-canvas-overlay` (2, the tool clusters floating over the canvas), `--z-canvas-top` (3, the pre-reveal cover). Use these inside that view rather than raw `1`/`2`/`3`.

### 1.9 Disabled state - `--opacity-disabled`

| Token                | Value | When to use |
|----------------------|-------|-------------|
| `--opacity-disabled` | 0.5   | The `opacity` value on every `:disabled` / `.disabled` button or interactive element styled via opacity. |

There is only one disabled opacity. Don't write `opacity: 0.35` or `opacity: 0.4` because it "looks right" - a rendered style audit found seven different ad-hoc values in the wild, several of which dropped text below readable contrast.

**Don't use opacity to disable text on a saturated background.** Opacity dims the rendered pixels toward the background, so 50% white on the accent-blue header collapses to a mid-blue that fails contrast. For buttons sitting on `--header-bg` (or any saturated surface), shift `color` to a theme-aware dimmed token (e.g. `--header-text-dim`) and keep the cursor change - skip `opacity` entirely.

### 1.10 Decorative-dim state - `--opacity-dim`

| Token          | Value | When to use |
|----------------|-------|-------------|
| `--opacity-dim` | 0.7   | A *decorative or secondary* element intentionally shown muted at rest: a dimmed type/status icon, a dropdown chevron, a faint accent divider line, a hint/note paragraph, a locked achievement row. |

Distinct from `--opacity-disabled`: that one is reserved for interactive `:disabled` / `.disabled` states, this one is about visual hierarchy. Like the disabled value, there is only one decorative-dim value - don't hand-pick 0.55/0.6/0.75 for the same "recede this decoration" job. **Not** for animation keyframes (0↔1 fades), hover-brighten rest states, or the two-tier done/future progression dims; those carry their own values by design.

### 1.11 Letter-spacing - `--tracking-wide`

| Token             | Value  | When to use |
|-------------------|--------|-------------|
| `--tracking-wide` | 0.05em | The uppercase micro-labels that pepper the UI: table headers, section micro-headings, tier chips. |

One canonical "wide" value, em-based so it tracks the label's font size. A rendered audit found a dozen hand-picked values (0.02-0.06em, plus a raw `0.5px`) all doing this same job; they resolve here now. Uppercase text without tracking is the only other legal option - don't invent a third value.

### 1.12 Modal widths - `--modal-w-*`

Documented with the modal pattern in §2.4.

---

## 2. Shared component classes

These live in the global stylesheets - `_components.scss` (general), `_picker-shared.scss` (picker/tab domain), `_data-table.scss` (tables), `_layout.scss` (the 3-panel grid) - and are unencapsulated, so a component template pulls them in just by writing `class="..."`. Each subsection below names the file its classes come from. **Do not redeclare these styles in component SCSS** - extend with a child selector if needed (e.g. `.progress-row .btn--cancel { ... }` to scope a specific layout).

### 2.1 Typography

Use the right tag for the role; the global rules will style it correctly.

```html
<h1>Page header</h1>          <!-- --font-3xl semibold -->
<h2>Modal / section title</h2><!-- --font-2xl semibold -->
<h3>Sub-section</h3>          <!-- --font-xl  semibold -->
<h4>Minor group label</h4>    <!-- --font-lg  medium -->
```

Headings have `margin: 0`. Add spacing via the parent's flex `gap` or `margin-bottom` on the heading using a `--space-*` token. **Do not set `font-size`/`font-weight` to make a smaller tag look like a bigger one.**

Two heading helpers:

- **`.section-title`** - asymmetric margins (`--space-2xl` top, `--space-sm` bottom, first child resets its top) so a title adheres to the content *below* it. Use it on the headings of a panel that stacks several sections; see §3.0.
- **`.subhead`** - styles a heading tag (typically `<h4>`) as a small muted sub-section label inside a larger panel. Carries identity only (font-size, color); vertical rhythm stays scoped to the component that uses it.

Text utility classes (use directly on `<p>` / `<span>`):

| Class           | What it is |
|-----------------|------------|
| `.info-text`    | Body helper text - muted color (`--text-muted`), `--font-md` |
| `.error-text`   | Red error message |
| `.success-text` | Green success message |
| `.status-text`  | Inline status / secondary message (`--text-secondary`) |
| `.empty-state`  | Centered "no items" placeholder, padded (see below) |

All four `<p>`-based utilities explicitly zero the UA `<p>` margin so the surrounding layout's flex `gap` owns inter-row spacing (§5, anti-pattern 15).

**`.empty-state`** is the canonical placeholder shown wherever a list, table, or panel has no content yet. The common case is a bare `<p class="empty-state">No datasets yet.</p>`; the richer form stacks an optional icon, the message, and an optional action:

```html
<div class="empty-state">
  <span class="empty-state__icon"><vt-icon type="…" /></span>
  <p class="empty-state__message">Nothing here yet.</p>
  <button class="empty-state__action btn btn--secondary">Add One</button>
</div>
```

Modifiers: `.empty-state--fill` stretches to fill a flex-column parent and centers vertically (the dashboard list panels); `.empty-state--inline` (in `_picker-shared.scss`) collapses it back to a compact, left-aligned, italic inline notice under a tab bar.

### 2.2 Buttons

The full taxonomy is `.btn` plus optional variant + optional size. Always start with `.btn`:

```html
<button class="btn">Default outline</button>
<button class="btn btn--primary">Commit / Save</button>
<button class="btn btn--secondary">Secondary action</button>
<button class="btn btn--danger">Delete</button>
<button class="btn btn--cancel">Cancel</button>
<button class="btn btn--toolbar">Import Labels</button>
<button class="btn btn--sm">Compact</button>
<button class="btn btn--primary btn--sm">Compact primary</button>
<button class="btn btn--xs">Inline mini</button>
<button class="btn btn--icon-square">+</button>
```

Rules:

- **Default size** for prominent actions (modal footers, top-of-panel actions).
- **`.btn--sm`** for inline actions inside cards and rows.
- **`.btn--xs`** for the densest tables / inline edit rows.
- **`.btn--cancel`** for low-weight cancel/close affordances next to progress bars.
- **`.btn--toolbar`** for compact outline buttons in toolbars and panel action rows (the right-panel Import Labels / Add Corrections / Export cluster, the image-view-controls strip): tighter padding, `--font-xs`, a muted resting color, `white-space: nowrap`, and a hover that promotes the border to `--accent`.
- **`.btn--icon-square`** for square plus/icon buttons (28×28).
- **Do not write per-component `padding` / `font-size` / `border-radius` on a button.** Every "small button" implementation in component SCSS that bypassed this taxonomy has been removed.

A borderless icon button inside a card row or action cluster is **not** a `.btn` variant - use `.card-icon-btn` (§2.11).

### 2.3 Forms

```html
<div class="form-group">
  <label class="form-label">Dataset name <span class="required">*</span></label>
  <input class="form-input" />
  <p class="form-hint">JSON, CSV, or NPZ accepted.</p>
</div>
<select class="form-select">...</select>
<textarea class="form-textarea"></textarea>
```

- `.form-group` is the label+control wrapper: a flex column with `gap: var(--space-xs)`. It owns the spacing *inside* one field; the surrounding form (`.importer-form`, §2.5) owns the spacing between fields. `.form-group--section` adds `margin-top: var(--space-xl)` above a grouped block inside an importer form (the Embedder/Clipper "Advanced" block).
- `.form-input` and `.form-select` share padding, border, focus state. They sit on `--bg-subtle` so they read as "input wells."
- `.form-select--compact` is the toolbar-sized select: sized to its content rather than full width, tighter padding, `--font-xs`, on `--bg-surface`. Use it in dense bars (label sort, browse selection panel), not in forms.
- `.form-textarea` extends `.form-input` for multi-line entry: vertical resize only, a `5rem` floor, and `--font-mono` so long term lists line up.
- `.form-label` is `--font-md`, `--weight-medium`, `--text-primary` - sized to match `.form-input` so the header is never visually smaller than the value the user types/picks underneath it. Custom `<button>`-based dropdown triggers that play the role of `.form-select` (e.g. icon-bearing media-type pickers) must set `font-size: var(--font-md)` explicitly: `<button>` doesn't inherit page font by default, and component-scoped overrides (`font: inherit`, etc.) silently win over the global `.form-select` because Angular view encapsulation raises their specificity. If the trigger text ever renders larger than the label above it, that rule is the regression.
- `.form-hint` (used for plugin-field `hint` strings) is muted `--font-mono` at `--font-sm`, and preserves newlines (`white-space: pre-wrap`) so a multi-line schema hint keeps its shape.
- `.required` marks required fields with `--color-bad`.

Focus state is provided globally by `:focus-visible { outline: 2px solid var(--accent); }`. Inputs additionally swap their border to `--accent` on focus. **Do not override focus styling per component.**

### 2.4 Modals - always `<vt-modal>`

**Every dialog is a `<vt-modal>`** (`frontend/src/app/components/modal/`). It owns the backdrop/header/body/footer markup; the only file in the app that writes `class="modal-backdrop"` by hand is the modal component's own template. Do not copy that markup into a new component - see the a11y note below.

```html
<vt-modal title="Import Labels" [open]="true" (closed)="close()">
  <!-- default slot → .modal-body -->
  <p class="info-text">Pick a file to import labels from.</p>

  <div modal-footer>
    <button class="btn btn--secondary" (click)="close()">Cancel</button>
    <button class="btn btn--primary" (click)="submit()">Import</button>
  </div>
</vt-modal>
```

| API | What it does |
|-----|--------------|
| `title` (input) | Renders as the header `<h2>` **and** the dialog's `aria-label`. Title Case (§4.1). |
| `open` (input) | Whether the dialog is mounted. Most call sites are already inside an `@if` and pass `[open]="true"`. |
| `showCloseButton` (input, default `true`) | Renders the header `×`. See "Close-less dialogs" below. |
| `closed` (output) | Fired by the `×`, a backdrop click, and Escape. Wire it to whatever tears the dialog down. |
| default `<ng-content>` | Projected into `.modal-body` (the only scrollable region). |
| `[modal-footer]` slot | Projected into `.modal-footer`. Put the action buttons in a plain `<div modal-footer>`. |

**Why the component and not the markup.** `vt-modal` supplies `cdkTrapFocus` with auto-capture (focus moves into the dialog on open, Tab cycles inside it, focus returns to the trigger on close), `role="dialog"`, `aria-modal="true"`, the `aria-label`, backdrop-click dismissal, and Escape handling that closes **only the topmost** modal (stacked flows like New Detector → media-crop would otherwise all collapse on one keypress). Hand-copied markup silently drops every one of those - it is an accessibility regression, not a styling shortcut.

Spacing inside modals (the shared classes already do this - do not redo it in your content):
- `.modal-content` already has `padding: var(--space-2xl)` - **do not** wrap your projected content in a padding div.
- `.modal-header` and `.modal-body` already have `margin-bottom: var(--space-xl)` - **do not** add it again.
- `.modal-footer` is a right-aligned flex row with `gap: var(--space-md)`, so buttons projected into `[modal-footer]` need no wrapper layout of their own.
- `.modal-content` uses `--shadow-lg` and `--radius-xl`, and is a flex column so the body is the only scrolling region - the header and footer stay pinned when content overflows.

**Width scale.** A dialog that needs a fixed width picks one of the three
tokens instead of hand-rolling a `px` value; a dialog that sizes to its content
sets none:

| Token          | Width | Use |
|----------------|-------|-----|
| `--modal-w-sm` | 480px | Single-column forms: label importer, combine detectors, the blank new-detector form. |
| `--modal-w-md` | 720px | Picker / table / settings dialogs: dataset importer, export, settings, keyboard help. |
| `--modal-w-lg` | 900px | Wide views that need room for a table + browser side by side: the new-detector / dataset media-picker view. |

Set the width on `.modal-content` (via `::ng-deep`, since the element belongs to
`vt-modal`), not on an inner wrapper, so inner content can `width: 100%` and fill
the dialog rather than leaving side gutters. A raw `px`/`rem` width on a
`.modal-content` is a violation.

```scss
:host ::ng-deep .modal-content {
  width: var(--modal-w-md);
}
```

**Footer buttons.** Cancel/back on the left, the primary action on the far
right. **"Close" is never `.btn--primary`.** A Close button dismisses the
dialog - it commits nothing - so styling it as the primary (accent-fill) action
misreads as "save". Dismiss buttons are `.btn--secondary`. When a dialog
auto-saves (e.g. Settings) the dismiss verb is **"Done"**, not "Close", so it
doesn't imply the changes were pending a commit.

**Close-less dialogs.** A handful of modals set `[showCloseButton]="false"` and
render no header `×`, on purpose: they are decision points that must be resolved
by an explicit footer action (or a `← Back`) rather than dismissed ambiguously.
These are the new-detector modal, the clipper-chooser, combine-detectors,
resort-prompt, and dialog-host. Every *other* modal keeps the header `×`. Do not
add `[showCloseButton]="false"` to a new modal without a comparable reason.

#### Back vs Cancel

This pattern is **mandatory** for any modal with an outer→inner view (importer picker → importer form, exporter picker → exporter form, new-detector → media picker, etc.). See CLAUDE.md for the canonical writeup. Short version:

- **`← Back`** (top-left of inner view, `.back-btn`) - navigates to the previous view without committing the step. Use it any time the user is "going back" to a parent view, including from a child modal.
- **`Cancel`** (footer, `.btn`) - abandons the whole dialog. Use it only at the leaves of a flow.

The canonical back-button markup:

```html
<button class="btn btn--secondary btn--sm back-btn" (click)="back()" title="Return to ...">
  &larr; Back
</button>
```

### 2.5 Picker cards (`.picker-card`)

`.picker-card` (`_components.scss`) is the one selectable option card - icon + title + description - for every picker that lists choices: the settings import/export pickers, label import/export, the New Detector "Trained" tab, and the load-sort / resort-prompt source pickers. It carries the shared padding, gap, border, radius, and accent hover, plus a `.selected` state for pickers that keep a persistent choice.

```html
<button class="picker-card" (click)="select(src)">
  <span class="picker-card__title">
    <span class="picker-card__icon"><vt-icon [icon]="src.icon" /></span>{{ src.display_name }}
  </span>
  <span class="picker-card__desc">{{ src.description }}</span>
</button>
```

| Class                 | Role |
|-----------------------|------|
| `.picker-card`        | The card itself (a `<button>`). Add `.selected` for a persistent choice. |
| `.picker-card__icon`  | Inline icon wrapper inside the title. |
| `.picker-card__title` | `--font-lg`, `--weight-semibold`. The one canonical title size - don't restyle it. |
| `.picker-card__desc`  | `--font-sm`, `--text-secondary`. |

**Only the card is shared; the container isn't (yet).** `.importer-picker` / `.exporter-picker` name the list wrapper, but the shared rule's responsive grid is currently overridden by every consumer, each of which declares its own `display: flex; flex-direction: column` locally (component-scoped rules win via Angular's encapsulation specificity bump). So set the container layout you actually want in your component SCSS rather than assuming the global one applies. Reconciling the two is tracked in `docs/plans/codebase-audit-2026-08.md`.

The form half of a picker modal uses `.importer-form` (flex column, `gap: var(--space-lg)`) wrapping `.form-group` fields (§2.3), with `.back-btn` at the top of the inner view (§2.4 → Back vs Cancel).

There are no `.importer-card` / `.importer-name` / `.importer-desc` classes; the names above replaced them.

### 2.6 Tabs

Three shared tab primitives, each with a different shape. Pick by orientation and by whether the tab flows into a panel.

**Horizontal strip - `.tab-bar` / `.tab`** (`_picker-shared.scss`). The default: a full-width bar with a bottom border, tabs that underline in `--accent` on activation (the underline wipes in via a `scaleX` pseudo-element) and go `--weight-semibold`. **The active class is the bare `.active`**, not a BEM modifier.

```html
<div class="tab-bar" role="tablist">
  <button class="tab" [class.active]="active() === 'audio'">Audio</button>
</div>
```

Subclasses extend the base via SCSS `@extend`, so they inherit every rule and add only their delta:

| Class | Extends | Delta |
|-------|---------|-------|
| `.importer-subtab-bar` / `.importer-subtab` | `.tab-bar` / `.tab` | Adds a `.disabled` / `:disabled` treatment. `.importer-subtab-icon` wraps its inline icon. |
| `.demo-tab-bar` / `.demo-tab` | `.tab-bar` / `.tab` | None - a semantic alias for the demo media-type strip. |

`.tab-bar-hint` is the italic muted line shown below a bar before the user has picked a tab ("Select what type of dataset to add."); `.empty-state--inline` is the matching "this category is empty" notice.

**Vertical rail - `.side-tab-bar` / `.side-tab`** (`_components.scss`). A fixed-width (140px) column of tabs down the left edge of a modal, with a 3px left-border active indicator instead of an underline. The Settings modal's rail is its only consumer today; it's the extract-and-fold target for any other vertical rail. Active class: `.side-tab--active`.

**Paneled tabs - `.view-tabs` / `.view-tab` / `.view-tab-content`** (`_components.scss`). A tab strip whose active tab visually flows into the inset content region below it: the active tab's background matches `.view-tab-content` so the border between them disappears, leaving only the accent underline. Use when the tabs sit *inside* a panel and own a bordered content box (the Auto-Find and Import-Defaults settings sections). Active class: `.view-tab--active`. `.view-tab-content` deliberately omits `display` - pick `flex` or `grid` for your inner layout.

Note the state-class inconsistency: `.tab` takes `.active`, while `.side-tab` and `.view-tab` take `--active` modifiers. Match the primitive you're using.

### 2.7 Tables

Every table opts onto the base **`.data-table`** core (`_data-table.scss`): full width, collapsed borders, cell padding, cell bottom-borders. The per-table knobs are CSS custom properties, so a variant retunes them without redeclaring the core:

| Knob | Default | What it sets |
|------|---------|--------------|
| `--dt-cell-pad-y` | `var(--space-sm)` | Vertical cell padding |
| `--dt-cell-pad-x` | `var(--space-md)` | Horizontal cell padding |
| `--dt-border-color` | `var(--border-subtle)` | Row divider color |

**`.data-table--grid`** is the interactive modifier: sortable / draggable / resizable columns, a sticky uppercase `--font-xs` header (tracked with `--tracking-wide`), `--font-md` body cells, ellipsis truncation, and row hover. Add it to any grid the user can sort or reorder.

Variants layer only their own deltas on top:

- **`.dash-table`** - dashboard grids: pinned select/actions columns and a selected-row state.
- **`.demo-table`** (`_picker-shared.scss`) - the importer demo picker: wider body cells and a disabled-row treatment.
- **`.stats-table`** - read-only key/value tables in the stats modals: semibold secondary header, no grid behavior.
- **`.table-fixed`** - a `table-layout: fixed` helper, combined with any of the above when columns are width-driven.

Compose them in the template, base first:

```html
<table class="data-table data-table--grid dash-table" [class.table-fixed]="cols.tableFixed">
```

**Do not invent a new table class** - start from `.data-table` (plus `--grid` if it's interactive) and add a row state or a padding knob.

### 2.8 Badges

Use the shared `.badge-ready` / `.badge-embedding` / `.badge-download` classes (`_picker-shared.scss`) for status badges. They share padding, font-size, and radius. If you need a new badge variant, add it next to those rules and reuse the same dimensions.

### 2.9 Layout

The 3-panel grid lives in `_layout.scss`: `.layout` is a three-column grid (`minmax(280px, 20%) 1fr minmax(280px, 20%)`, so the side panels track the viewport instead of pinning at a fixed width), with `.panel-left` / `.panel-center` / `.panel-right` carrying `padding: var(--space-xl)`, `--bg-panel`, and theme-aware dividing borders. Inside a panel, prefer flex with a `--space-md` or `--space-lg` gap.

### 2.10 Segmented toggle

A connected row of buttons acting as one single-select control (the View/Focus pickers in Settings, the size/focus toggles in the center-panel view toolbar). Buttons share a border via a negative left margin, the group's outer corners are rounded, and the selected button fills with `--accent` plus an inset shadow.

```html
<div class="segmented-toggle segmented-toggle--stretch">
  <button class="segmented-toggle__btn" [class.segmented-toggle__btn--active]="mode() === 'grid'">Grid</button>
  <button class="segmented-toggle__btn" [class.segmented-toggle__btn--active]="mode() === 'list'">List</button>
</div>
```

Variants: `.segmented-toggle--stretch` makes the buttons equal-width and fill the row (labelled toggles); `.segmented-toggle--compact` makes them dense and icon-sized with a border-highlight hover instead of a fill (toolbars). Use this rather than a row of `.btn`s whenever the buttons are mutually exclusive states of one setting.

### 2.11 Entity cards and card actions

The dashboard's dataset/detector rows (`vt-dataset-card`, `vt-detector-card`) render as `<tr>`s inside a `.dash-table` and share three primitives:

| Class | What it is |
|-------|------------|
| `.entity-card` | The row itself: pointer cursor, hover tint, `.selected` (accent-highlight fill) and `.dimmed` (faded, pointer-events off) states. Applied via the component's `host` metadata, not the template. |
| `.card-actions` | The right-aligned inline action cluster in the row's Actions cell. |
| `.card-icon-btn` | The compact borderless icon button inside that cluster (edit, load, delete, overflow) - also used by the right-panel goods-action cluster. `.card-icon-btn--danger` reddens on hover for destructive actions. |

### 2.12 Pane divider

`.pane-divider` is the draggable rule between resizable layout panes (Find, Label, and VTSBrowse views): an 8px hit target with a 4px visible line centered inside - the same wider-hit-zone pattern as `.col-resize-handle`, because the thin line alone is hard to grab. The line tracks `--border` at rest and switches to `--accent` on hover; the drag handlers add `.dragging` to keep the accent lit through the drag. Don't hand-roll a divider; use this and bind `.dragging`.

### 2.13 Motion and misc utilities

Global classes that any component can apply. All animations here are silenced by the reduce-motion rules in `styles.scss` (§1.6).

| Class | What it is |
|-------|------------|
| `.drawer-enter-left` / `.drawer-leave-left` / `.drawer-enter-right` / `.drawer-leave-right` | Slide-from-edge transitions for side panels, bound via Angular's `animate.enter` / `animate.leave`. |
| `.swipe-left` / `.swipe-right` | The vote fling that throws the current media off-screen (bad / good). |
| `.icon-waggle` | 2s rotate-and-hold loop signalling "a slow job kicked off by this control is running" (Find/Train buttons, in-flight import/export submits). |
| `.skeleton-bg` | Flat subtle placeholder painted behind a lazy-loaded `<img>` so it doesn't pop in against nothing. |
| `.waveform-mask` | Paints `--accent` through an audio waveform's alpha mask so thumbnails tint per theme. The `mask-image` URL is set per instance via a style binding. |
| `.sr-only` | Visually hidden, screen-reader-visible text. |

Keyframes are global even under Angular's emulated encapsulation, so these live in `_components.scss` on purpose - two component-local copies of a `@keyframes` name are a collision waiting to drift.

---

## 3. Patterns and conventions

### 3.0 Vertical rhythm - titles belong to the content BELOW them

When a panel or modal stacks multiple labelled sections, the gap between a section's title and its content must be **smaller** than the gap between that content and the next section's title. Equal-spaced gaps make titles look like they belong to the section above:

```
WRONG                            RIGHT
Section A                        Section A
[24px gap - equal]               [8px gap - small]
content A content A              content A content A
[24px gap - equal]               [24px gap - LARGER]
Section B                        Section B
[24px gap - equal]               [8px gap - small]
content B content B              content B content B
```

How to achieve the right rhythm:

- **Asymmetric heading margins.** Inside a section container, give the section title (h3/h4 or a `.section-title` element) `margin-top: var(--space-2xl)` and `margin-bottom: var(--space-sm)`, with `:first-child { margin-top: 0 }`. The shared `.section-title` utility class in `_components.scss` does exactly this - use it when you have a panel that stacks sections (no extra flex gap needed).
- **Or use container padding + flex gap.** If each section lives in its own bordered panel (like the dashboard sections or `.vote-section` in the label list), the panel padding plus container `gap` already produces the right rhythm; no heading margin needed.
- **Do not** use a flex container's `gap` to do all the spacing and leave headings with `margin: 0`. That produces equal gaps everywhere and breaks the hierarchy.

If you're authoring a settings-style modal with several h3 sub-sections inside one panel, the canonical pattern is:

```scss
.panel {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);          // in-section rhythm

  h3 {
    margin: var(--space-xl) 0 0; // larger top, no bottom (flex gap handles below)
    &:first-child { margin-top: 0; }
  }
}
```

### 3.1 Flex gaps

Pick the gap from the spacing scale that matches the visual density:

- **Tight inline rows** (icon + label, related controls): `gap: var(--space-xs)` or `var(--space-sm)`
- **Standard toolbars / button groups**: `gap: var(--space-md)`
- **Card-to-card / section-to-section**: `gap: var(--space-lg)` or `var(--space-xl)`

Within one feature, **use the same gap for the same visual pattern**. If a header has `gap: var(--space-md)` between title and actions, every similar header should use the same value.

### 3.2 Alignment

`align-items: center` is the default in horizontal flex rows. Justify-content choices:

- `justify-content: flex-end` - button rows, modal footers, action columns.
- `justify-content: space-between` - title + close/action header pairs.
- `justify-content: center` - empty states, single-item centered layouts.

### 3.3 Focus and disabled states

Don't write per-component focus / disabled CSS. The base classes handle both:

- `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }` (global)
- `.btn:disabled { opacity: var(--opacity-disabled); cursor: not-allowed; }` (`.btn--primary` swaps to a neutral fill + dimmed text instead, since opacity can't hold contrast over the accent)
- `.form-input:focus { border-color: var(--accent); }`

### 3.4 Theme overrides inside a component

Avoid them. The right answer is almost always "use a theme variable." If a component genuinely needs a per-theme tweak (rare), wrap the override in `[data-theme="light"]` or `[data-theme="highviz"]` and add a comment explaining why a token-based solution didn't work.

---

## 4. Copy style (voice, casing, vocabulary)

Microcopy is styling too. Visible UI text - button labels, headings, form
labels, placeholders, hints, empty states, tooltips - follows the same
"one rule, applied everywhere" discipline as the tokens. The rules below are
not machine-enforced (casing can't be reliably linted), so they live here and
are applied by hand during review.

### 4.1 Casing: Title Case for chrome, sentence case for content

Two buckets, one rule each:

- **Title Case** - the framing "chrome" of the UI: **modal titles** (`vt-modal
  title="…"`), **section headings** (`<h1>`–`<h4>`, `.section-title`,
  `.dashboard-section-title`), and **button labels** (`.btn`, toolbar/icon
  buttons with a text label, radio-pill labels). These name a surface or an
  action, so they read as a title.
- **sentence case** - everything the user *reads or fills in*: **form field
  labels** (`.form-label`, `.col-label`), **placeholders**, **hints**
  (`.form-hint`, `.info-text`), **empty-state messages** (`.empty-state`),
  **tooltips** (`title="…"`), **aria-labels**, and any inline description or
  status sentence. Capitalize only the first word and proper nouns.

**Title Case rule.** Capitalize the first and last word and every "major"
word. Keep these lowercase unless they're first or last: articles (`a`, `an`,
`the`), coordinating conjunctions (`and`, `but`, `or`, `nor`, `for`, `so`,
`yet`), and short prepositions (`to`, `of`, `in`, `on`, `at`, `by`, `for`,
`with`, `from`, …). So it's `Add Media to Bad`, `Sort by Detector`, `Copy to
Clipboard` (not `Copy To Clipboard`), `Add Corrections to Detector`.

| Surface | Case | Example |
|---------|------|---------|
| Modal title | Title Case | `Combine Datasets`, `Crop Example`, `Use This Example?` |
| Section heading (`h2`–`h4`, `.section-title`) | Title Case | `Detector Accuracy`, `What You'll Get` |
| Button label | Title Case | `Import Labels`, `Rebuild Map`, `Download Bundle` |
| Form field label | sentence case | `Detector name`, `Conflict policy`, `Cell size` |
| Placeholder | sentence case | `Combined dataset name`, `Describe what you're looking for` |
| Hint / info / empty-state | sentence case | `No detectors yet. Click + to add one.` |
| Tooltip (`title=`) / aria-label | sentence case | `Order labeled items by time, name, or detector confidence` |

Proper nouns keep their own casing everywhere (`HuggingFace`, `VTSearch`,
`UMAP`), as do the canonical capitalized buckets **Good** / **Bad** when they
name the two vote piles (`Add Media to Good`, `Mark this media as a Bad
example`).

### 4.2 Ellipsis: the `…` character only

Use the single Unicode ellipsis character **`…` (U+2026)**. Never the three-dot
ASCII `...`, and never the HTML entity `&hellip;`. This holds for both templates
and any user-facing string literal in a `.ts` file (`signal('Loading…')`,
status messages, fallback labels).

- **In-progress button/status labels** end in `…`: `Creating…`, `Importing…`,
  `Combining…`, `Saving…`, `Scoring with example media…`.
- **Truncation** markers use `…` too (`… 12 more rows`, `text.slice(0, 300) +
  '…'`).
- **Placeholders do *not* end in `…`** (see §4.3).

### 4.3 Placeholder format

- **Sentence case**, no trailing ellipsis, no trailing period.
- For "here's what to type" examples, prefix with `e.g. ` and match the casing
  of the real value: a free-text query is lowercase (`e.g. dog barking
  sounds`), a proper name is Title Case (`e.g. Dog Barks`).
- For "leave blank to get a default" inputs, state that: `Leave blank to use a
  default name`.
- For a server/file path, use the shared hint `path/to/file` (or
  `/absolute/server/path/to/file` when an absolute path is required) - don't
  invent a new spelling.

### 4.4 Canonical product vocabulary

One concept, one word. The product noun for the trained ranker is
**detector** - **never** "model" in user-facing text. ("Model" is correct only
when it genuinely means a HuggingFace **AI model** / embedder, e.g. the gated-model
download copy in Settings.) Internal identifiers (`ngModel`, `trainMode.model`,
CSS classes) are exempt - this rule is about *visible* strings only.

| Concept | Canonical word | Don't write |
|---------|----------------|-------------|
| The trained ranker (the product's core object) | **detector** | model |
| Making a detector by voting good/bad | **Train** (verb) / **Learned** (the sort mode) | — |
| Running a detector across a dataset to score items | **Find** (the action) / **Auto-Find** (the automatic/CLI variant) | — |
| The two vote piles | **Good** / **Bad** | positives/negatives (in general UI; the ML terms are fine inside a stats table) |

`Train` and `Find` are the two flow verbs surfaced to users; keep them stable.
Meaning-bearing distinctions are *not* drift and stay as-is: `Verified Good`
vs. `Good` (a real Find-mode state), and `Positives`/`Negatives` inside the
detector-stats table (standard ML terminology in that context).

---

## 5. Anti-patterns (do not do these)

1. **Hardcoded `px` / `rem` for padding, margin, gap.** Pick a `--space-*` token.
2. **Hardcoded font sizes.** Pick a `--font-*` token. `0.78rem`, `0.95rem`, `13px` are all violations.
3. **Hex colors in component SCSS.** Pick a theme variable; add one to all three theme blocks if it doesn't exist yet.
4. **Per-component button styles.** Use `.btn` + variant + size modifier. Never restyle a `.btn`'s padding/font-size/radius from a component.
5. **Custom focus / disabled / hover styling that overrides the base.** Extend the base, don't replace it.
6. **`font-weight: 700` / `bold`.** Use `--weight-semibold` (600) or rethink the layout.
7. **Restyling an `<h1>`/`<h2>` to look like a different heading.** Use the right tag.
8. **Wrapping a modal's body in a div with extra padding.** `.modal-content` already pads.
9. **Inventing a new table class.** Start from `.data-table` (+ `.data-table--grid` when interactive) and layer a delta, as `.dash-table` / `.demo-table` / `.stats-table` do.
10. **`transition: all <duration>` with a custom duration.** Use `var(--transition-base)`.
11. **Cancel as a Back button (or vice versa).** See §2.4.
12. **Designing for mobile.** Desktop only.
13. **`font: inherit` on a class that combines with `.form-input` / `.form-select`** (or any shared element class whose font-size is set globally). Angular's view-encapsulated component selectors get an attribute-selector specificity bump that beats the global `.form-input` rule, so a component-scoped `font: inherit` silently drops `var(--font-md)` and renders the page-root `1rem` instead - which is why a custom dropdown trigger can render its content larger than the `.form-label` above it. If you need the button to inherit something from the parent, be explicit: `font-family: inherit; font-size: var(--font-md);`. The same trap applies to any shorthand that sets `font-size` (raw `font: 14px ...`, `font: bold 1rem`, etc.) under a component-scoped selector.
14. **`flex-direction: column` without an explicit `gap`** (and no per-child margins). A stacked-column container has to own its inter-row spacing - either set `gap: var(--space-*)` on the parent, or commit to a child class (`.form-group`, `.section-title`) that carries its own margins. Mixing the two ad-hoc produces uneven rhythms like "no space between drop zone and the input below it, but huge space between the section header and its description." Pick one mechanism per container.
15. **Redeclaring shared utility classes locally.** `.info-text`, `.error-text`, `.success-text`, `.status-text`, `.form-label`, `.form-input`, `.form-select`, `.form-group`, `.btn`, `.modal-*`, `.back-btn` live in `_components.scss` as the single source of truth. Copying their bodies into a component SCSS file - even with the same property values - causes drift the moment someone tunes the global rule. Need a scoped tweak? Extend with a descendant selector (`.my-panel .info-text { ... }`) instead of redeclaring. As a corollary, **shared `<p>`-based utility classes (`.info-text` etc.) must reset `margin: 0`** so the surrounding layout's flex `gap` owns inter-row spacing - UA `<p>` margins inject ~1em above and below and break the §3.0 rhythm.
16. **Hand-writing modal markup instead of using `<vt-modal>`.** Copying `.modal-backdrop` / `.modal-content` / `.modal-header` into a component drops focus trapping, `role="dialog"`, `aria-modal`, the Escape stack, and backdrop dismissal. See §2.4.

> A static scan for items 1-3, 6-7, 10, 13-15 lives at
> `.claude/scripts/style-check.py` (invoked via the `/style-check`
> skill). It also checks the token-level rules that aren't anti-pattern
> items: raw `z-index` (§1.8), raw decorative-dim opacity (§1.10),
> hand-rolled accent tints, unresolvable `var()`s, and deleted token
> aliases. Run it before a styling-heavy PR, or whenever a layout
> regression shows up that you can't explain by reading one file.

---

## 6. Adding new shared styles

When you have a pattern that recurs (the third time you copy similar SCSS from one component to another), promote it to `_components.scss` (general shared) or `_picker-shared.scss` (importer/exporter/picker domain) or `_data-table.scss` (tables). Don't let the same "kind of thing" diverge across components - that's how the codebase ends up with three "small buttons" that all look slightly different.

When you add a new token:
- Add it under the matching scale in `_variables.scss` (all three theme blocks if it's color-bearing).
- Document it here.
- Use it everywhere it applies.

---

## 7. References

- `frontend/src/scss/_variables.scss` - every design token.
- `frontend/src/scss/_components.scss` - buttons, forms, modal chrome, headings, picker cards, segmented toggle, side/view tabs, entity cards, pane divider, info/error/success text, motion utilities.
- `frontend/src/scss/_picker-shared.scss` - the horizontal tab strip and its subclasses, picker table deltas, badges.
- `frontend/src/scss/_data-table.scss` - the `.data-table` core, the interactive grid modifier, and the dashboard/stats variants.
- `frontend/src/scss/_layout.scss` - 3-panel grid.
- `frontend/src/styles.scss` - global resets, reduce-motion rules.
- `frontend/src/app/components/modal/` - the `<vt-modal>` component every dialog uses (§2.4).
- `docs/FRONTEND.md` - SPA architecture: feature areas, service layer, zoneless change detection, component conventions.
- `CLAUDE.md` - Back vs Cancel rules, desktop-only scope.
- `.claude/scripts/style-check.py` - static SCSS audit (invoked via the `/style-check` skill).
