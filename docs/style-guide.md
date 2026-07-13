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
- **Status rows (red/yellow/green sets):** `--status-{color}-{border|dot|label|sub}`

If you need a color that does not exist, add it to all three theme blocks in `_variables.scss` - don't introduce a hex literal "just this once."

### 1.6 Transitions - `--transition-*`

| Token               | Value  | When to use |
|---------------------|--------|-------------|
| `--transition-fast` | 0.1s   | Hover backgrounds where the cursor moves quickly |
| `--transition-base` | 0.15s  | **Default** for color/border/background changes |
| `--transition-slow` | 0.3s   | Progress fills, width animations, opacity reveals |

`0.2s`, `0.4s`, and other ad-hoc durations should be rounded to one of these. `prefers-reduced-motion` overrides all three globally (see `styles.scss`).

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
| `--z-burger-menu`    | 2000   | Dropdown menus that must clear modals |
| `--z-offline-banner` | 4500   | Offline / connection-lost banner |
| `--z-toast`          | 5000   | Toast notifications |
| `--z-login`          | 9999   | Login gate |

Never use a raw z-index. If you need a new layer, add a token.

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

---

## 2. Shared component classes

All these live in `frontend/src/scss/_components.scss` and are global. Apply them via `class="..."` in templates. **Do not redeclare these styles in component SCSS** - extend with a child selector if needed (e.g. `.progress-row .btn--cancel { ... }` to scope a specific layout).

### 2.1 Typography

Use the right tag for the role; the global rules will style it correctly.

```html
<h1>Page header</h1>          <!-- 1.4rem semibold -->
<h2>Modal / section title</h2><!-- 1.1rem semibold -->
<h3>Sub-section</h3>          <!-- 1rem semibold -->
<h4>Minor group label</h4>    <!-- 0.9rem medium -->
```

Headings have `margin: 0`. Add spacing via the parent's flex `gap` or `margin-bottom` on the heading using a `--space-*` token. **Do not set `font-size`/`font-weight` to make a smaller tag look like a bigger one.**

Text utility classes (use directly on `<p>` / `<span>`):

| Class           | What it is |
|-----------------|------------|
| `.info-text`    | Body helper text - secondary color, default font-size |
| `.error-text`   | Red error message |
| `.success-text` | Green success message |
| `.status-text`  | Inline status / secondary message |
| `.empty-state`  | Centered "no items" message, padded |

### 2.2 Buttons

The full taxonomy is `.btn` plus optional variant + optional size. Always start with `.btn`:

```html
<button class="btn">Default outline</button>
<button class="btn btn--primary">Commit / Save</button>
<button class="btn btn--secondary">Secondary action</button>
<button class="btn btn--danger">Delete</button>
<button class="btn btn--cancel">Cancel</button>
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
- **`.btn--icon-square`** for square plus/icon buttons (28×28).
- **Do not write per-component `padding` / `font-size` / `border-radius` on a button.** Every "small button" implementation in component SCSS that bypassed this taxonomy has been removed.

### 2.3 Forms

```html
<label class="form-label">Dataset name</label>
<input class="form-input" />
<select class="form-select">...</select>
<p class="form-hint">JSON, CSV, or NPZ accepted.</p>
```

- `.form-input` and `.form-select` share padding, border, focus state. They sit on `--bg-subtle` so they read as "input wells."
- `.form-label` is `--font-md`, `--weight-medium`, `--text-primary` - sized to match `.form-input` so the header is never visually smaller than the value the user types/picks underneath it. Custom `<button>`-based dropdown triggers that play the role of `.form-select` (e.g. icon-bearing media-type pickers) must set `font-size: var(--font-md)` explicitly: `<button>` doesn't inherit page font by default, and component-scoped overrides (`font: inherit`, etc.) silently win over the global `.form-select` because Angular view encapsulation raises their specificity. If the trigger text ever renders larger than the label above it, that rule is the regression.
- `.form-hint` (used for plugin-field `hint` strings) is muted monospace.
- `.required` marks required fields with `--color-bad`.

Focus state is provided globally by `:focus-visible { outline: 2px solid var(--accent); }`. Inputs additionally swap their border to `--accent` on focus. **Do not override focus styling per component.**

### 2.4 Modals

The modal framework is the standard markup pattern; copy it.

```html
<div class="modal-backdrop">
  <div class="modal-content">
    <header class="modal-header">
      <h2>Title</h2>
      <button class="modal-close">×</button>
    </header>
    <div class="modal-body">
      <!-- content -->
    </div>
    <footer class="modal-footer">
      <button class="btn">Cancel</button>
      <button class="btn btn--primary">Save</button>
    </footer>
  </div>
</div>
```

Spacing inside modals:
- `.modal-content` already has `padding: var(--space-2xl)` - **do not** add a wrapping `padding` div.
- `.modal-header`, `.modal-body` already have `margin-bottom: var(--space-xl)` - **do not** add it again.
- `.modal-footer` has `gap: var(--space-md)` between buttons.
- The modal-content uses `--shadow-lg` and `--radius-xl`.

**Width scale.** A dialog that needs a fixed width picks one of the three
tokens instead of hand-rolling a `px` value; a dialog that sizes to its content
sets none:

| Token          | Width | Use |
|----------------|-------|-----|
| `--modal-w-sm` | 480px | Single-column forms: label importer, combine detectors, the blank new-detector form. |
| `--modal-w-md` | 720px | Picker / table / settings dialogs: dataset importer, export, settings, keyboard help. |
| `--modal-w-lg` | 900px | Wide views that need room for a table + browser side by side: the new-detector / dataset media-picker view. |

Set the width on `.modal-content` (via `::ng-deep`), not on an inner wrapper,
so inner content can `width: 100%` and fill the dialog rather than leaving side
gutters. A raw `px`/`rem` width on a `.modal-content` is a violation.

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

### 2.5 Importer / exporter pickers

The shared classes `.importer-picker` / `.importer-card` / `.importer-name` / `.importer-desc` / `.importer-form` / `.form-group` / `.required` cover the dataset importer modal, label importer modal, settings importer/exporter modals, exporter modal, and new-detector picker. **Don't duplicate the card/tile look** - apply the shared classes and the cards will pick up consistent padding, gap, hover, and radius.

### 2.6 Tabs (`.importer-tab-bar` / `.importer-tab`)

Tab strips inside modals use the shared classes in `_picker-shared.scss`. Subclasses (`.importer-subtab`, `.demo-tab`, etc.) extend the base via SCSS `@extend`. Active state: accent color, accent underline, `--weight-semibold`.

### 2.7 Tables

Two near-identical table classes share most styling:

- **`.dash-table`** (`_data-table.scss`) - dashboard grids, with pinned select/actions columns.
- **`.demo-table`** (`_picker-shared.scss`) - picker tables (demo, labels, etc.).

Both use `--font-md` cells, `--font-xs` uppercase headers, `--space-sm var(--space-md)` cell padding. **Do not invent a new table class** - extend one of these with a row state if needed.

### 2.8 Badges

Use the shared `.badge-ready` / `.badge-embedding` / `.badge-download` classes (`_picker-shared.scss`) for status badges. They share padding, font-size, and radius. If you need a new badge variant, add it next to those rules and reuse the same dimensions.

### 2.9 Layout

The 3-panel grid lives in `_layout.scss`. Panels have `padding: var(--space-xl)` and theme-aware borders. Inside a panel, prefer flex with a `--space-md` or `--space-lg` gap.

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
- `.btn:disabled { opacity: 0.5; cursor: not-allowed; }`
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
9. **Inventing a new table class.** Extend `.dash-table` or `.demo-table`.
10. **`transition: all <duration>` with a custom duration.** Use `var(--transition-base)`.
11. **Cancel as a Back button (or vice versa).** See §2.4.
12. **Designing for mobile.** Desktop only.
13. **`font: inherit` on a class that combines with `.form-input` / `.form-select`** (or any shared element class whose font-size is set globally). Angular's view-encapsulated component selectors get an attribute-selector specificity bump that beats the global `.form-input` rule, so a component-scoped `font: inherit` silently drops `var(--font-md)` and renders the page-root `1rem` instead - which is why a custom dropdown trigger can render its content larger than the `.form-label` above it. If you need the button to inherit something from the parent, be explicit: `font-family: inherit; font-size: var(--font-md);`. The same trap applies to any shorthand that sets `font-size` (raw `font: 14px ...`, `font: bold 1rem`, etc.) under a component-scoped selector.
14. **`flex-direction: column` without an explicit `gap`** (and no per-child margins). A stacked-column container has to own its inter-row spacing - either set `gap: var(--space-*)` on the parent, or commit to a child class (`.form-group`, `.section-title`) that carries its own margins. Mixing the two ad-hoc produces uneven rhythms like "no space between drop zone and the input below it, but huge space between the section header and its description." Pick one mechanism per container.
15. **Redeclaring shared utility classes locally.** `.info-text`, `.error-text`, `.success-text`, `.status-text`, `.form-label`, `.form-input`, `.form-select`, `.form-group`, `.btn`, `.modal-*`, `.back-btn` live in `_components.scss` as the single source of truth. Copying their bodies into a component SCSS file - even with the same property values - causes drift the moment someone tunes the global rule. Need a scoped tweak? Extend with a descendant selector (`.my-panel .info-text { ... }`) instead of redeclaring. As a corollary, **shared `<p>`-based utility classes (`.info-text` etc.) must reset `margin: 0`** so the surrounding layout's flex `gap` owns inter-row spacing - UA `<p>` margins inject ~1em above and below and break the §3.0 rhythm.

> A static scan for items 1-3, 6-7, 10, 13-15 lives at
> `.claude/scripts/style-check.py` (invoked via the `/style-check`
> skill). Run it before a styling-heavy PR, or whenever a layout
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
- `frontend/src/scss/_components.scss` - buttons, forms, modals, headings, info/error/success text.
- `frontend/src/scss/_picker-shared.scss` - importer/exporter tabs, picker table, badges.
- `frontend/src/scss/_data-table.scss` - dashboard table.
- `frontend/src/scss/_layout.scss` - 3-panel grid.
- `CLAUDE.md` - Back vs Cancel rules, desktop-only scope.
- `.claude/scripts/style-check.py` - static SCSS audit (invoked via the `/style-check` skill).
