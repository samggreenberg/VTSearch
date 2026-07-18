# UI Motion Vocabulary

Proposed expansion of VTSearch's animation surface. Today the tool has three
deliberate, non-distracting motions: **VTSBrowse pan/zoom** (hand-rolled
`requestAnimationFrame` tweens in `browse-canvas.component.ts`), the **vote
swipe** (`center-panel` `@keyframes swipe-left/right`), and the **waggling task
icons** (`.icon-waggle` in `_components.scss`, opted into by Find/Train/import
buttons mid-job). This plan proposes what else could join that vocabulary
**without crossing the "not distracting" bar**, and the shared infrastructure
several of the ideas depend on.

None of this is implemented yet — this is a design/sequencing doc, not a ship
log. Promote any item to a GitHub issue once it's concrete enough to ship on its
own (then replace its body here with a one-line `#N` pointer per the plan-file
policy in `CLAUDE.md`).

## Design constraints (read before speccing any item)

- **Motion must explain, not decorate.** The best additions below animate a
  *state change the user needs to understand* (results reshuffled, your vote
  landed somewhere, a panel came from this button). Pure ornament is the
  distraction the user wants to avoid — flag it explicitly when an item risks it.
- **Everything gates on the existing story.** `showAnimations()` (app setting)
  and `prefersReducedMotion()` (`frontend/src/app/utils/reduced-motion.ts`, OS +
  app), plus the `suppress-motion()` mixin in `styles.scss`, already cover both
  the CSS and JS paths. New motion is opt-in by construction — route through
  these, never add an ungated animation.
- **Reuse the duration tokens.** `--transition-fast: 0.1s`, `--transition-base:
  0.15s`, `--transition-slow: 0.3s` (`_variables.scss`). Do not invent new
  durations per component.
- **Desktop only.** No touch/mobile motion considerations (per `CLAUDE.md`).

## Shared infrastructure (enabling work — several items depend on these)

<!-- item-sep -->

- **Reusable flight / FLIP helper** — No measure-rect-then-tween utility exists
  (the browse-canvas zoom tween is bespoke and canvas-only). A small shared
  helper — capture `getBoundingClientRect()` before/after, tween the delta (or
  fly a cloned ghost node between two rects) — unlocks *vote-flies-to-list* and
  *re-sort settle* as a batch. Must respect the reduced-motion gate and clean
  up its ghost nodes. Files: new util under `frontend/src/app/utils/`, consumed
  by center-panel / left-panel / modal.

<!-- item-sep -->

## Media & voting flow (extends the swipe language)

<!-- item-sep -->

- **Voted item flies to the labeled-items list** — After a good/bad swipe, a
  ghost of the card arcs toward the left-panel labeled list, which fades in the
  new row as the ghost "lands." Makes "your vote went *somewhere*" legible —
  the voting analogue of the button-origin idea. Depends on the **flight
  helper**. Files: `center-panel/`, `left-panel/media-item/`. **Risk:** fires on
  every vote — keep it fast and subtle or it will grate during rapid voting.

<!-- item-sep -->

- **List insertion/removal transitions** — When rows enter/leave the labeled
  list or media grid, transition height + opacity instead of a hard reflow
  jump. Pure jank reduction, essentially no distraction risk. Files:
  `left-panel/`, `left-panel/media-item/`.

<!-- item-sep -->

- **Re-sort settle (FLIP)** — After Find/Train re-ranks, let rows FLIP to their
  new positions (measure before/after, tween the delta) rather than snapping.
  This is motion that *explains* — "the results just reshuffled" — the good
  kind. Depends on the **flight/FLIP helper**. Files: results list in
  `left-panel/` / `find-view/`.

<!-- item-sep -->

- **Score-bar fill** — When detector scores arrive, animate each item's score
  bar filling from 0 rather than appearing full. Small, informative. Files:
  wherever the per-item score bar renders (`left-panel/media-item/`).

## Task / detector lifecycle (extends the waggle vocabulary)

<!-- item-sep -->

- **Completion pulse** — When a Find/Train job finishes, the waggling button
  does a single settle-and-pulse (one glow, not a loop) as a terminal state for
  `.icon-waggle`. Files: `_components.scss` (`.icon-waggle` siblings),
  dashboard/find buttons.

<!-- item-sep -->

- **Detector "trained" highlight** — A brief highlight sweep across the detector
  card when training completes, so a job that finishes while the user is
  elsewhere leaves a visible trace. Files: `dashboard/detector-card/`.

<!-- item-sep -->

- **Progress count-up** — Tween the numbers in the progress modal (items
  processed, embeddings computed) instead of jumping. Fits precedent: the
  `.progress-fill` indeterminate sweep is already exempt from the animations-off
  toggle as functional feedback. Files: `modals/progress-modal/`, `progress-bar/`.

## Panels, tabs, canvas

<!-- item-sep -->

<!-- item-sep -->

- **Tab-underline slide** — Slide the active-tab indicator between tabs instead
  of hard-cutting (dashboard, left-panel, achievements strips). Classic,
  unobtrusive. Files: tab styling in `_components.scss`.

<!-- item-sep -->

- **Browse bin-popup / hover-preview fade** — `browse-bin-popup/` and
  `browse-hover-preview/` pop instantly; a fast fade softens them. **Risk:** the
  hover preview fires constantly during browse — cap at `--transition-fast`
  (≤100ms) or it feels sticky, which is exactly the distraction to avoid. Files:
  `browse-bin-popup/`, `browse-hover-preview/`.

<!-- item-sep -->

- **Minimap viewport tween** — When the browse viewport jumps, tween the minimap
  rectangle to its new position so the spatial relationship stays readable.
  Files: `browse-minimap/`.

## Delight (higher distraction risk — spec conservatively)

<!-- item-sep -->

- **Achievement unlock flourish** — Unlocks are toast-only today
  (`achievement-unlock-host`). A badge that scales/settles (optionally a *brief*,
  one-shot particle burst) makes them feel earned. The one place a particle
  effect is defensible — must be one-shot and reduced-motion-gated. Depends on
  the **flight helper** for any flying-badge variant. Files:
  `achievement-unlock-host/`, `achievements-*`.

<!-- item-sep -->

- **First-good-detector celebration** — A rare, one-time flourish when a user's
  first detector crosses a quality bar. Deliberately rare ⇒ not distracting.
  Reuses whatever particle/flight helper the achievement flourish needs. Files:
  detector-card / a new one-shot celebrate helper.

## Suggested sequencing

1. **Infra first:** the flight/FLIP helper (the shared easing tokens have
   shipped; modal open/close is now unanimated). It unlocks the highest-value
   remaining items.
2. **Best payoff-to-risk:** re-sort FLIP settle, list insertion/removal
   transitions.
3. **Everything else** as independent issues, sized per item.

Recommended implementer models when these are promoted to issues: **Sonnet 5**
for the self-contained CSS items (tab underline, count-up, fades); **Opus 4.8**
for the flight/FLIP helper (shared infra, reactivity-sensitive, regression-prone
across ~25 modals).
