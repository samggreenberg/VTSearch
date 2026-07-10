# VTSBrowse right-click takes two clicks

**Status:** open — root cause not yet confirmed. Needs a live browser to distinguish the
hypotheses below (the container this was investigated in has no Chromium).

## Symptom

In the VTSBrowse canvas, opening a bin's detail popup by **right-click** takes **two**
right-clicks. Reported flow:

1. Mouse over an image bin → its representative thumbnail enlarges (hover works).
2. Right-click the (already-hovered, enlarged) bin → **nothing visible happens**; the bin
   just stays large. No detail popup.
3. A **second** right-click on the same bin → the detail popup finally opens.

## Repro

- Open any dataset in VTSBrowse (image media makes the enlarge obvious).
- Hover a bin so it enlarges, then right-click it once → expect popup, observe none.
- Right-click again → popup opens.

## Code map (line numbers as of this branch)

- `frontend/src/app/components/browse-canvas/browse-canvas.component.ts`
  - `onContextMenu` (~2079): right-click handler. Resolves the target bin as
    `const cell = this.hoveredCell ?? this.hitTest(mx, my);` (2091), `clearHover()` (2094),
    then sets `this.pinnedCell = cell` (2114), `requestRedraw()`, and
    `this.contextMenu.emit({...})` (2116).
  - Enlarge is drawn from `const enlargedCell = this.pinnedCell ?? this.hoveredCell;` (1240)
    → so whenever a bin *looks* enlarged with nothing pinned, `hoveredCell` is non-null.
  - Hover is debounced 30 ms in `onCanvasMouseMove` (2519) → `emitHoverHit` (2551) which sets
    `this.hoveredCell = hit`. Hover is suppressed while `pinnedCell` is set (2536, 2554).
  - `unpinCell()` (2131): called by the view when the popup is dismissed; clears `pinnedCell`
    and resumes hover at the last cursor position.
- `frontend/src/app/components/browse-view/browse-view.component.ts`
  - `onCanvasContextMenu` (~796): if `event.members.length === 0` → `dismissContextMenu()`;
    else sets `contextMembers/contextRepId/contextMenuX/Y/contextBounds` and
    `contextMenuOpen = true`.
  - `dismissContextMenu` (809): `contextMenuOpen = false; this.canvas?.unpinCell();`
- `frontend/src/app/components/browse-view/browse-view.component.html`
  - `@if (contextMenuOpen) { <vt-browse-bin-popup ... /> }` (117).
- `frontend/src/app/components/browse-bin-popup/browse-bin-popup.component.ts`
  - Popup is kept `visibility: hidden` until `placed` flips true. Template:
    `[style.visibility]="placed ? 'visible' : 'hidden'"` (popup .html line 7).
  - `placed` flips true **only** in `nudgeOnScreen` (1113), which is scheduled via
    `afterNextRender` from `place()` (1073), and **only when settings are loaded**:
    `const settingsReady = this.settingsState.settingsSignal() != null;` (1152).
  - `@HostListener('document:click')` `onDocumentClick` (732) dismisses on outside click.

## Already tried (present and intact on this branch)

The "first right-click" fix from commit `c65ca0c7` is already in place: `onContextMenu` reads
`this.hoveredCell` *before* falling back to a raw `hitTest`, precisely so a broken-out hovered
thumbnail (whose true hex the cursor may have left) still resolves. This alone does **not** fix
the reported symptom — so the failure is either downstream of cell resolution, or `hoveredCell`
is somehow null at contextmenu time despite the visible enlarge.

## Ruled out by static reading

- **Hover-preview overlay stealing the click** — `.hover-popup` has `pointer-events: none`
  (`browse-hover-preview.component.scss:8`), so it can't intercept the right-click.
- **`onMouseDown` interfering** — returns early for `event.button !== 0`
  (`browse-canvas.component.ts:1920`); the right button never starts a marquee/pan.
- **`document:click` dismissing on the same right-click** — a right button does not emit a
  `click` event, so `onDocumentClick` should not fire from the right-click itself. (Worth
  double-checking on the actual input device — see experiment.)
- **A recent regression** — commits `ed1097b6`, `2bf982ae`, `1fd71ad4` did not touch
  `onContextMenu` / `hoveredCell` / `emitHoverHit` / `pinnedCell`.
- **Settings never loaded** — both `app.component.ts` and `browse-view.component.ts` call
  `settingsState.load()`, so by the time the user is in Browse, `settingsSignal()` is normally
  non-null (weakens, but does not eliminate, hypothesis B).

## Leading hypotheses (to distinguish in a browser)

<!-- item-sep -->

- **A — Popup opens+pins but never reveals on the first summon (reveal/placement race).**
  On the first right-click, `cell` resolves, `pinnedCell` is set (bin stays large), and
  `contextMenuOpen` flips true — but the popup stays `visibility: hidden` because `placed`
  never flips to true on that first creation. Second right-click re-summons (new `memberIds`
  array reference → `ngOnChanges` → `place()` again) and *this* time it reveals. This matches
  the symptom exactly ("still just large, no window; second click opens it"). Suspect the
  `afterNextRender` → `nudgeOnScreen` chain not firing/placing on the component's very first
  creation under zoneless CD, or `settingsReady` being false at that instant. **Most likely.**

<!-- item-sep -->

- **B — `settingsSignal()` is null at first `nudgeOnScreen`.** A special case of A: if the
  Browse popup is the first thing to need settings after a `clear()`/navigation, the reveal
  gate at line 1152 holds `placed=false`, and the settings-load `effect` only re-`place()`s
  when `previewOverride`/`metadataShown` actually change (298) — which they don't on a plain
  first load — so the popup is stranded hidden until the next summon.

<!-- item-sep -->

- **C — `hoveredCell` is null at contextmenu despite the visible enlarge.** If the enlarge the
  user sees is a stale paint (hover cleared without repaint) or driven by a leftover
  `pinnedCell`, `onContextMenu` would fall to `hitTest`, which can miss under a broken-out
  thumbnail, resolve `cell = null`, pin nothing, and emit empty members → `dismissContextMenu`.
  Second click's `hitTest` (cursor nearer the true hex center) then succeeds. Less likely given
  the enlarge is `pinnedCell ?? hoveredCell`, but cheap to confirm.

<!-- item-sep -->

- **D — Popup reveals then is dismissed within the same gesture.** e.g. the input device (Mac
  trackpad two-finger tap, or a mouse that also emits a synthetic `click`) triggers
  `onDocumentClick` → `dismissContextMenu` right after opening. Confirm by logging in
  `onDocumentClick`/`dismissContextMenu`.

## Instrumentation to add before running (temporary `console.debug`)

Add these logs, rebuild (`cd frontend && npm run build:prod`), reproduce, and read the console
across click #1 vs click #2. Remove before committing the real fix.

- `browse-canvas.component.ts` `onContextMenu`, right after line 2091:
  `console.debug('[ctx] hovered=', !!this.hoveredCell, 'hit=', !!this.hitTest(mx,my), 'cell=', !!cell);`
- `browse-view.component.ts` `onCanvasContextMenu`, first line:
  `console.debug('[view] members=', event.members.length, 'repId=', event.repId);`
- `browse-view.component.ts` `dismissContextMenu`, first line:
  `console.debug('[view] dismiss', new Error().stack);`
- `browse-bin-popup.component.ts`:
  - `ngOnChanges` first line: `console.debug('[popup] onChanges', Object.keys(changes));`
  - `place()` first line: `console.debug('[popup] place panel=', !!this.panelRef?.nativeElement);`
  - `nudgeOnScreen` right before line 1153: `console.debug('[popup] nudge placed=', this.placed, 'settingsReady=', settingsReady, 'moved=', moved);`
  - `onDocumentClick`, inside the `if`: `console.debug('[popup] documentClick dismiss, target=', event.target);`

## Experiment protocol

1. Hover a bin until it enlarges. Right-click once. Record every `[ctx]/[view]/[popup]` line.
2. Right-click the same bin again. Record the lines.
3. Compare click #1 vs #2. Decision tree:
   - `[ctx] cell=false` on click #1 but `true` on #2 → **hypothesis C** (cell resolution).
   - `[ctx] cell=true` + `[view] members>0` on click #1, but no `[popup] nudge placed=true`
     (or `settingsReady=false`) → **hypothesis A/B** (reveal gate).
   - `[popup] documentClick dismiss` / `[view] dismiss` fires right after opening on click #1 →
     **hypothesis D**.

## Candidate fixes (apply the one the experiment points to; then re-verify in the browser)

<!-- item-sep -->

- **For A/B (reveal race) — decouple the first reveal from settings + guarantee a place().**
  In `browse-bin-popup.component.ts`: (1) In the settings `effect` (280), call `place()` on the
  *first* settings arrival even when override/metadata didn't change, so a popup that mounted
  before settings loaded still gets revealed. (2) Consider relaxing the `settingsReady` gate at
  1152 so `placed` can flip on the first `nudgeOnScreen` regardless, accepting a possible
  re-clamp when settings land (the popup already re-clamps on the settings effect). Verify the
  first right-click now reveals, and that there's no flash at default sizes.

<!-- item-sep -->

- **For C (cell resolution) — make `onContextMenu` prefer the enlarged cell unconditionally.**
  Resolve against `this.pinnedCell ?? this.hoveredCell ?? this.hitTest(mx, my)` and/or widen the
  `hitTest` tolerance for a broken-out hovered thumbnail. Confirm `cell` is non-null on the
  first click for a bin the cursor is resting on.

<!-- item-sep -->

- **For D (spurious dismiss) — ignore the dismiss that belongs to the opening gesture.**
  Guard `onDocumentClick` so it does not dismiss for a click whose timeStamp coincides with the
  summoning contextmenu (or switch the outside-dismiss listener to `mousedown`/`pointerdown`
  and skip button 2). Confirm the popup survives the opening right-click.

## Notes for the runner

- Build: `cd frontend && npm install && npm run build:prod` (Angular → `static/`), then
  `python app.py --local` and load a dataset into Browse.
- After confirming root cause and a fix, remove the `console.debug` instrumentation, run
  `./run-tests.sh frontend` (build + audit + Vitest) and `./run-tests.sh` for the full suite,
  add a regression test (zoneless spec next to `browse-bin-popup.zoneless.spec.ts` or a
  browse-canvas spec), and delete this plan file per the repo's plan-file policy (fold any
  durable rationale into the popup component's doc comments).
