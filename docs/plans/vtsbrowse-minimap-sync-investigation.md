# VTSBrowse minimap "viewed rectangle" mis-sync — investigation handoff

> **Status:** Investigated headlessly (no browser in this environment), root
> cause **not yet confirmed**. This note records the static-analysis findings so
> the next Claude — running where the rendered canvas can actually be *seen and
> operated* — can confirm the repro and land the fix quickly. See the
> *§What to do next* checklist at the bottom.

## The reported symptom

> "There's a mis-sync between the VTSBrowse minimap 'viewed rectangle' vs the
> actual rectangle we're viewing. The map shows that I'm only seeing **half the
> heatmap, aimed too north**. But I'm definitely seeing the **whole thing**."

So: the white viewport rectangle drawn on the minimap covers roughly the
**north (top) half** of the minimap's density heatmap, while the main canvas is
actually showing the entire projection.

The reporter has **not** yet confirmed whether the rectangle's *width* is
correct (vertical-only error) or whether it's a uniform both-axes shrink — that
distinction is the single most important thing to nail down first (see below).

## How the feature is wired (file map)

| Concern | File | Key bits |
|---|---|---|
| Main canvas (pan/zoom, draws hexes, **publishes** the visible region) | `frontend/src/app/components/browse-canvas/browse-canvas.component.ts` | `projToScreen`/`screenToProj` (L237-249), `getVisibleBounds` (L251-260), publishes via `this.viewport.setViewport(this.getVisibleBounds())` in `draw()` (L341) |
| Minimap (draws overview heatmap + the **viewport rectangle**, handles click-to-recenter) | `frontend/src/app/components/browse-minimap/browse-minimap.component.ts` | `fit()` (L196-207), `projToMap`/`mapToProj` (L209-215), `drawViewportRect` (L257-276) |
| Pub/sub channel between them | `frontend/src/app/services/browse-viewport.service.ts` | `viewport$` (BehaviorSubject, `[xmin,ymin,xmax,ymax]` in **projection space**), `recenter$` |
| Host that owns both + the shared service | `frontend/src/app/components/browse-view/browse-view.component.{ts,html,scss}` | provides `BrowseViewportService`; passes `displayScale`, minimap size |
| Projection meta / hex models | `frontend/src/app/models/projection.models.ts` | `ProjectionMeta.bounds = [xmin,ymin,xmax,ymax]`, `HexCellPayload.{cx,cy}` |
| Backend: 2-D layout + **bounds** | `vtscore/projection/umap_projection.py` | `Projection.bounds` = `coords.min/max(axis=0)` (L52-59) |
| Backend: hex binning + cell centers | `vtscore/projection/hexbin.py` | `hexbin_assign` (L33-84), `hex_center` → `cy = r * 1.5 * radius` (L87-101) |
| Backend: pyramid/tiles, passes `bounds` through unchanged | `vtscore/projection/pyramid.py` | `_build_level` (L140-175), `Pyramid.meta()` (L108-117) |

### The coordinate chain (all in CSS px / projection units, DPR handled separately)

1. **Canvas → projection.** `screenToProj(sx,sy) = ((sx - W/2)/z + centerX, (sy - H/2)/z + centerY)` where `z = effZoom = transform.zoom * displayScale`.
2. **Visible region published** = `getVisibleBounds()` = projToProj of the two screen corners `(0,0)` and `(W,H)`, min/max'd. Published to `viewport$` in **projection space** (DPR- and zoom-independent).
3. **Minimap → screen.** `fit()` computes one uniform `scale = min((width-8)/dataW, (height-8)/dataH)` and centers the data at `(width/2, height/2)`. `projToMap(px,py) = (width/2 + (px-cx)*scale, height/2 + (py-cy)*scale)` where `cx,cy = data-bounds center`.
4. **Rectangle** = `projToMap` of `viewport$`'s `[xmin,ymin]` and `[xmax,ymax]`, drawn over the heatmap with the **same `f`** the heatmap uses.

## What static analysis rules OUT

- **No Y-axis flip anywhere.** Backend `bounds` and hex `(cx,cy)` derive from the
  same coords; frontend `screenToProj`/`projToScreen`/`projToMap`/`mapToProj`
  all keep Y oriented the same way (screen-top ↔ smaller proj-Y ↔ minimap-top).
- **No axis-asymmetric code.** Every transform uses a **single scalar** for both
  axes (`effZoom`, the minimap `scale`, the `fitToData` zoom). There is *no*
  place where X and Y get different treatment. **Consequence:** a purely
  vertical error (correct width, wrong height/position) is essentially
  impossible to produce from these transforms alone — any genuine mismatch is
  either a *uniform* both-axes scale error or a *center shift*.
- **Heatmap and rectangle are locked together.** In `draw()` both use the same
  `f = this.fit()`, so the rectangle can't be scaled/offset relative to the
  heatmap by a transform bug — only by the *published bounds* being wrong.
- **The rectangle is centered on a clean load.** On first fit,
  `transform.centerX/Y = data-bounds center`, and the minimap centers the data
  at its own center, so `projToMap(center) = (width/2, height/2)`. The rectangle
  is therefore **centered**, not north-shifted. A north shift requires
  `transform.centerY != data-center`, which on a clean load only happens after
  the user **pans** (or wheel-zooms off-center).

**Net:** the exact reported shape ("correct-ish width, half height, aimed
north") does **not** fall out of the code for an untouched, freshly-loaded view.
That mismatch is why the repro details matter — see *§Hypotheses*.

## The one concrete structural defect found (likely related, maybe not the whole story)

`fitToData()` is called **only** from `ngOnChanges(meta)`
(`browse-canvas.component.ts:163`). It is **never** re-run from `resize()`.

```ts
private fitToData(): void {
  ...
  const w = this.width || 800;   // <-- fallback when layout hasn't happened yet
  const h = this.height || 600;  // <--
  this.transform.zoom = Math.min(w / padW, h / padH) / this.displayScale;
  this.transform.centerX = (xmin + xmax) / 2;
  this.transform.centerY = (ymin + ymax) / 2;
  ...
}
```

If `meta` arrives **before the canvas has laid out** (`this.width === 0`), the
fit is computed against the hardcoded **800×600** fallback and then **never
corrected** when the real `ResizeObserver` size arrives. The real `resize()`
(L192-205) updates `width/height` and redraws, but does **not** refit, so the
zoom stays sized for 800×600.

- Effect: the **published viewport is a uniformly wrong size** vs. the data
  (too big if the real canvas is larger than 800×600, too small if smaller) —
  but still **centered**. On a typical large browse area this usually reads as
  "rectangle bigger than the heatmap" (you see margin around the data), not
  "half." On a *short* browse area it could read as a centered band covering the
  middle ~half. **Neither explains "aimed north."**

This is still worth fixing regardless (it's a real sync defect), but on its own
it does not reproduce the reported north bias.

## Hypotheses, ranked (for the browser-equipped session to confirm/reject)

1. **It's actually after a pan, not a clean load.** A north-half rectangle is
   exactly what you'd get if `transform.centerY` is north of the data center and
   zoom is high — i.e. the canvas really is showing the north half. If the user
   believes they "see the whole thing" but the data's dense cluster is in the
   north (sparse south), they might not notice the south is cut off. *Check: does
   it happen on a genuinely untouched load?*
2. **Never-refit size mismatch (the defect above)**, perceived as "half" because
   the data clusters north and the centered-but-wrong-size rectangle happens to
   sit over the dense north region. *Check: does it only happen when `meta`
   arrives before layout — e.g. fast-loading cached projection — and does
   forcing a refit on first real resize fix it?*
3. **Minimap heatmap is half-blank, not the rectangle.** The overview heatmap
   renders a **deeper** pyramid level than the canvas (`overviewLevel`, targets
   ~5px hexes vs the canvas's ~28px), so its tiles aren't already in cache from
   the canvas and must be fetched by `requestOverviewTiles()`. Note the
   **runaway guard** in `overviewTiles()`:
   `if ((txMax-txMin)*(tyMax-tyMin) > 1024) return []` — for a deep overview
   level on an elongated projection this can return **no tiles**, or async
   timing can leave part of the field unpainted, so density appears in only part
   of the map. *Check: is the white rectangle actually fine and it's the
   coloring that's missing in the south?*
4. **DPR / layout edge case** making `this.height` not what the rectangle math
   assumes. Considered and not found in static analysis, but a real browser may
   reveal a stale `getBoundingClientRect` or a parent that includes/excludes
   space. *Check with devtools: log `this.width/this.height`, `transform`, and
   the published `getVisibleBounds()` vs `meta.bounds`.*

## What to do next (browser-equipped session)

1. **Reproduce and classify** using the two questions that are still open:
   - Does it happen on a **fresh, untouched load**, or only **after
     resize/pan/zoom**?
   - Is the rectangle's **width correct** (→ genuinely vertical-only, which
     means there's axis-asymmetric behavior we haven't found and should hunt
     for) or **too small in both axes** (→ uniform scale mismatch, i.e.
     hypothesis 2)?
2. **Instrument** `draw()` in `browse-canvas.component.ts`: temporarily
   `console.log` `this.width`, `this.height`, `this.transform`,
   `getVisibleBounds()`, and `this.meta.bounds`. Compare the published bounds to
   the data bounds and to what's visually on screen. This single log resolves
   hypotheses 1 vs 2 immediately.
3. **If hypothesis 2 (never-refit):** make the canvas refit (or reframe) when it
   first learns its real size. Safe approach: in `resize()`, if the previous fit
   was done against the fallback (track e.g. `private fittedAgainstRealSize =
   false`), call `fitToData()` once the real size is known; **do not** refit on
   every resize (that would clobber the user's pan/zoom on window resize).
   Alternatively, defer the initial `fitToData()` until after the first
   `ResizeObserver` callback. Either way, ensure `draw()` re-publishes the
   viewport afterward.
4. **If hypothesis 1 (it's really showing the north half):** the bug is upstream
   of the minimap — the canvas itself is mis-framed; same refit fix likely
   applies, plus verify `centerY` after fit equals `(ymin+ymax)/2`.
5. **If hypothesis 3 (heatmap half-blank):** fix `overviewTiles()` coverage —
   raise/replace the `> 1024` guard for the overview level, or clamp
   `overviewLevel` shallower so the field always paints fully.
6. **Add a regression test** once the cause is known. The transform math is
   pure and unit-testable headlessly: feed known `bounds` + a known
   `transform`/canvas size, assert `getVisibleBounds()` and the minimap
   `projToMap(viewportBounds)` produce a centered, correctly-sized rectangle.
   Put it under `frontend` specs (they must at least typecheck) or, better, a
   small pure-TS helper extracted from the components so it's directly testable.

## Quick reference: the exact functions in play

`browse-canvas.component.ts`
```ts
private screenToProj(sx, sy) {
  const z = this.effZoom;                 // transform.zoom * displayScale
  return [(sx - this.width/2)/z + this.transform.centerX,
          (sy - this.height/2)/z + this.transform.centerY];
}
private getVisibleBounds() {
  const [xmin,ymin] = this.screenToProj(0,0);
  const [xmax,ymax] = this.screenToProj(this.width, this.height);
  return [min(xmin,xmax), min(ymin,ymax), max(xmin,xmax), max(ymin,ymax)];
}
// draw(): ... this.viewport.setViewport(this.getVisibleBounds());
```

`browse-minimap.component.ts`
```ts
private fit() {
  const [xmin,ymin,xmax,ymax] = this.meta.bounds;
  const scale = Math.min((this.width-8)/(xmax-xmin||1),
                         (this.height-8)/(ymax-ymin||1));
  return { scale, cx:(xmin+xmax)/2, cy:(ymin+ymax)/2, margin:4 };
}
private projToMap(px, py, f) {
  return [this.width/2 + (px-f.cx)*f.scale,
          this.height/2 + (py-f.cy)*f.scale];
}
// drawViewportRect(): projToMap(vxmin,vymin) .. projToMap(vxmax,vymax)
```

Both sides exchange data only through `viewport$` as projection-space
`[xmin,ymin,xmax,ymax]`, so any mismatch is in (a) what the canvas *publishes*
(framing/zoom/center) or (b) what the minimap *draws as the heatmap* (tile
coverage) — the rectangle transform itself is faithful.
