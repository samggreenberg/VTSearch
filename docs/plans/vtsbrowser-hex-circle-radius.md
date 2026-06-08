# VTSBrowse: singleton circle radius == bin interior radius

**Status:** Investigation only — needs *visual* verification in a browser-capable
environment before any code change. Blind code reading suggests the requested
behavior is **already implemented**, but the user reports "you're not seeing it,"
so a future Claude that can actually render the canvas should look before editing.

## The request

> In the hex display for VTSBrowser, the circles should have the same radius as
> the **interior** radius of the hexes. So the singletons are **smaller area**
> than the plural bins.

i.e. a cell holding exactly one item is drawn as a disc, and that disc should be
the hex's *inscribed* circle (radius = apothem / inradius = √3/2 · circumradius),
which is visibly smaller in area than the surrounding hex. Same idea for the
square bin shape: the disc should be the square's inscribed circle.

## What the code does today (HEAD `c2056231`, dev)

The browse canvas got a **hex/square bin-shape toggle** in commit `bd2c9b3e`
("Add hex/square bin-shape toggle to VTSBrowse"). Cell drawing now goes through a
`BinGeometry` abstraction, **not** the old `traceCellPath` import directly. Note:
an earlier read of this repo at commit `239ecf18` showed the pre-toggle code
(`hex-render.util.ts` `traceCellPath` called directly from both canvas and
minimap); the working tree advanced to `c2056231` mid-session, so trust the
`bin-geometry.ts` path described here.

### Render pipeline

- **Full canvas:** `browse-canvas.component.ts`
  - `get geom()` (line ~124) → `binGeometry(this.meta?.bin_shape)`
  - `drawHex(ctx, cx, cy, radius, cell)` (line ~363): `single = cell.count === 1`,
    then `this.geom.traceCell(ctx, cx, cy, radius, single)` (line ~373).
  - `radius` here is `screenRadius`, the cell **circumradius** in screen px.
  - In thumbnail mode the thumb is `ctx.clip()`-ed to that same path, then
    `drawImageCover(..., radius)` paints a `2·radius` square cover behind the clip
    (line ~382, ~406). So the disc clip already constrains singleton thumbnails.
- **Minimap:** `browse-minimap.component.ts` (line ~232–239)
  - `geom = binGeometry(this.meta!.bin_shape)`, `cellR = base_radius/2^level · scale`,
    then `geom.traceCell(ctx, sx, sy, cellR, cell.count === 1)`.

### The geometry (`frontend/src/app/components/browse-canvas/bin-geometry.ts`)

```ts
// HEX
traceCell: (ctx, cx, cy, radius, single) => traceHexCellPath(ctx, cx, cy, radius, single),
// → hex-render.util.ts:
//   HEX_INRADIUS_RATIO = SQRT3 / 2;            // ≈ 0.866
//   if (single) ctx.arc(cx, cy, radius * HEX_INRADIUS_RATIO, 0, 2π);  // inscribed disc
//   else        traceHexPath(ctx, cx, cy, radius);                    // full hex (vertices at `radius`)

// SQUARE
traceCell: (ctx, cx, cy, radius, single) => {
  const half = (radius * SQRT3) / 2;           // square side = radius·√3, so half = inradius
  if (single) ctx.arc(cx, cy, half, 0, 2π);    // inscribed disc
  else        ctx.rect(cx - half, cy - half, half*2, half*2);  // full square
}
```

`traceHexPath` puts vertices at distance `radius` from center (pointy-top, angles
`(π/3)·i − π/6`), so **`radius` is the circumradius**. The hex's interior radius
(apothem) is `radius·√3/2`, which is exactly the disc radius used for singletons.

### Conclusion from reading the code

For **both** shapes the singleton disc is already the bin's inscribed circle
(interior radius):
- Hex disc radius = `radius·√3/2` (apothem). Disc area ≈ `π·0.75·R²` ≈ **0.907×**
  the hex area (`(3√3/2)·R²`). Smaller — as requested.
- Square disc radius = `side/2` (apothem). Disc area = `π/4` ≈ **0.785×** the
  square area. Smaller — as requested.

So a literal reading says the request is **satisfied**. That contradicts the
user's report, which is why this needs eyes-on verification.

## Why the user might still be seeing it "wrong" — hypotheses to check visually

A future Claude with a real browser should open VTSBrowse and check these, in
order of likelihood:

1. **Visual subtlety, not a bug.** `√3/2 ≈ 0.866` shrinks the *radius* by only
   ~13% (~9% area for hex). At small on-screen `radius`, a disc inscribed in a hex
   can look nearly the same size as the hex. The user may expect a *more obviously*
   smaller dot. If so, the fix is a smaller multiplier (e.g. a tunable
   `SINGLETON_DISC_RATIO < HEX_INRADIUS_RATIO`), **not** the apothem. Confirm with
   the user what "interior radius" should mean if the apothem isn't visually
   distinct enough. → goes through `AskUserQuestion` per CLAUDE.md.
2. **Stale build / wrong branch.** Confirm the running app is built from `dev`
   (HEAD ≥ `bd2c9b3e`). The public default is `main`; if the user was viewing a
   `main`/production build, singleton discs may predate this and render at the
   full circumradius. Rebuild frontend (`cd frontend && npm run build:prod`) and
   reload before judging.
3. **`base_radius` semantics.** Verify (via the projection API / `meta`) that the
   `radius` threaded into `traceCell` is genuinely the **circumradius**, not the
   apothem. If the projection backend already hands the canvas an apothem-scaled
   radius, multiplying by `√3/2` would make the disc *too small*, not the interior
   radius. Check `projection.models.ts` `meta.base_radius` and the pyramid layout
   in `vtscore`/`vtsearch` projection code (`docs/plans/vtsbrowse.md`).
4. **Hex vs square mismatch.** The user said "hexes"; confirm the active
   `bin_shape` is `hex`. If they were toggled to `square`, the geometry is
   different (and also already inscribed).
5. **Hover/stroke overpaint.** The hovered cell strokes white at `lineWidth 2`
   (line ~392); on a small disc that stroke could make it read closer to the hex
   footprint. Probably not the cause, but note it.

## What to actually do (browser-capable session)

1. Run the app with a projected dataset and open VTSBrowse. See
   `docs/plans/browser-vision-testing.md` for the vision-testing workflow, and the
   `run` / `verify` skills. (No Chrome in the cloud container — this *must* be a
   browser-capable env.)
2. Zoom to a region with both singleton and multi-item cells. Screenshot.
3. Decide which hypothesis above is true. If it's #1 (want smaller-than-apothem),
   ask the user for the exact desired ratio via `AskUserQuestion`, then change the
   multiplier in `bin-geometry.ts` / `hex-render.util.ts` (`HEX_INRADIUS_RATIO`
   and the square `half`). If it's #3, fix the radius semantics instead.
4. Add/adjust a `tests_lib/projection/` test if the geometry constant changes
   (this is the `projection` test group). Run `./run-tests.sh core` (frontend
   build) + the projection group.

## Key files

- `frontend/src/app/components/browse-canvas/bin-geometry.ts` — `traceCell` per shape (disc vs full).
- `frontend/src/app/components/browse-canvas/hex-render.util.ts` — `HEX_INRADIUS_RATIO = √3/2`, `traceCellPath`, `traceHexPath`.
- `frontend/src/app/components/browse-canvas/browse-canvas.component.ts` — `drawHex` (~363), `geom` (~124), `drawImageCover` (~406).
- `frontend/src/app/components/browse-minimap/browse-minimap.component.ts` — `render` (~220), `traceCell` call (~236).
- `frontend/src/app/models/projection.models.ts` — `BinShape`, `meta` (`base_radius`, `bin_shape`, `tile_span`).
- `docs/plans/vtsbrowse.md`, `docs/plans/browser-vision-testing.md` — projection design + vision-testing workflow.

## Open question for the user (when resuming)

If the apothem-inscribed disc is in fact what's rendering, the real ask is likely
"make the singleton dot *noticeably* smaller than the hex." Need the user to pick
the target size (apothem as-is / a smaller fixed fraction / explicit px) before
editing — via `AskUserQuestion`, not prose.
