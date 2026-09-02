/**
 * Tile-set geometry for the browse canvas's prefetch paths.
 *
 * The canvas warms two things ahead of the user: tile *geometry* (from the tile
 * cache) and representative *thumbnails* (from the thumb store). Both need the
 * same question answered first — which tiles is the view about to need? — and
 * that question is pure geometry over the pyramid:
 *
 * - {@link offViewRing} — the tiles a **pan** is about to scroll in.
 * - {@link finerTilesForZoom} — the tiles a **zoom-in** is about to reveal.
 * - {@link tilesAtLevel} — the same region re-expressed at an adjacent level.
 * - {@link smoothPanDirection} — the direction estimate that biases the ring.
 *
 * Everything here is a pure function of numbers and tile coordinates: no tile
 * cache, no thumbnails, no canvas, no Angular. That keeps the "what to warm"
 * decision testable on its own, separately from the "how to fetch and retain
 * it" half in `thumb-store.ts`.
 */

/** A tile address within one pyramid level. */
export interface TileCoord {
  tx: number;
  ty: number;
}

/**
 * How strong the smoothed pan direction must be on an axis before the ring is
 * extended on that side. Below this the motion is treated as incidental jitter
 * and the ring stays symmetric, so a view being nudged around doesn't keep
 * flip-flopping which side it warms.
 */
export const PAN_DIR_BIAS_THRESHOLD = 0.3;

/** Exponential smoothing factor for the pan-direction estimate: how much of
 *  each new frame's direction is folded in. Low enough that one jittery frame
 *  can't flip the bias, high enough to track a real gesture within a few frames. */
export const PAN_DIR_SMOOTHING = 0.4;

/** A smoothed unit-ish pan direction (each component in [-1, 1]). */
export interface PanDirection {
  x: number;
  y: number;
}

/**
 * Fold one frame's movement into the smoothed pan direction.
 *
 * ``from`` and ``to`` are consecutive view centres in projection units. The
 * exponential smoothing keeps a single jittery frame from flipping the bias and
 * lets the direction decay toward zero when the view holds still, so a
 * stationary view warms its ring symmetrically. Returns a new value rather than
 * mutating, so the caller owns the state.
 */
export function smoothPanDirection(
  previous: PanDirection,
  from: { x: number; y: number },
  to: { x: number; y: number },
  smoothing = PAN_DIR_SMOOTHING,
): PanDirection {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const mag = Math.hypot(dx, dy);
  const ux = mag > 1e-6 ? dx / mag : 0;
  const uy = mag > 1e-6 ? dy / mag : 0;
  return {
    x: previous.x + (ux - previous.x) * smoothing,
    y: previous.y + (uy - previous.y) * smoothing,
  };
}

/**
 * The ring of tiles just outside the drawn set: the bounding box of
 * ``visibleTiles`` grown by one tile on every side (8-connected, so diagonals
 * are included), minus the box itself.
 *
 * The growth is extended by a second tile on whichever sides the view is panning
 * toward, so a sustained pan warms further ahead in the direction of travel.
 * Callers pass a ``visibleTiles`` set that already carries a one-tile margin, so
 * this ring sits one to two tiles beyond what is actually painted.
 */
export function offViewRing(visibleTiles: TileCoord[], panDir: PanDirection): TileCoord[] {
  if (visibleTiles.length === 0) return [];
  let txMin = Infinity;
  let txMax = -Infinity;
  let tyMin = Infinity;
  let tyMax = -Infinity;
  for (const { tx, ty } of visibleTiles) {
    if (tx < txMin) txMin = tx;
    if (tx > txMax) txMax = tx;
    if (ty < tyMin) tyMin = ty;
    if (ty > tyMax) tyMax = ty;
  }
  // One ring on every side, plus a directional extra tile on the leading edges.
  const t = PAN_DIR_BIAS_THRESHOLD;
  const outTxMin = txMin - (panDir.x < -t ? 2 : 1);
  const outTxMax = txMax + (panDir.x > t ? 2 : 1);
  const outTyMin = tyMin - (panDir.y < -t ? 2 : 1);
  const outTyMax = tyMax + (panDir.y > t ? 2 : 1);
  const ring: TileCoord[] = [];
  for (let tx = outTxMin; tx <= outTxMax; tx++) {
    for (let ty = outTyMin; ty <= outTyMax; ty++) {
      // Keep only the new border; the interior is the already-drawn box.
      if (tx >= txMin && tx <= txMax && ty >= tyMin && ty <= tyMax) continue;
      ring.push({ tx, ty });
    }
  }
  return ring;
}

/**
 * The next finer level's tiles covering the current viewport: what a zoom-in
 * would render.
 *
 * A finer level halves the bin radius, so each visible tile maps to a 2×2 block
 * of finer tiles. The union is returned sorted by distance from the view centre,
 * since a zoom-in anchors near the centre (or the cursor) and so reveals the
 * central tiles first. ``centreTx`` / ``centreTy`` are the view centre expressed
 * in *finer-level tile units*, which is what makes the ranking a plain squared
 * distance.
 */
export function finerTilesForZoom(
  visibleTiles: TileCoord[],
  centreTx: number,
  centreTy: number,
): TileCoord[] {
  const seen = new Set<string>();
  const tiles: { tx: number; ty: number; d2: number }[] = [];
  for (const { tx, ty } of visibleTiles) {
    // Each current-level tile spans a 2×2 block at the finer level.
    for (let dx = 0; dx <= 1; dx++) {
      for (let dy = 0; dy <= 1; dy++) {
        const ftx = tx * 2 + dx;
        const fty = ty * 2 + dy;
        const key = `${ftx}:${fty}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const ex = ftx + 0.5 - centreTx;
        const ey = fty + 0.5 - centreTy;
        tiles.push({ tx: ftx, ty: fty, d2: ex * ex + ey * ey });
      }
    }
  }
  tiles.sort((a, b) => a.d2 - b.d2);
  return tiles.map(({ tx, ty }) => ({ tx, ty }));
}

/**
 * Re-express a set of source tiles at an adjacent pyramid level, padded by one
 * tile on every side.
 *
 * ``ratio`` is the source level's bin radius over the target level's, so a
 * coarser target (bigger bins) gives a ratio below 1 and a finer target one
 * above. Each source tile maps to a target tile plus its 8-connected
 * neighbourhood, de-duplicated — the padding covers the fact that the two
 * levels' tile grids don't align, so a source tile can straddle a target
 * boundary.
 */
export function tilesAtLevel(sourceTiles: TileCoord[], ratio: number): TileCoord[] {
  const seen = new Set<string>();
  const out: TileCoord[] = [];
  for (const { tx, ty } of sourceTiles) {
    const ttx = Math.floor(tx * ratio);
    const tty = Math.floor(ty * ratio);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const key = `${ttx + dx}:${tty + dy}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ tx: ttx + dx, ty: tty + dy });
      }
    }
  }
  return out;
}
