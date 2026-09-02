import { describe, expect, it } from 'vitest';

import {
  finerTilesForZoom,
  offViewRing,
  PAN_DIR_BIAS_THRESHOLD,
  smoothPanDirection,
  tilesAtLevel,
  type TileCoord,
} from './prefetch-geometry';

/**
 * Unit coverage for the browse canvas's prefetch tile-set geometry: which tiles
 * a pan is about to scroll in, which a zoom-in is about to reveal, how a region
 * re-expresses at an adjacent pyramid level, and the smoothed direction estimate
 * that biases the pan ring.
 */

/** Stable key for set comparisons, since the functions don't promise an order
 *  (except {@link finerTilesForZoom}, whose ranking is asserted directly). */
const key = ({ tx, ty }: TileCoord) => `${tx}:${ty}`;
const keys = (tiles: TileCoord[]) => tiles.map(key).sort();

/** The tiles of the inclusive box [x0,x1]×[y0,y1]. */
function box(x0: number, x1: number, y0: number, y1: number): TileCoord[] {
  const out: TileCoord[] = [];
  for (let tx = x0; tx <= x1; tx++) for (let ty = y0; ty <= y1; ty++) out.push({ tx, ty });
  return out;
}

const STILL = { x: 0, y: 0 };

describe('offViewRing', () => {
  it('returns nothing for an empty visible set', () => {
    expect(offViewRing([], STILL)).toEqual([]);
  });

  it('rings a single tile with its 8 neighbours', () => {
    const ring = offViewRing([{ tx: 4, ty: 7 }], STILL);
    expect(keys(ring)).toEqual(keys(box(3, 5, 6, 8).filter((t) => key(t) !== '4:7')));
  });

  it('excludes the visible box itself', () => {
    const visible = box(0, 2, 0, 2);
    const ring = offViewRing(visible, STILL);
    for (const tile of visible) expect(keys(ring)).not.toContain(key(tile));
  });

  it('rings the bounding box, not each tile, for a ragged visible set', () => {
    // A sparse L-shape: the ring is still the box's border, so interior holes
    // are not re-warmed as if they were outside.
    const ragged: TileCoord[] = [
      { tx: 0, ty: 0 },
      { tx: 2, ty: 0 },
      { tx: 0, ty: 2 },
    ];
    const ring = offViewRing(ragged, STILL);
    expect(keys(ring)).toEqual(
      keys(box(-1, 3, -1, 3).filter((t) => !(t.tx >= 0 && t.tx <= 2 && t.ty >= 0 && t.ty <= 2))),
    );
  });

  it('extends two tiles on the leading edge when panning right', () => {
    const ring = offViewRing(box(0, 1, 0, 1), { x: 1, y: 0 });
    const txs = ring.map((t) => t.tx);
    // Leading (+x) edge reaches 3; trailing edge stays at the single ring.
    expect(Math.max(...txs)).toBe(3);
    expect(Math.min(...txs)).toBe(-1);
    // The cross-axis is unbiased.
    const tys = ring.map((t) => t.ty);
    expect(Math.max(...tys)).toBe(2);
    expect(Math.min(...tys)).toBe(-1);
  });

  it('extends the negative edge when panning left/up', () => {
    const ring = offViewRing(box(0, 1, 0, 1), { x: -1, y: -1 });
    expect(Math.min(...ring.map((t) => t.tx))).toBe(-2);
    expect(Math.min(...ring.map((t) => t.ty))).toBe(-2);
    expect(Math.max(...ring.map((t) => t.tx))).toBe(2);
    expect(Math.max(...ring.map((t) => t.ty))).toBe(2);
  });

  it('stays symmetric for a drift below the bias threshold', () => {
    const drift = PAN_DIR_BIAS_THRESHOLD * 0.9;
    expect(keys(offViewRing(box(0, 1, 0, 1), { x: drift, y: drift }))).toEqual(
      keys(offViewRing(box(0, 1, 0, 1), STILL)),
    );
  });

  it('never repeats a tile', () => {
    const ring = offViewRing(box(0, 3, 0, 3), { x: 1, y: 1 });
    expect(new Set(ring.map(key)).size).toBe(ring.length);
  });
});

describe('finerTilesForZoom', () => {
  it('maps each visible tile to its 2x2 finer block', () => {
    const finer = finerTilesForZoom([{ tx: 3, ty: 5 }], 7, 11);
    expect(keys(finer)).toEqual(['6:10', '6:11', '7:10', '7:11']);
  });

  it('de-duplicates the union across adjacent visible tiles', () => {
    const finer = finerTilesForZoom(box(0, 1, 0, 1), 2, 2);
    // Four source tiles × 4 = 16 finer tiles, all distinct.
    expect(finer).toHaveLength(16);
    expect(new Set(finer.map(key)).size).toBe(16);
  });

  it('ranks tiles centre-out, since a zoom-in reveals the centre first', () => {
    const finer = finerTilesForZoom(box(0, 2, 0, 2), 3, 3);
    // Centre (3,3) in finer-tile units sits on the corner of tiles 2..3; the
    // nearest tile centres (x.5) are the four touching it.
    expect(keys(finer.slice(0, 4))).toEqual(['2:2', '2:3', '3:2', '3:3']);
    // ...and the ranking is monotone in distance from the centre.
    const d2 = finer.map(({ tx, ty }) => (tx + 0.5 - 3) ** 2 + (ty + 0.5 - 3) ** 2);
    for (let i = 1; i < d2.length; i++) expect(d2[i]).toBeGreaterThanOrEqual(d2[i - 1]);
  });

  it('returns nothing for an empty visible set', () => {
    expect(finerTilesForZoom([], 0, 0)).toEqual([]);
  });
});

describe('tilesAtLevel', () => {
  it('pads a mapped tile with its 8-connected neighbourhood', () => {
    // ratio 1: the same grid, so the result is the 3x3 around the source.
    expect(keys(tilesAtLevel([{ tx: 5, ty: 5 }], 1))).toEqual(keys(box(4, 6, 4, 6)));
  });

  it('halves coordinates toward a coarser level', () => {
    // A coarser level has bins twice as wide, so ratio = 1/2 and tile 6 maps to 3.
    const out = tilesAtLevel([{ tx: 6, ty: 8 }], 0.5);
    expect(keys(out)).toEqual(keys(box(2, 4, 3, 5)));
  });

  it('doubles coordinates toward a finer level', () => {
    expect(keys(tilesAtLevel([{ tx: 3, ty: 4 }], 2))).toEqual(keys(box(5, 7, 7, 9)));
  });

  it('floors rather than truncates toward zero for negative tiles', () => {
    // Math.floor(-3 * 0.5) is -2, not -1: a tile left of the origin must map to
    // the coarser tile that actually contains it.
    const out = tilesAtLevel([{ tx: -3, ty: -3 }], 0.5);
    expect(keys(out)).toEqual(keys(box(-3, -1, -3, -1)));
  });

  it('de-duplicates where neighbourhoods overlap', () => {
    const out = tilesAtLevel(box(0, 1, 0, 1), 0.5);
    // All four source tiles map to (0,0), so the union is one 3x3 block.
    expect(out).toHaveLength(9);
    expect(new Set(out.map(key)).size).toBe(9);
  });

  it('returns nothing for an empty source set', () => {
    expect(tilesAtLevel([], 2)).toEqual([]);
  });
});

describe('smoothPanDirection', () => {
  it('eases toward the unit direction of travel', () => {
    const next = smoothPanDirection({ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 10, y: 0 }, 0.4);
    expect(next.x).toBeCloseTo(0.4);
    expect(next.y).toBeCloseTo(0);
  });

  it('normalizes, so a fast and a slow pan agree on direction', () => {
    const slow = smoothPanDirection(STILL, { x: 0, y: 0 }, { x: 0.01, y: 0 }, 0.4);
    const fast = smoothPanDirection(STILL, { x: 0, y: 0 }, { x: 1000, y: 0 }, 0.4);
    expect(slow.x).toBeCloseTo(fast.x);
  });

  it('converges toward the direction under sustained motion', () => {
    let dir = { x: 0, y: 0 };
    for (let i = 0; i < 20; i++) {
      dir = smoothPanDirection(dir, { x: i, y: 0 }, { x: i + 1, y: 0 }, 0.4);
    }
    expect(dir.x).toBeCloseTo(1, 3);
    // ...and crosses the bias threshold, so the ring actually leads the pan.
    expect(dir.x).toBeGreaterThan(PAN_DIR_BIAS_THRESHOLD);
  });

  it('decays toward zero when the view holds still', () => {
    let dir = { x: 1, y: 1 };
    for (let i = 0; i < 20; i++) {
      dir = smoothPanDirection(dir, { x: 5, y: 5 }, { x: 5, y: 5 }, 0.4);
    }
    expect(dir.x).toBeCloseTo(0, 3);
    expect(dir.y).toBeCloseTo(0, 3);
    expect(Math.abs(dir.x)).toBeLessThan(PAN_DIR_BIAS_THRESHOLD);
  });

  it('does not let one jittery frame flip an established bias', () => {
    const established = { x: 1, y: 0 };
    const jittered = smoothPanDirection(established, { x: 5, y: 0 }, { x: 4, y: 0 }, 0.4);
    // One reversed frame pulls the estimate back toward neutral — at most the
    // ring goes symmetric for a frame — but it must never swing far enough to
    // start leading the *opposite* way off a single sample.
    expect(jittered.x).toBeLessThan(established.x);
    expect(jittered.x).toBeGreaterThan(-PAN_DIR_BIAS_THRESHOLD);
  });

  it('needs several consistent frames to reverse an established bias', () => {
    let dir = { x: 1, y: 0 };
    dir = smoothPanDirection(dir, { x: 5, y: 0 }, { x: 4, y: 0 }, 0.4);
    expect(dir.x).toBeGreaterThan(-PAN_DIR_BIAS_THRESHOLD);
    for (let i = 0; i < 5; i++) {
      dir = smoothPanDirection(dir, { x: 5 - i, y: 0 }, { x: 4 - i, y: 0 }, 0.4);
    }
    // A sustained reversal does eventually lead the other way.
    expect(dir.x).toBeLessThan(-PAN_DIR_BIAS_THRESHOLD);
  });

  it('treats a sub-epsilon move as stationary rather than amplifying noise', () => {
    const next = smoothPanDirection({ x: 0.5, y: 0 }, { x: 0, y: 0 }, { x: 1e-9, y: 0 }, 0.4);
    // Decays (toward zero), never jumps to a full unit vector.
    expect(next.x).toBeCloseTo(0.3);
  });

  it('is pure — it does not mutate the previous direction', () => {
    const previous = { x: 0.2, y: 0.2 };
    smoothPanDirection(previous, { x: 0, y: 0 }, { x: 1, y: 1 }, 0.4);
    expect(previous).toEqual({ x: 0.2, y: 0.2 });
  });
});
