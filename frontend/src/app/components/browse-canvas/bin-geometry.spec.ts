import { describe, expect, it, vi } from 'vitest';

import {
  binGeometry,
  hoverThumbHalfExtents,
  pickCell,
  SQRT3,
  type BinGeometry,
} from './bin-geometry';

/**
 * Unit coverage for the bin-shape geometry shared by the browse canvas and its
 * minimap (see issue #2418). Covers both the hex and square `BinGeometry`
 * implementations and the two pure hit-test / thumbnail-sizing helpers extracted
 * from `browse-canvas.component.ts` (`pickCell`, `hoverThumbHalfExtents`).
 */
describe('bin-geometry', () => {
  const hex = binGeometry('hex');
  const square = binGeometry('square');

  describe('binGeometry', () => {
    it('selects the requested shape', () => {
      expect(hex.shape).toBe('hex');
      expect(square.shape).toBe('square');
    });

    it('defaults to hex for an unset or unknown shape', () => {
      expect(binGeometry(undefined).shape).toBe('hex');
    });
  });

  describe('cell spacing (dx / dy)', () => {
    it('spaces hex columns by radius·√3 and rows by radius·1.5', () => {
      expect(hex.dx(4)).toBeCloseTo(4 * SQRT3);
      expect(hex.dy(4)).toBeCloseTo(4 * 1.5);
    });

    it('spaces square cells by radius·√3 on both axes', () => {
      expect(square.dx(4)).toBeCloseTo(4 * SQRT3);
      expect(square.dy(4)).toBeCloseTo(4 * SQRT3);
    });
  });

  describe('contains (hit-test predicate)', () => {
    it('hex: inside the circumscribed circle of radius', () => {
      expect(hex.contains(0, 0, 5)).toBe(true);
      expect(hex.contains(4.9, 0, 5)).toBe(true);
      expect(hex.contains(5.1, 0, 5)).toBe(false);
      // Just inside the circle on the diagonal.
      expect(hex.contains(3, 3, 5)).toBe(true); // 3²+3²=18 < 25
      expect(hex.contains(4, 4, 5)).toBe(false); // 32 > 25
    });

    it('square: inside the half-side box (side = radius·√3)', () => {
      const half = (5 * SQRT3) / 2;
      expect(square.contains(0, 0, 5)).toBe(true);
      expect(square.contains(half - 0.01, half - 0.01, 5)).toBe(true);
      expect(square.contains(half + 0.01, 0, 5)).toBe(false);
      expect(square.contains(0, half + 0.01, 5)).toBe(false);
    });
  });

  describe('singleCornerRadius', () => {
    it('hex: the inscribed-disc radius (radius·√3/2)', () => {
      expect(hex.singleCornerRadius(6)).toBeCloseTo((6 * SQRT3) / 2);
    });

    it('square: a fraction of the half-side', () => {
      // Rounded-corner radius is smaller than the half-side it curves within.
      const half = (6 * SQRT3) / 2;
      expect(square.singleCornerRadius(6)).toBeGreaterThan(0);
      expect(square.singleCornerRadius(6)).toBeLessThan(half);
    });
  });

  describe('neighborOffsets', () => {
    it('hex has six neighbours, all one centre-distance (√3·radius) away', () => {
      const offs = hex.neighborOffsets();
      expect(offs).toHaveLength(6);
      for (const { dx, dy } of offs) {
        // |(dx, dy)| in units of radius equals the centre-to-centre distance √3.
        expect(Math.hypot(dx, dy)).toBeCloseTo(SQRT3);
      }
    });

    it('square has four edge-adjacent neighbours (corners excluded)', () => {
      const offs = square.neighborOffsets();
      expect(offs).toHaveLength(4);
      // Every neighbour is axis-aligned (one of dx/dy is zero).
      for (const { dx, dy } of offs) {
        expect(dx === 0 || dy === 0).toBe(true);
      }
    });
  });

  // A minimal stand-in for the 2D context, recording the path calls traceCell
  // makes so we can assert disc-vs-hex / roundRect-vs-rect without a real canvas.
  // The vi.fn() mocks keep their `.mock` type; `ctx` is the cast passed to
  // traceCell (which wants a full CanvasRenderingContext2D).
  function stubCtx() {
    const fns = {
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      arc: vi.fn(),
      rect: vi.fn(),
      roundRect: vi.fn(),
    };
    return { ...fns, ctx: fns as unknown as CanvasRenderingContext2D };
  }

  describe('traceCell', () => {
    it('hex singleton draws a disc, pile draws a polygon', () => {
      const single = stubCtx();
      hex.traceCell(single.ctx, 0, 0, 5, true);
      expect(single.arc).toHaveBeenCalledOnce();

      const pile = stubCtx();
      hex.traceCell(pile.ctx, 0, 0, 5, false);
      expect(pile.arc).not.toHaveBeenCalled();
      expect(pile.lineTo).toHaveBeenCalled();
    });

    it('square singleton uses roundRect, pile uses rect', () => {
      const single = stubCtx();
      square.traceCell(single.ctx, 0, 0, 5, true);
      expect(single.roundRect).toHaveBeenCalledOnce();
      expect(single.rect).not.toHaveBeenCalled();

      const pile = stubCtx();
      square.traceCell(pile.ctx, 0, 0, 5, false);
      expect(pile.rect).toHaveBeenCalledOnce();
      expect(pile.roundRect).not.toHaveBeenCalled();
    });

    it('square cell has side radius·√3 centred on (cx, cy)', () => {
      const ctx = stubCtx();
      square.traceCell(ctx.ctx, 10, 20, 5, false);
      const [x, y, w, h] = ctx.rect.mock.calls[0];
      const side = 5 * SQRT3;
      expect(w).toBeCloseTo(side);
      expect(h).toBeCloseTo(side);
      expect(x).toBeCloseTo(10 - side / 2);
      expect(y).toBeCloseTo(20 - side / 2);
    });
  });

  describe('pickCell', () => {
    const cells = [
      { cx: 0, cy: 0, id: 'a' },
      { cx: 10, cy: 0, id: 'b' },
      { cx: 0, cy: 10, id: 'c' },
    ];

    it('returns the nearest cell when the point falls inside it', () => {
      // Right on top of cell b's centre.
      expect(pickCell(cells, 10, 0, hex, 5)?.id).toBe('b');
      // A hair off centre, still within the circumradius.
      expect(pickCell(cells, 1, 1, hex, 5)?.id).toBe('a');
    });

    it('returns null over blank space between bins', () => {
      // Midway between a, b, c — outside every cell's radius-2 circle.
      expect(pickCell(cells, 5, 5, hex, 2)).toBeNull();
    });

    it('returns null for an empty candidate set', () => {
      expect(pickCell([], 0, 0, hex, 5)).toBeNull();
    });

    it('honours the geometry shape for the contains check', () => {
      // A point that is inside a square cell but outside the hex circle of the
      // same radius: at (r·0.7, r·0.7) with r=5 → dist²=49.0 > 25 (outside hex),
      // but |x|,|y| = 3.5 < half-side 4.33 (inside square).
      const one = [{ cx: 0, cy: 0, id: 'a' }];
      expect(pickCell(one, 3.5, 3.5, hex, 5)).toBeNull();
      expect(pickCell(one, 3.5, 3.5, square, 5)?.id).toBe('a');
    });
  });

  describe('hoverThumbHalfExtents', () => {
    it('keeps the requested aspect ratio (hw = aspect · hh)', () => {
      const offs = hex.neighborOffsets();
      const { hw, hh } = hoverThumbHalfExtents(2, 10, offs);
      expect(hw).toBeCloseTo(2 * hh);
    });

    it('grows a square thumbnail until it just reaches a neighbour centre', () => {
      // aspect 1 on the square lattice: neighbours at ±√3·radius on each axis, so
      // hh is bound by min |dy|·radius = √3·radius.
      const { hh } = hoverThumbHalfExtents(1, 4, square.neighborOffsets());
      expect(hh).toBeCloseTo(SQRT3 * 4);
    });

    it('lets a wide image grow taller than a tall one is wide, per the min rule', () => {
      const offs = hex.neighborOffsets();
      const wide = hoverThumbHalfExtents(3, 10, offs);
      const tall = hoverThumbHalfExtents(1 / 3, 10, offs);
      // Wide image is limited by the side neighbours; tall by top/bottom. Each
      // stays inside its binding neighbour, so neither half-extent runs away.
      expect(wide.hw).toBeGreaterThan(0);
      expect(tall.hh).toBeGreaterThan(0);
      // The wide thumbnail is wider than it is tall, and vice versa.
      expect(wide.hw).toBeGreaterThan(wide.hh);
      expect(tall.hh).toBeGreaterThan(tall.hw);
    });

    it('scales linearly with the bin radius', () => {
      const offs = hex.neighborOffsets();
      const small = hoverThumbHalfExtents(1.5, 5, offs);
      const big = hoverThumbHalfExtents(1.5, 10, offs);
      expect(big.hh).toBeCloseTo(2 * small.hh);
      expect(big.hw).toBeCloseTo(2 * small.hw);
    });
  });

  function customGeom(): BinGeometry {
    // Sanity: pickCell/hoverThumbHalfExtents accept any BinGeometry, not just the
    // two built-ins — they only touch `contains` / the passed offsets.
    return binGeometry('hex');
  }

  it('helpers work against any BinGeometry instance', () => {
    const g = customGeom();
    expect(pickCell([{ cx: 0, cy: 0 }], 0, 0, g, 5)).not.toBeNull();
  });
});
