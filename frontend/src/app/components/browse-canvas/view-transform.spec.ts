import { describe, expect, it } from 'vitest';

import {
  clampAxis,
  clampedTransform,
  computeFitZoom,
  computeMaxZoom,
  getVisibleBounds,
  levelForEffZoom,
  panLimits,
  projToScreen,
  rubber,
  rubberAxis,
  screenToProj,
  softCeilZoom,
  softClampPan,
  softFloorZoom,
  type ViewGeomContext,
} from './view-transform';
import type { ProjectionMeta, ViewTransform } from '../../models/projection.models';

/**
 * Unit coverage for the framework-free pan/zoom/clamp/rubber-band geometry
 * extracted from `browse-canvas.component.ts` (see the plan item in
 * `docs/plans/code-structure-review.md` Theme B). These are pure functions —
 * no Angular, no canvas — so a projection's bounds and a viewport size are
 * enough to exercise every invariant the component's JSDoc calls out.
 */
describe('view-transform', () => {
  function meta(partial: Partial<ProjectionMeta> = {}): ProjectionMeta {
    return {
      projection_id: 'p1',
      bounds: [0, 0, 100, 100],
      base_radius: 10,
      tile_span: 8,
      point_count: 4,
      levels: [
        { level: 0, radius: 10, n_cells: 4 },
        { level: 1, radius: 5, n_cells: 16 },
        { level: 2, radius: 2.5, n_cells: 64 },
      ],
      ...partial,
    };
  }

  function ctx(partial: Partial<ViewGeomContext> = {}): ViewGeomContext {
    return { meta: meta(), width: 800, height: 600, targetRadius: 28, ...partial };
  }

  function t(partial: Partial<ViewTransform> = {}): ViewTransform {
    return { centerX: 50, centerY: 50, zoom: 1, ...partial };
  }

  describe('computeFitZoom', () => {
    it('returns the current zoom unchanged when there is no meta', () => {
      expect(computeFitZoom({ meta: null, width: 800, height: 600, targetRadius: 28 }, 3)).toBe(3);
    });

    it('picks a zoom that fits the padded content in the smaller viewport dimension', () => {
      const c = ctx({ width: 200, height: 1000 });
      const zoom = computeFitZoom(c, 1);
      // The width is the binding constraint (200 vs a 100-wide dataset), so the
      // fit zoom should land near width / dataW, discounted a bit for padding
      // and the edge-bin circumradius margin.
      expect(zoom).toBeGreaterThan(0);
      expect(zoom).toBeLessThan(2);
    });
  });

  describe('computeMaxZoom', () => {
    it('never drops below the fit-zoom floor, even for a tiny projection', () => {
      const c = ctx({ meta: meta({ bounds: [0, 0, 1, 1] }), targetRadius: 1000 });
      const minZoom = computeFitZoom(c, 1);
      const maxZoom = computeMaxZoom(c, 1);
      expect(maxZoom).toBeGreaterThanOrEqual(minZoom);
    });

    it('scales with targetRadius so bigger thumbnails lift the ceiling', () => {
      const small = computeMaxZoom(ctx({ targetRadius: 20 }), 1);
      const big = computeMaxZoom(ctx({ targetRadius: 40 }), 1);
      expect(big).toBeGreaterThan(small);
    });
  });

  describe('levelForEffZoom', () => {
    it('floors at level 0 and caps at the finest level', () => {
      const c = ctx();
      expect(levelForEffZoom(c, 0.0001)).toBe(0);
      expect(levelForEffZoom(c, 1e9)).toBe(2);
    });

    it('returns 0 when there is no meta or no levels', () => {
      expect(levelForEffZoom({ meta: null, width: 800, height: 600, targetRadius: 28 }, 5)).toBe(0);
      expect(levelForEffZoom(ctx({ meta: meta({ levels: [] }) }), 5)).toBe(0);
    });
  });

  describe('clampAxis', () => {
    it('holds the value inside [lo, hi] at every zoom, never pinning to the midpoint', () => {
      // Even when the range is degenerate (viewport far wider than content), the
      // clamp stays [lo, hi] rather than collapsing to (lo+hi)/2 — see the
      // extensive JSDoc on the original `clampAxis` for why.
      expect(clampAxis(5, 0, 10)).toBe(5);
      expect(clampAxis(-5, 0, 10)).toBe(0);
      expect(clampAxis(15, 0, 10)).toBe(10);
    });
  });

  describe('panLimits', () => {
    it('returns null when there is no data to frame', () => {
      expect(panLimits({ meta: null, width: 800, height: 600, targetRadius: 28 }, 1)).toBeNull();
      expect(panLimits(ctx({ meta: meta({ point_count: 0 }) }), 1)).toBeNull();
    });

    it('grows the content rectangle by the edge bins circumradius', () => {
      const c = ctx();
      const lim = panLimits(c, 1)!;
      const level = levelForEffZoom(c, 1);
      const r = c.meta!.base_radius / Math.pow(2, level);
      expect(lim.loX).toBeCloseTo(0 - r);
      expect(lim.hiX).toBeCloseTo(100 + r);
    });
  });

  describe('clampedTransform', () => {
    it('is idempotent once inside the bounds', () => {
      const c = ctx();
      const once = clampedTransform(t({ zoom: 2 }), c);
      const twice = clampedTransform(once, c);
      expect(twice).toEqual(once);
    });

    it('floors zoom at the whole-projection fit', () => {
      const c = ctx();
      const minZoom = computeFitZoom(c, 1);
      const clamped = clampedTransform(t({ zoom: minZoom / 100 }), c);
      expect(clamped.zoom).toBeCloseTo(minZoom);
    });

    it('caps zoom at the finest-level ceiling', () => {
      const c = ctx();
      const maxZoom = computeMaxZoom(c, 1);
      const clamped = clampedTransform(t({ zoom: maxZoom * 100 }), c);
      expect(clamped.zoom).toBeCloseTo(maxZoom);
    });

    it('does not mutate the input transform', () => {
      const input = t({ zoom: 0.0001 });
      const snapshot = { ...input };
      clampedTransform(input, ctx());
      expect(input).toEqual(snapshot);
    });
  });

  describe('rubber', () => {
    it('returns 0 for zero overshoot budget', () => {
      expect(rubber(5, 0)).toBe(0);
    });

    it('is signed and asymptotes toward maxOver without exceeding it', () => {
      expect(rubber(1e9, 10)).toBeLessThan(10);
      expect(rubber(1e9, 10)).toBeGreaterThan(9);
      expect(rubber(-1e9, 10)).toBeGreaterThan(-10);
      expect(rubber(-1e9, 10)).toBeLessThan(-9);
    });

    it('tracks the input near zero at the configured give', () => {
      // f'(0) = c (RUBBER_GIVE = 0.5), so a small pull moves about half as far.
      expect(rubber(0.001, 100)).toBeCloseTo(0.0005, 4);
    });
  });

  describe('rubberAxis', () => {
    it('is the identity inside [lo, hi]', () => {
      expect(rubberAxis(5, 0, 10, 100)).toBe(5);
    });

    it('gives elastically past either edge, bounded by the viewport extent', () => {
      const over = rubberAxis(1e9, 0, 10, 100);
      expect(over).toBeGreaterThan(10);
      expect(over).toBeLessThan(10 + 0.18 * 100);
      const under = rubberAxis(-1e9, 0, 10, 100);
      expect(under).toBeLessThan(0);
      expect(under).toBeGreaterThan(-0.18 * 100);
    });
  });

  describe('softClampPan', () => {
    it('leaves the centre untouched when there is no data', () => {
      const p = softClampPan(t(), { meta: null, width: 800, height: 600, targetRadius: 28 }, 1);
      expect(p).toEqual({ centerX: 50, centerY: 50 });
    });

    it('lets the centre drift past the hard clamp instead of stopping dead', () => {
      const c = ctx();
      const hard = clampedTransform(t({ centerX: 1e6, zoom: 1 }), c);
      const soft = softClampPan(t({ centerX: 1e6 }), c, 1);
      expect(soft.centerX).toBeGreaterThan(hard.centerX);
    });
  });

  describe('softFloorZoom / softCeilZoom', () => {
    it('passes zoom through unchanged when inside the hard bounds', () => {
      const c = ctx();
      const minZoom = computeFitZoom(c, 1);
      const maxZoom = computeMaxZoom(c, 1);
      const mid = (minZoom + maxZoom) / 2;
      expect(softFloorZoom(mid, c, 1)).toBe(mid);
      expect(softCeilZoom(mid, c, 1)).toBe(mid);
    });

    it('resists but does not block a zoom past either edge', () => {
      const c = ctx();
      const minZoom = computeFitZoom(c, 1);
      const maxZoom = computeMaxZoom(c, 1);
      const floored = softFloorZoom(minZoom / 1000, c, 1);
      expect(floored).toBeLessThan(minZoom);
      expect(floored).toBeGreaterThan(minZoom / 1000);
      const ceiled = softCeilZoom(maxZoom * 1000, c, 1);
      expect(ceiled).toBeGreaterThan(maxZoom);
      expect(ceiled).toBeLessThan(maxZoom * 1000);
    });
  });

  describe('projToScreen / screenToProj', () => {
    it('round-trips a point through screen space', () => {
      const transform = t({ centerX: 12, centerY: -7, zoom: 2.5 });
      const [sx, sy] = projToScreen(3, 4, transform, 800, 600);
      const [px, py] = screenToProj(sx, sy, transform, 800, 600);
      expect(px).toBeCloseTo(3);
      expect(py).toBeCloseTo(4);
    });

    it('maps the transform centre to the viewport centre', () => {
      const transform = t({ centerX: 12, centerY: -7, zoom: 2.5 });
      const [sx, sy] = projToScreen(12, -7, transform, 800, 600);
      expect(sx).toBeCloseTo(400);
      expect(sy).toBeCloseTo(300);
    });
  });

  describe('getVisibleBounds', () => {
    it('returns the min/max-ordered rectangle visible at the given transform', () => {
      const transform = t({ centerX: 0, centerY: 0, zoom: 1 });
      const [xmin, ymin, xmax, ymax] = getVisibleBounds(transform, 800, 600);
      expect(xmin).toBeCloseTo(-400);
      expect(xmax).toBeCloseTo(400);
      expect(ymin).toBeCloseTo(-300);
      expect(ymax).toBeCloseTo(300);
    });
  });
});
