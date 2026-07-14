import { describe, expect, it, vi } from 'vitest';

import {
  BROWSE_COLORMAP_IDS,
  DEFAULT_THUMBNAIL_BORDER,
  densityColor,
  gradientStops,
  HEX_ANGLES,
  HEX_INRADIUS_RATIO,
  MAX_THUMBNAIL_BORDER,
  resolveColormap,
  rgbString,
  SQRT3,
  traceCellPath,
  traceHexPath,
  usesThumbnails,
  type CanvasTheme,
} from './hex-render.util';

/**
 * Unit coverage for the framework-free hex-grid rendering primitives shared by
 * the browse canvas and its minimap (see issue #2418). These are pure functions
 * over colours and geometry — the only "canvas" here is a tiny stub 2D context
 * that records the path commands `traceHexPath` / `traceCellPath` emit.
 */
describe('hex-render.util', () => {
  describe('usesThumbnails', () => {
    it('is true for the four thumbnail-backed media types', () => {
      for (const t of ['image', 'video', 'document', 'audio']) {
        expect(usesThumbnails(t)).toBe(true);
      }
    });

    it('is false for flat-density types like text', () => {
      expect(usesThumbnails('text')).toBe(false);
      expect(usesThumbnails('anything-else')).toBe(false);
      expect(usesThumbnails('')).toBe(false);
    });
  });

  describe('resolveColormap', () => {
    it("picks Ocean in light mode and Heat in dark/high-viz for 'auto'", () => {
      // Ocean darkens with density (low end lighter than high); Heat brightens.
      const light = resolveColormap('auto', 'light');
      const dark = resolveColormap('auto', 'dark');
      const highviz = resolveColormap('auto', 'highviz');
      expect(dark).toEqual(highviz);
      // Ocean's ramp gets darker (sum falls); Heat's gets brighter (sum rises).
      const sum = (c: readonly [number, number, number]) => c[0] + c[1] + c[2];
      expect(sum(light.ramp[light.ramp.length - 1])).toBeLessThan(sum(light.ramp[0]));
      expect(sum(dark.ramp[dark.ramp.length - 1])).toBeGreaterThan(sum(dark.ramp[0]));
    });

    it('returns Heat and Ocean regardless of theme for their explicit ids', () => {
      expect(resolveColormap('heat', 'light')).toEqual(resolveColormap('heat', 'dark'));
      expect(resolveColormap('ocean', 'light')).toEqual(resolveColormap('ocean', 'dark'));
    });

    it('flips grayscale direction with the theme', () => {
      const grayLight = resolveColormap('gray', 'light');
      const grayDark = resolveColormap('gray', 'dark');
      const sum = (c: readonly [number, number, number]) => c[0] + c[1] + c[2];
      // Light: darkens as density grows (moves away from the light background).
      expect(sum(grayLight.ramp[grayLight.ramp.length - 1])).toBeLessThan(sum(grayLight.ramp[0]));
      // Dark: lightens as density grows.
      expect(sum(grayDark.ramp[grayDark.ramp.length - 1])).toBeGreaterThan(sum(grayDark.ramp[0]));
    });

    it('exposes every selectable id, auto first', () => {
      expect(BROWSE_COLORMAP_IDS[0]).toBe('auto');
      // Every id resolves to a non-empty ramp under every theme.
      for (const id of BROWSE_COLORMAP_IDS) {
        for (const theme of ['light', 'dark', 'highviz'] as CanvasTheme[]) {
          expect(resolveColormap(id, theme).ramp.length).toBeGreaterThan(0);
        }
      }
    });
  });

  describe('rgbString', () => {
    it('formats an [r, g, b] triple as a CSS rgb(...) string', () => {
      expect(rgbString([1, 2, 3])).toBe('rgb(1,2,3)');
    });
  });

  describe('densityColor', () => {
    it('returns the ramp endpoints at t=0 and t=1', () => {
      const ramp: [number, number, number][] = [
        [0, 0, 0],
        [100, 100, 100],
        [200, 200, 200],
      ];
      expect(densityColor(0, ramp)).toBe('rgb(0,0,0)');
      expect(densityColor(1, ramp)).toBe('rgb(200,200,200)');
    });

    it('interpolates linearly between adjacent stops', () => {
      const ramp: [number, number, number][] = [
        [0, 0, 0],
        [100, 0, 0],
      ];
      // Halfway along a two-stop ramp is the midpoint colour.
      expect(densityColor(0.5, ramp)).toBe('rgb(50,0,0)');
    });

    it('lands a quarter of the way into the first segment of a longer ramp', () => {
      const ramp: [number, number, number][] = [
        [0, 0, 0],
        [40, 80, 120],
        [200, 200, 200],
      ];
      // n = 2 segments; t=0.25 → idx 0.5 → halfway into segment 0.
      expect(densityColor(0.25, ramp)).toBe('rgb(20,40,60)');
    });
  });

  describe('gradientStops', () => {
    it('spreads stops evenly from 0% to 100%', () => {
      const stops = gradientStops([
        [0, 0, 0],
        [128, 128, 128],
        [255, 255, 255],
      ]);
      expect(stops).toBe('rgb(0,0,0) 0%, rgb(128,128,128) 50%, rgb(255,255,255) 100%');
    });

    it('does not divide by zero for a single-colour ramp', () => {
      expect(gradientStops([[10, 20, 30]])).toBe('rgb(10,20,30) 0%');
    });
  });

  describe('HEX_ANGLES', () => {
    it('is a pointy-top hexagon: six vertices, first straight up', () => {
      expect(HEX_ANGLES).toHaveLength(6);
      // First vertex at -90° (straight up) for a pointy-top hex.
      expect(HEX_ANGLES[0]).toBeCloseTo(-Math.PI / 2);
      // Vertices are spaced 60° apart.
      expect(HEX_ANGLES[1] - HEX_ANGLES[0]).toBeCloseTo(Math.PI / 3);
    });
  });

  describe('HEX_INRADIUS_RATIO', () => {
    it('is √3/2 — the largest disc that fits inside the hex', () => {
      expect(HEX_INRADIUS_RATIO).toBeCloseTo(SQRT3 / 2);
      expect(HEX_INRADIUS_RATIO).toBeLessThan(1);
    });
  });

  describe('border constants', () => {
    it('has a sane default below the max clamp', () => {
      expect(DEFAULT_THUMBNAIL_BORDER).toBeGreaterThan(0);
      expect(DEFAULT_THUMBNAIL_BORDER).toBeLessThanOrEqual(MAX_THUMBNAIL_BORDER);
    });
  });

  // A minimal stand-in for the 2D context, recording just the path calls the
  // trace helpers make so we can assert the shape without a real canvas.
  function stubCtx() {
    return {
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      arc: vi.fn(),
    } as unknown as CanvasRenderingContext2D & {
      beginPath: ReturnType<typeof vi.fn>;
      moveTo: ReturnType<typeof vi.fn>;
      lineTo: ReturnType<typeof vi.fn>;
      closePath: ReturnType<typeof vi.fn>;
      arc: ReturnType<typeof vi.fn>;
    };
  }

  describe('traceHexPath', () => {
    it('traces six vertices centred on (cx, cy) at the circumradius', () => {
      const ctx = stubCtx();
      traceHexPath(ctx, 10, 20, 5);
      expect(ctx.beginPath).toHaveBeenCalledOnce();
      expect(ctx.closePath).toHaveBeenCalledOnce();
      // One moveTo (first vertex) + five lineTo (remaining vertices) = 6 total.
      expect(ctx.moveTo).toHaveBeenCalledOnce();
      expect(ctx.lineTo).toHaveBeenCalledTimes(5);
      // First vertex sits straight up from the centre at the circumradius.
      const [x, y] = ctx.moveTo.mock.calls[0];
      expect(x).toBeCloseTo(10);
      expect(y).toBeCloseTo(20 - 5);
    });
  });

  describe('traceCellPath', () => {
    it('draws the inscribed disc for a singleton', () => {
      const ctx = stubCtx();
      traceCellPath(ctx, 10, 20, 5, true);
      expect(ctx.arc).toHaveBeenCalledOnce();
      const [cx, cy, r] = ctx.arc.mock.calls[0];
      expect(cx).toBe(10);
      expect(cy).toBe(20);
      expect(r).toBeCloseTo(5 * HEX_INRADIUS_RATIO);
      // No polygon path for a singleton disc.
      expect(ctx.moveTo).not.toHaveBeenCalled();
    });

    it('draws the full hexagon for a pile (non-single)', () => {
      const ctx = stubCtx();
      traceCellPath(ctx, 10, 20, 5, false);
      expect(ctx.arc).not.toHaveBeenCalled();
      expect(ctx.moveTo).toHaveBeenCalledOnce();
      expect(ctx.lineTo).toHaveBeenCalledTimes(5);
    });
  });
});
