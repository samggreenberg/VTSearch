import { describe, expect, it } from 'vitest';
import {
  layoutSigns,
  SIGN_APPEAR_DELTA,
  SIGN_BASE_FONT_PX,
  SIGN_EXPIRE_DELTA,
  SIGN_FADE_IN_SPAN,
  SIGN_FADE_OUT_SPAN,
  SIGN_SHADOW_MAX_SCALE,
  SIGN_SHADOW_MIN_SCALE,
  signAppearance,
  signShadow,
  viewLevelForZoom,
  type SignViewContext,
} from './sign-layout';
import type { RegionLabelPayload, ViewTransform } from '../../models/projection.models';

/** Character-count text measurer: deterministic, canvas-free. */
const measure = (text: string, fontPx: number): number => text.length * fontPx * 0.6;

const centeredView = (viewLevel: number, zoom = 1): SignViewContext => ({
  transform: { centerX: 0, centerY: 0, zoom } as ViewTransform,
  width: 800,
  height: 600,
  viewLevel,
});

const label = (over: Partial<RegionLabelPayload> = {}): RegionLabelPayload => ({
  level: 0,
  x: 0,
  y: 0,
  text: 'Birdsong',
  score: 1,
  source: 'test',
  ...over,
});

describe('viewLevelForZoom', () => {
  it('is the unrounded form of the LOD picker: level 0 when a base-radius bin renders at target size', () => {
    // base_radius * zoom == targetRadius → log2(1) == 0.
    expect(viewLevelForZoom(28, 1, 28)).toBe(0);
  });

  it('gains exactly one level per 2x of zoom', () => {
    const base = viewLevelForZoom(10, 1, 28);
    expect(viewLevelForZoom(10, 2, 28)).toBeCloseTo(base + 1, 10);
    expect(viewLevelForZoom(10, 8, 28)).toBeCloseTo(base + 3, 10);
  });

  it('drops as the target radius (thumbnail size) grows, matching level selection', () => {
    expect(viewLevelForZoom(28, 1, 56)).toBe(-1);
  });
});

describe('signAppearance', () => {
  it('is invisible far below the user zoom (sign much finer than the view)', () => {
    expect(signAppearance(SIGN_APPEAR_DELTA)).toBeNull();
    expect(signAppearance(-5)).toBeNull();
  });

  it('is gone far above the user zoom (view zoomed far past the sign)', () => {
    expect(signAppearance(SIGN_EXPIRE_DELTA)).toBeNull();
    expect(signAppearance(10)).toBeNull();
  });

  it('fades in as small text just below the user zoom', () => {
    const midFade = signAppearance(SIGN_APPEAR_DELTA + SIGN_FADE_IN_SPAN / 2)!;
    expect(midFade.alpha).toBeCloseTo(0.5, 10);
    expect(midFade.scale).toBeLessThan(1);
    const justBelow = signAppearance(-1)!;
    expect(justBelow.alpha).toBe(1);
    expect(justBelow.scale).toBeLessThan(1);
  });

  it('is full-size and opaque at the user zoom', () => {
    const at = signAppearance(0)!;
    expect(at.alpha).toBe(1);
    expect(at.scale).toBeCloseTo(0.9, 1);
    expect(signAppearance(0.5)!.scale).toBe(1);
  });

  it('enlarges again above the user zoom before fading out', () => {
    const above = signAppearance(1)!;
    expect(above.scale).toBeGreaterThan(1);
    expect(above.alpha).toBe(1);
    const fading = signAppearance(SIGN_EXPIRE_DELTA - SIGN_FADE_OUT_SPAN / 2)!;
    expect(fading.alpha).toBeCloseTo(0.5, 10);
    expect(fading.scale).toBeGreaterThan(above.scale);
  });

  it('grows monotonically in delta across the whole visible band', () => {
    let prev = 0;
    for (let d = SIGN_APPEAR_DELTA + 0.01; d < SIGN_EXPIRE_DELTA; d += 0.05) {
      const a = signAppearance(d)!;
      expect(a.scale).toBeGreaterThanOrEqual(prev);
      prev = a.scale;
    }
  });

  it('rejects non-finite deltas', () => {
    expect(signAppearance(Number.NaN)).toBeNull();
    expect(signAppearance(Infinity)).toBeNull();
  });
});

describe('signShadow', () => {
  it('is flat (no shadow) at and below the minimum scale — the smallest signs', () => {
    expect(signShadow(SIGN_SHADOW_MIN_SCALE, 13)).toBeNull();
    expect(signShadow(SIGN_SHADOW_MIN_SCALE - 0.05, 13)).toBeNull();
    expect(signShadow(0.65, 13)).toBeNull();
  });

  it('casts a shadow for medium signs, deeper for larger ones', () => {
    const medium = signShadow(1.0, SIGN_BASE_FONT_PX)!;
    const large = signShadow(1.25, SIGN_BASE_FONT_PX * 1.25)!;
    expect(medium).not.toBeNull();
    // Every dimension of the depth cue grows as the sign gets bigger.
    expect(large.blur).toBeGreaterThan(medium.blur);
    expect(large.offsetY).toBeGreaterThan(medium.offsetY);
    expect(large.alpha).toBeGreaterThan(medium.alpha);
  });

  it('is deepest at the maximum scale — about to float past the viewer', () => {
    const fontPx = SIGN_BASE_FONT_PX * SIGN_SHADOW_MAX_SCALE;
    const biggest = signShadow(SIGN_SHADOW_MAX_SCALE, fontPx)!;
    // lift saturates at 1, so blur/offset are the full per-em fraction of fontPx
    // and the shadow is at its configured peak opacity.
    expect(biggest.blur).toBeCloseTo(fontPx * 1.2, 5);
    expect(biggest.offsetY).toBeCloseTo(fontPx * 0.55, 5);
    expect(biggest.alpha).toBeCloseTo(0.55, 5);
  });

  it('clamps beyond the maximum scale rather than over-lifting', () => {
    const fontPx = SIGN_BASE_FONT_PX * SIGN_SHADOW_MAX_SCALE;
    const atMax = signShadow(SIGN_SHADOW_MAX_SCALE, fontPx)!;
    const beyond = signShadow(SIGN_SHADOW_MAX_SCALE + 1, fontPx)!;
    expect(beyond.blur).toBeCloseTo(atMax.blur, 10);
    expect(beyond.alpha).toBeCloseTo(atMax.alpha, 10);
  });

  it('scales the shadow with font size so it stays proportional', () => {
    const small = signShadow(1.2, 10)!;
    const big = signShadow(1.2, 20)!;
    // Same lift, double the font → double the blur/offset.
    expect(big.blur).toBeCloseTo(small.blur * 2, 10);
    expect(big.offsetY).toBeCloseTo(small.offsetY * 2, 10);
    expect(big.alpha).toBeCloseTo(small.alpha, 10);
  });
});

describe('layoutSigns', () => {
  it('places a visible sign at its projected screen position', () => {
    const placed = layoutSigns([label({ x: 0, y: 0 })], centeredView(0), measure);
    expect(placed).toHaveLength(1);
    expect(placed[0].sx).toBe(400);
    expect(placed[0].sy).toBe(300);
    expect(placed[0].fontPx).toBeCloseTo(SIGN_BASE_FONT_PX * 0.9, 5);
    expect(placed[0].scale).toBeCloseTo(0.9, 5);
    expect(placed[0].alpha).toBe(1);
  });

  it('drops signs whose level is outside the visibility band', () => {
    // View at continent level: a state-level sign (3 levels finer) is invisible;
    // zoomed to state level, the continent sign has expired.
    expect(layoutSigns([label({ level: 3 })], centeredView(0), measure)).toHaveLength(0);
    expect(layoutSigns([label({ level: 0 })], centeredView(3), measure)).toHaveLength(0);
  });

  it('shows coarse and fine signs together only in their overlapping band', () => {
    const labels = [label({ level: 0, text: 'Europe' }), label({ level: 1, x: 100, text: 'France' })];
    // At view level 0.75, the level-0 sign is enlarged and the level-1 sign is
    // still small — both visible, the coarser one bigger.
    const placed = layoutSigns(labels, centeredView(0.75), measure);
    expect(placed).toHaveLength(2);
    const europe = placed.find((p) => p.label.text === 'Europe')!;
    const france = placed.find((p) => p.label.text === 'France')!;
    expect(europe.fontPx).toBeGreaterThan(france.fontPx);
  });

  it('culls signs entirely off screen', () => {
    const placed = layoutSigns(
      [label({ x: 10_000, y: 0 })],
      centeredView(0),
      measure,
    );
    expect(placed).toHaveLength(0);
  });

  it('de-clutters overlapping pills, keeping the bigger sign', () => {
    // Same anchor: the coarser (bigger at this zoom) sign must win.
    const labels = [
      label({ level: 1, text: 'small-sign' }),
      label({ level: 0.5, text: 'big-sign' }),
    ];
    const placed = layoutSigns(labels, centeredView(0.5), measure);
    expect(placed).toHaveLength(1);
    expect(placed[0].label.text).toBe('big-sign');
  });

  it('breaks priority ties by score', () => {
    const labels = [
      label({ text: 'loser-x', score: 0.2 }),
      label({ text: 'winnerx', score: 0.9 }),
    ];
    const placed = layoutSigns(labels, centeredView(0), measure);
    expect(placed).toHaveLength(1);
    expect(placed[0].label.text).toBe('winnerx');
  });

  it('keeps non-overlapping signs of the same level', () => {
    const labels = [label({ x: -300, text: 'west' }), label({ x: 300, text: 'east' })];
    // zoom 1 → 1 projection unit = 1 px; 600 px apart, far wider than a pill.
    const placed = layoutSigns(labels, centeredView(0), measure);
    expect(placed).toHaveLength(2);
  });

  it('skips empty-text labels', () => {
    expect(layoutSigns([label({ text: '' })], centeredView(0), measure)).toHaveLength(0);
  });
});
