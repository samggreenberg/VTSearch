import { describe, expect, it } from 'vitest';
import {
  approximateLevelKeys,
  GROUND_TRUTH_SIGN_SOURCE,
  layoutSigns,
  SIGN_APPEAR_DELTA,
  SIGN_APPROXIMATE_ALPHA,
  SIGN_APPROXIMATE_PREFIX,
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

describe('signAppearance — terminal signs (no hand-off neighbour)', () => {
  it('a root (no coarser parent) stays opaque below its coarse edge instead of fading/culling', () => {
    // A normal sign is culled far below its coarse edge...
    expect(signAppearance(-5)).toBeNull();
    // ...but a root persists at full opacity (nothing coarser names the area).
    const root = signAppearance(-5, false, true)!;
    expect(root).not.toBeNull();
    expect(root.alpha).toBe(1);
    // And inside the fade-in band it's opaque where a normal sign is half-faded.
    const d = SIGN_APPEAR_DELTA + SIGN_FADE_IN_SPAN / 2;
    expect(signAppearance(d)!.alpha).toBeCloseTo(0.5, 10);
    expect(signAppearance(d, false, true)!.alpha).toBe(1);
  });

  it('still expires a root at its fine edge — only the coarse-edge fade is skipped', () => {
    expect(signAppearance(10, false, true)).toBeNull();
  });

  it('a leaf (no finer child) stays opaque above its fine edge instead of expiring', () => {
    // A normal sign is gone far above its fine edge...
    expect(signAppearance(10)).toBeNull();
    // ...but a leaf persists at full opacity (nothing finer takes over).
    const leaf = signAppearance(10, true, false)!;
    expect(leaf).not.toBeNull();
    expect(leaf.alpha).toBe(1);
    // And inside the fade-out band it's opaque where a normal sign is half-faded.
    const d = SIGN_EXPIRE_DELTA - SIGN_FADE_OUT_SPAN / 2;
    expect(signAppearance(d)!.alpha).toBeCloseTo(0.5, 10);
    expect(signAppearance(d, true, false)!.alpha).toBe(1);
  });

  it('still hides a leaf below its coarse edge — only the fine-edge fade is skipped', () => {
    expect(signAppearance(-5, true, false)).toBeNull();
  });

  it('an isolated sign (both edges terminal) is visible on both sides of the band', () => {
    expect(signAppearance(-5, false, false)!.alpha).toBe(1);
    expect(signAppearance(10, false, false)!.alpha).toBe(1);
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

  it('never lifts an approximate sign off the map, at any scale', () => {
    // The lift reads as a sign asserting itself; a hedged name should not.
    expect(signShadow(1.2, SIGN_BASE_FONT_PX)).not.toBeNull();
    expect(signShadow(1.2, SIGN_BASE_FONT_PX, true)).toBeNull();
    expect(signShadow(SIGN_SHADOW_MAX_SCALE, 20, true)).toBeNull();
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

  it('keeps a leaf island lettered when zoomed far past it', () => {
    // A normal level-0 sign has expired by view level 3...
    expect(layoutSigns([label({ level: 0 })], centeredView(3), measure)).toHaveLength(0);
    // ...but a leaf (no finer child to hand off to) stays on the map.
    const placed = layoutSigns([label({ level: 0, has_finer: false })], centeredView(3), measure);
    expect(placed).toHaveLength(1);
  });

  it('keeps a root island lettered when zoomed out past it', () => {
    // A normal level-3 sign is invisible at view level 0...
    expect(layoutSigns([label({ level: 3 })], centeredView(0), measure)).toHaveLength(0);
    // ...but a root (no coarser parent naming the area) stays on the map.
    const placed = layoutSigns([label({ level: 3, has_coarser: false })], centeredView(0), measure);
    expect(placed).toHaveLength(1);
  });
});

/** The server's zoom-band step between adjacent Toponymy layers
 *  (`signpost_build._LEVEL_STEP`); a four-layer tree lands on 5.4/3.6/1.8/0. */
const STEP = 1.8;

/** A four-layer clustered tree, finest (rank 0, `level` 5.4) to coarsest
 *  (rank 3, `level` 0) — the shape the C4 table measured. Anchors are spread
 *  across the 800 px viewport so nothing de-clutters or falls off screen. */
const fourLayers = (): RegionLabelPayload[] =>
  [0, 1, 2, 3].map((rank) =>
    label({
      level: (3 - rank) * STEP,
      x: -300 + rank * 200,
      text: `layer-${rank}`,
      source: 'keyphrase',
    }),
  );

/** A view level showing exactly one crisp band (rank 1) and one hedged band
 *  (rank 2): with bands 1.8 apart and a 4-level-wide visibility window, those
 *  are the only two of the four in view. */
const CRISP_AND_HEDGED_VIEW = 3.0;

describe('approximateLevelKeys', () => {
  it('marks the two coarsest bands of a four-layer tree', () => {
    const keys = approximateLevelKeys(fourLayers());
    // Ranks 0 and 1 (levels 5.4, 3.6) are crisp; ranks 2 and 3 (1.8, 0) hedge.
    expect(keys.has(Math.round(3 * STEP * 1000))).toBe(false);
    expect(keys.has(Math.round(2 * STEP * 1000))).toBe(false);
    expect(keys.has(Math.round(STEP * 1000))).toBe(true);
    expect(keys.has(0)).toBe(true);
  });

  it('ranks from the finest band present, so a three-layer tree hedges only its coarsest', () => {
    // A three-layer tree emits 3.6 / 1.8 / 0 — the same absolute levels as a
    // four-layer tree's ranks 1–3, but only one of them is the C4 table's layer
    // 2. Ranking, not a level threshold, is what gets this right.
    const keys = approximateLevelKeys([
      label({ level: 2 * STEP, text: 'fine' }),
      label({ level: STEP, x: 200, text: 'mid' }),
      label({ level: 0, x: 400, text: 'coarse' }),
    ]);
    expect(keys.has(Math.round(2 * STEP * 1000))).toBe(false);
    expect(keys.has(Math.round(STEP * 1000))).toBe(false);
    expect(keys.has(0)).toBe(true);
  });

  it('hedges nothing when the tree has two bands or fewer', () => {
    expect(approximateLevelKeys([label({ level: 0 })]).size).toBe(0);
    expect(
      approximateLevelKeys([label({ level: STEP }), label({ level: 0, x: 200 })]).size,
    ).toBe(0);
  });

  it('never hedges ground-truth signs, however deep the taxonomy', () => {
    const gt = fourLayers().map((l) => ({ ...l, source: GROUND_TRUTH_SIGN_SOURCE }));
    expect(approximateLevelKeys(gt).size).toBe(0);
  });

  it('buckets levels that differ by a float hair into one band', () => {
    // Bands are derived from a float product server-side (`(n_layers - 1 - i) *
    // 1.8`). Two signs of one band splitting into two would shift every rank
    // below them and hedge a layer the measurement found fine, so the bucketing
    // has to be tolerant rather than exact.
    const drifted = 3 * STEP + Number.EPSILON * 4;
    expect(drifted).not.toBe(3 * STEP);
    const keys = approximateLevelKeys([
      label({ level: drifted, text: 'a' }),
      label({ level: 3 * STEP, x: 200, text: 'b' }),
      label({ level: STEP, x: 400, text: 'c' }),
      label({ level: 0, x: 600, text: 'd' }),
    ]);
    // Three bands, not four → only the coarsest hedges.
    expect(keys.size).toBe(1);
    expect(keys.has(0)).toBe(true);
  });

  it('ignores empty-text labels when ranking bands', () => {
    // An unnamed band never reaches the map, so it must not push real bands
    // one rank coarser.
    const keys = approximateLevelKeys([
      label({ level: 3 * STEP, text: '' }),
      label({ level: 2 * STEP, x: 200, text: 'fine' }),
      label({ level: STEP, x: 400, text: 'mid' }),
      label({ level: 0, x: 600, text: 'coarse' }),
    ]);
    expect(keys.size).toBe(1);
    expect(keys.has(0)).toBe(true);
  });
});

describe('layoutSigns — approximate coarse bands', () => {
  it('hedges a coarse band and leaves the fine one alone', () => {
    const placed = layoutSigns(fourLayers(), centeredView(CRISP_AND_HEDGED_VIEW), measure);
    expect(placed.map((p) => p.label.text).sort()).toEqual(['layer-1', 'layer-2']);

    const crisp = placed.find((p) => p.label.text === 'layer-1')!;
    expect(crisp.approximate).toBe(false);
    expect(crisp.text).toBe('layer-1');

    const hedged = placed.find((p) => p.label.text === 'layer-2')!;
    expect(hedged.approximate).toBe(true);
    expect(hedged.text).toBe(`${SIGN_APPROXIMATE_PREFIX}layer-2`);
    // `label.text` stays the raw name — only the painted text carries the hedge.
    expect(hedged.label.text).toBe('layer-2');
  });

  it('dims an approximate sign by SIGN_APPROXIMATE_ALPHA', () => {
    const placed = layoutSigns(fourLayers(), centeredView(CRISP_AND_HEDGED_VIEW), measure);
    // Both bands sit inside the opaque plateau, so the only alpha difference is
    // the hedge.
    expect(placed.find((p) => p.label.text === 'layer-1')!.alpha).toBe(1);
    expect(placed.find((p) => p.label.text === 'layer-2')!.alpha).toBeCloseTo(
      SIGN_APPROXIMATE_ALPHA,
      6,
    );
  });

  it('measures the prefixed text, and tells the measurer which face to use', () => {
    const seen: [string, boolean][] = [];
    const spy = (text: string, fontPx: number, approximate: boolean) => {
      seen.push([text, approximate]);
      return measure(text, fontPx);
    };
    layoutSigns(fourLayers(), centeredView(CRISP_AND_HEDGED_VIEW), spy);
    expect(seen).toContainEqual(['layer-1', false]);
    expect(seen).toContainEqual([`${SIGN_APPROXIMATE_PREFIX}layer-2`, true]);
  });

  it('changes only presentation — the same bands appear either way', () => {
    const clustered = layoutSigns(fourLayers(), centeredView(CRISP_AND_HEDGED_VIEW), measure);
    const groundTruth = layoutSigns(
      fourLayers().map((l) => ({ ...l, source: GROUND_TRUTH_SIGN_SOURCE })),
      centeredView(CRISP_AND_HEDGED_VIEW),
      measure,
    );
    expect(clustered.map((p) => p.label.text).sort()).toEqual(
      groundTruth.map((p) => p.label.text).sort(),
    );
    expect(groundTruth.every((p) => !p.approximate)).toBe(true);
  });

  it('leaves a single-band label set entirely un-hedged', () => {
    const placed = layoutSigns([label({ level: 0 })], centeredView(0), measure);
    expect(placed[0].approximate).toBe(false);
    expect(placed[0].text).toBe('Birdsong');
  });
});
