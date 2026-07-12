// Pure geometry/visibility math for the browse canvas's region signposts — the
// "street sign" name layer (see docs/plans/vtsbrowse-toponymy.md, G5). Framework-
// free (no Angular dependency) so it's independently unit-testable, mirroring the
// sibling `view-transform.ts`: plain exported functions, no state of its own. The
// canvas component owns the live labels/transform and threads them through.
//
// The model: every sign carries a pyramid `level` (0 = coarsest — continents;
// deeper = finer — countries, then states). The viewer's zoom maps to a
// *continuous* level on the same scale ({@link viewLevelForZoom}), and a sign's
// appearance is a function of `delta = viewLevel - sign.level` — how far the
// view has zoomed past the sign's own level:
//
//   delta < -1.5          invisible (the sign is far finer than the view)
//   -1.5 → -1.0           fades in as small text (just below the user's zoom)
//   -1.0 → 0              grows toward full size
//   0                     full size (at the user's zoom)
//   0 → +1.5              keeps enlarging (the view is inside the sign's region)
//   +1.5 → +2.5           enlarges a little more while fading out
//   delta > +2.5          gone (the view has zoomed far past the sign)
//
// so panning down through the zoom stack dissolves coarse names into the finer
// names that refine them, the way a paper map hands "EUROPE" off to "France".

import type { RegionLabelPayload, ViewTransform } from '../../models/projection.models';
import { projToScreen } from './view-transform';

/** Scale + opacity a sign renders at, before de-cluttering. */
export interface SignAppearance {
  /** Multiplier on {@link SIGN_BASE_FONT_PX}. Monotonically increasing in
   *  `delta`, so a sign only ever grows as the user zooms in past it. */
  scale: number;
  /** 0..1 opacity: ramps in at the fine edge, out at the coarse edge. */
  alpha: number;
}

/** A sign that survived visibility + de-cluttering, ready to paint. */
export interface PlacedSign {
  label: RegionLabelPayload;
  /** Screen (canvas CSS px) centre of the sign. */
  sx: number;
  sy: number;
  /** Resolved font size (CSS px) after the appearance scale. */
  fontPx: number;
  alpha: number;
  /** Pill box (CSS px), centred on `(sx, sy)`. */
  w: number;
  h: number;
}

/** Everything {@link layoutSigns} needs to know about the current view. */
export interface SignViewContext {
  transform: ViewTransform;
  width: number;
  height: number;
  /** Continuous pyramid level of the view; see {@link viewLevelForZoom}. */
  viewLevel: number;
}

/** Font size (CSS px) of a sign exactly at the user's zoom (`delta = 0`). */
export const SIGN_BASE_FONT_PX = 13;
/** Font the signs render/measure with. Weight rides separately (600). */
export const SIGN_FONT_FAMILY = 'system-ui, sans-serif';

/** `delta` below which a sign is invisible: it lives that many levels finer
 *  than the view ("far below the user's zoom"). */
export const SIGN_APPEAR_DELTA = -1.5;
/** How many levels the fade-in spans, starting at {@link SIGN_APPEAR_DELTA}. */
export const SIGN_FADE_IN_SPAN = 0.5;
/** `delta` above which a sign is gone: the view has zoomed that many levels
 *  past it ("far above the user's zoom"). */
export const SIGN_EXPIRE_DELTA = 2.5;
/** How many levels the fade-out spans, ending at {@link SIGN_EXPIRE_DELTA}. */
export const SIGN_FADE_OUT_SPAN = 1.0;

/** Piecewise-linear scale curve: `(delta, scale)` stops, interpolated between
 *  and clamped outside. Monotonic, so a sign grows continuously as the user
 *  zooms in past it — small text just below their zoom, full size at it,
 *  enlarged again above it — with no snapping between discrete size tiers. */
const SIGN_SCALE_STOPS: readonly [number, number][] = [
  [SIGN_APPEAR_DELTA, 0.65],
  [-0.5, 0.8],
  [0.5, 1.0],
  [1.5, 1.25],
  [SIGN_EXPIRE_DELTA, 1.45],
];

/** Minimum gap (CSS px) enforced between sign pills by the de-clutter pass. */
const SIGN_GAP_PX = 4;
/** Pill horizontal padding, as a fraction of the font size (per side). */
const SIGN_PAD_X_EM = 0.6;
/** Pill height, as a multiple of the font size. */
const SIGN_HEIGHT_EM = 1.7;

/**
 * The view's *continuous* pyramid level at on-screen zoom `effZoom` — the
 * unrounded form of `levelForEffZoom` in `view-transform.ts` (which quantises
 * this to pick the bin LOD). Signs interpolate on it instead so their size and
 * opacity track the zoom smoothly rather than stepping at level boundaries.
 * Unclamped: near the whole-projection fit it can dip below 0, which simply
 * reads as "slightly coarser than the coarsest layer".
 */
export function viewLevelForZoom(baseRadius: number, effZoom: number, targetRadius: number): number {
  return Math.log2((baseRadius * effZoom) / targetRadius);
}

/**
 * Scale + opacity for a sign whose level sits `delta` levels below the view
 * (`delta = viewLevel - sign.level`), or `null` when the sign is outside its
 * visibility band entirely. Pure and total — see the module comment for the
 * band-by-band behaviour.
 */
export function signAppearance(delta: number): SignAppearance | null {
  if (!Number.isFinite(delta)) return null;
  if (delta <= SIGN_APPEAR_DELTA || delta >= SIGN_EXPIRE_DELTA) return null;

  let alpha = 1;
  const fadeInEnd = SIGN_APPEAR_DELTA + SIGN_FADE_IN_SPAN;
  const fadeOutStart = SIGN_EXPIRE_DELTA - SIGN_FADE_OUT_SPAN;
  if (delta < fadeInEnd) {
    alpha = (delta - SIGN_APPEAR_DELTA) / SIGN_FADE_IN_SPAN;
  } else if (delta > fadeOutStart) {
    alpha = (SIGN_EXPIRE_DELTA - delta) / SIGN_FADE_OUT_SPAN;
  }

  return { scale: interpolateScale(delta), alpha };
}

/** Piecewise-linear interpolation through {@link SIGN_SCALE_STOPS}. */
function interpolateScale(delta: number): number {
  const stops = SIGN_SCALE_STOPS;
  if (delta <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [d1, s1] = stops[i];
    if (delta <= d1) {
      const [d0, s0] = stops[i - 1];
      return s0 + ((delta - d0) / (d1 - d0)) * (s1 - s0);
    }
  }
  return stops[stops.length - 1][1];
}

/**
 * Resolve which signs are visible at the current view and where, de-cluttered
 * so no two pills overlap. `measure` returns the rendered width of `text` at
 * `fontPx` (the canvas passes `ctx.measureText`; tests pass a stub), keeping
 * this module free of any canvas dependency.
 *
 * De-cluttering is greedy by priority: larger (coarser-relative) signs first,
 * score as the tiebreak, so when a country name and a state name compete for
 * the same pixels the one nearer its own zoom band — the bigger, more legible
 * one — wins and the loser simply isn't drawn this frame. Losing is transient:
 * a later zoom re-runs the layout and the sign gets another chance.
 */
export function layoutSigns(
  labels: readonly RegionLabelPayload[],
  view: SignViewContext,
  measure: (text: string, fontPx: number) => number,
  baseFontPx: number = SIGN_BASE_FONT_PX,
): PlacedSign[] {
  const candidates: PlacedSign[] = [];
  for (const label of labels) {
    if (!label.text) continue;
    const appearance = signAppearance(view.viewLevel - label.level);
    if (!appearance || appearance.alpha <= 0.02) continue;

    const [sx, sy] = projToScreen(label.x, label.y, view.transform, view.width, view.height);
    const fontPx = baseFontPx * appearance.scale;
    const w = measure(label.text, fontPx) + 2 * SIGN_PAD_X_EM * fontPx;
    const h = SIGN_HEIGHT_EM * fontPx;
    // Cull signs entirely off screen (their pill can't reach the viewport).
    if (sx + w / 2 < 0 || sx - w / 2 > view.width) continue;
    if (sy + h / 2 < 0 || sy - h / 2 > view.height) continue;

    candidates.push({ label, sx, sy, fontPx, alpha: appearance.alpha, w, h });
  }

  // Priority: bigger signs first (they're the ones nearest their own zoom
  // band), then higher score. Sort is stable, so equal-priority signs keep
  // their input order and the layout doesn't flicker between frames.
  candidates.sort(
    (a, b) => b.fontPx - a.fontPx || (b.label.score ?? 0) - (a.label.score ?? 0),
  );

  const placed: PlacedSign[] = [];
  for (const sign of candidates) {
    if (placed.some((kept) => pillsOverlap(kept, sign))) continue;
    placed.push(sign);
  }
  return placed;
}

/** Whether two sign pills (inflated by the minimum gap) intersect. */
function pillsOverlap(a: PlacedSign, b: PlacedSign): boolean {
  return (
    Math.abs(a.sx - b.sx) < (a.w + b.w) / 2 + SIGN_GAP_PX &&
    Math.abs(a.sy - b.sy) < (a.h + b.h) / 2 + SIGN_GAP_PX
  );
}
