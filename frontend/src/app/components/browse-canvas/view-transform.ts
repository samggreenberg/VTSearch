// Pure pan/zoom/clamp/rubber-band geometry for the browse canvas. Framework-free
// (no Angular dependency) so it's independently unit-testable and reusable — the
// canvas component owns the live `ViewTransform` and viewport size and threads
// them through these functions rather than this module holding any state of its
// own. Mirrors the sibling `bin-geometry.ts`: plain exported functions over a
// small context type, not a stateful class.

import type { ProjectionMeta, ViewTransform } from '../../models/projection.models';

/** The inputs every geometry function needs beyond the live transform: the
 *  active projection's pyramid metadata, the viewport size (CSS px), and the
 *  on-screen bin radius level selection targets. Bundled so call sites build it
 *  once per call rather than threading four separate parameters through. */
export interface ViewGeomContext {
  meta: ProjectionMeta | null;
  width: number;
  height: number;
  targetRadius: number;
}

/** Pan-centre limits at a given zoom: the content rectangle the viewport
 *  *centre* is held within, plus the viewport extent on each axis (used to
 *  scale the rubber-band give). See {@link panLimits}. */
export interface PanLimits {
  loX: number;
  hiX: number;
  loY: number;
  hiY: number;
  viewX: number;
  viewY: number;
}

/** Initial give of the rubber band: a pull just past the edge moves the view by
 *  this fraction of the pull (tapering off from there). */
const RUBBER_GIVE = 0.5;
/** Cap on pan overshoot, as a fraction of the viewport extent on that axis. */
const RUBBER_PAN_MAX = 0.18;
/** Cap on zoom overshoot past either limit (the zoom-out fit floor or the
 *  zoom-in finest-level ceiling), in natural-log units (≈26%/≈35% extra zoom
 *  out/in at exp(∓0.3)). Shared by {@link softFloorZoom} and {@link softCeilZoom}. */
const RUBBER_ZOOM_MAX = 0.3;

/** Pyramid level whose bins render closest to the current thumbnail size
 * (`ctx.targetRadius`) at the given on-screen zoom. Shared by live level
 * selection and fit framing. */
export function levelForEffZoom(ctx: ViewGeomContext, effZoom: number): number {
  const meta = ctx.meta;
  if (!meta || meta.levels.length === 0) return 0;
  const idealLevel = Math.log2((meta.base_radius * effZoom) / ctx.targetRadius);
  return Math.max(0, Math.min(meta.levels.length - 1, Math.round(idealLevel)));
}

/**
 * The zoom at which the whole projection just fits the viewport — the framing
 * `fitToData` (Zoom Fit) lands on, and the floor {@link clampedTransform} holds
 * the user to (zooming out past this only adds blank margins, so it is the
 * "useful edge"). `bounds` is the extent of the bin *centres*, but each edge
 * bin is drawn out to its circumradius beyond its centre, so framing on the
 * centres alone clips the edge bins; add the bin circumradius (in projection
 * units) as margin, plus a little breathing room. The active level — and thus
 * the radius — depends on the zoom we're solving for, so iterate a few times
 * from the no-margin fit to a fixed point (the level is quantised and clamps
 * at 0, so this settles immediately). `currentZoom` is returned unchanged when
 * there's no meta to fit against.
 */
export function computeFitZoom(ctx: ViewGeomContext, currentZoom: number): number {
  const meta = ctx.meta;
  if (!meta) return currentZoom;
  const [xmin, ymin, xmax, ymax] = meta.bounds;
  const dataW = xmax - xmin || 1;
  const dataH = ymax - ymin || 1;
  // Small breathing room beyond the bins themselves.
  const padding = 0.05;
  const w = ctx.width || 800;
  const h = ctx.height || 600;
  let zoom = Math.min(
    w / (dataW * (1 + padding * 2)),
    h / (dataH * (1 + padding * 2)),
  );
  for (let i = 0; i < 3; i++) {
    const level = levelForEffZoom(ctx, zoom);
    const r = meta.base_radius / Math.pow(2, level);
    const padW = dataW + 2 * (r + dataW * padding);
    const padH = dataH + 2 * (r + dataH * padding);
    zoom = Math.min(w / padW, h / padH);
  }
  return zoom;
}

/**
 * The zoom ceiling: the most you can zoom *in* before bins render larger than
 * they ever do during normal browsing. Symmetric to {@link computeFitZoom}
 * (the floor). Level selection ({@link levelForEffZoom}) keeps each bin within
 * `[targetRadius/√2, targetRadius·√2]` by handing off to a finer pyramid level
 * as you zoom in — but at the finest level (`levels.length - 1`) there's
 * nothing finer to switch to, so once the zoom passes the point where that
 * level would hand off (`idealLevel = maxLevel + 0.5`, where bins sit at their
 * normal-browsing max of `targetRadius·√2`) the bins just upscale and keep
 * growing without bound. Cap the zoom there so the thumbnails can't *stay*
 * bigger than that normal max.
 *
 * Tracks `ctx.targetRadius`, so making the thumbnails bigger (the +/- size
 * buttons scale zoom and `targetRadius` together) lifts the ceiling in
 * lock-step rather than fighting the resize.
 */
export function computeMaxZoom(ctx: ViewGeomContext, currentZoom: number): number {
  const minZoom = computeFitZoom(ctx, currentZoom);
  const meta = ctx.meta;
  if (!meta || meta.levels.length === 0) return minZoom;
  const maxLevel = meta.levels.length - 1;
  const maxZoom = (ctx.targetRadius * Math.pow(2, maxLevel + 0.5)) / meta.base_radius;
  // A tiny projection can fit whole at a zoom already past its finest-level
  // size; never let the ceiling drop below the floor (that would invert the
  // clamp and trap the view between two crossed limits).
  return Math.max(maxZoom, minZoom);
}

/**
 * Pan-centre limits at zoom `z`: the content rectangle the viewport *centre*
 * is held within (the bin-centre `bounds` grown by the edge bins' circumradius
 * `r`, since those bins draw out that far past their centres) plus the viewport
 * extent on each axis (used to scale the rubber-band give). Returns null when
 * there's no data to frame.
 */
export function panLimits(ctx: ViewGeomContext, z: number): PanLimits | null {
  const meta = ctx.meta;
  if (!meta || meta.point_count === 0) return null;
  const [xmin, ymin, xmax, ymax] = meta.bounds;
  const r = meta.base_radius / Math.pow(2, levelForEffZoom(ctx, z));
  return {
    loX: xmin - r,
    hiX: xmax + r,
    loY: ymin - r,
    hiY: ymax + r,
    viewX: ctx.width / z,
    viewY: ctx.height / z,
  };
}

/**
 * Clamp a viewport *centre* on one axis to the content range `[lo, hi]`, so any
 * point in the content can be brought to screen centre (drag the top-left bin
 * out from under the Back button and park it dead-centre if you like). The cost
 * is blank margin past the content edge: at centre = `lo` the viewport spills
 * half its extent past the edge into background. That spill is half the
 * viewport, so it's larger when zoomed out and smaller when zoomed in — the
 * wall sits further from the data the further out you are, which is the price of
 * being able to centre an edge point at any zoom.
 *
 * The clamp is `[lo, hi]` at *every* zoom, including when the viewport is wider
 * than the whole content (at/near the whole-projection fit): we deliberately do
 * NOT pin the centre to `(lo + hi) / 2` there. Pinning froze panning the moment
 * an axis's viewport grew past its content span — which killed all panning when
 * zoomed out, and (because the test is per-axis) let you centre an edge bin
 * vertically but not horizontally whenever the content's aspect ratio differed
 * from the viewport's. Keeping the full `[lo, hi]` range on both axes lets you
 * pull any bin to the centre of the current zoom, all the way out.
 */
export function clampAxis(center: number, lo: number, hi: number): number {
  return Math.min(Math.max(center, lo), hi);
}

/**
 * The hard-clamped form of `t` (zoom floored at the whole-projection fit and
 * capped at the finest-level ceiling {@link computeMaxZoom}, pan reined inside
 * the content rectangle — see {@link panLimits}). Pure: returns a fresh
 * transform without touching `t`, so the boundary-settle animation can compute
 * its destination while the live view is still mid-overshoot. Caller is
 * responsible for the meta/size guards (a no-data or zero-size viewport should
 * leave the transform untouched; this function doesn't special-case that).
 */
export function clampedTransform(t: ViewTransform, ctx: ViewGeomContext): ViewTransform {
  const minZoom = computeFitZoom(ctx, t.zoom);
  const maxZoom = computeMaxZoom(ctx, t.zoom);
  const zoom = Math.min(Math.max(t.zoom, minZoom), maxZoom);
  const lim = panLimits(ctx, zoom);
  return {
    zoom,
    centerX: lim ? clampAxis(t.centerX, lim.loX, lim.hiX) : t.centerX,
    centerY: lim ? clampAxis(t.centerY, lim.loY, lim.hiY) : t.centerY,
  };
}

// --- Rubber-band boundaries -----------------------------------------------
// Hitting the hard clamp used to stop the view dead, which reads as the tool
// freezing. Instead, a gesture that pushes past the edge keeps moving the view
// — with diminishing travel, so the boundary feels elastic — and on release
// the view eases back to the clamp. That turns "unresponsive" into a legible
// "you've reached the edge" cue. Only the live pan/zoom gestures go soft;
// programmatic moves (minimap recenter, resize refit) still hard-clamp.

/**
 * Signed diminishing overshoot. Near zero it tracks the input at
 * {@link RUBBER_GIVE} (so the first bit past the edge still moves), and as the
 * input grows it asymptotes to ±`maxOver`, so the view drifts a bounded amount
 * past a limit and never runs away no matter how hard the gesture pushes. This
 * is the standard asymptotic rubber-band curve `f(x) = maxOver·x / (maxOver/c + x)`.
 */
export function rubber(x: number, maxOver: number): number {
  if (maxOver <= 0) return 0;
  const c = RUBBER_GIVE;
  const s = Math.sign(x);
  const a = Math.abs(x);
  return (s * (maxOver * a)) / (maxOver / c + a);
}

/**
 * `clampAxis` with elastic edges: inside `[lo, hi]` it's the identity; past
 * either edge it returns a rubber-banded overshoot. Like `clampAxis` it uses
 * the full `[lo, hi]` range at every zoom (no centre-pinning when the viewport
 * is wider than the content), so a drag can pull any bin to the centre even at
 * the furthest zoom.
 */
export function rubberAxis(center: number, lo: number, hi: number, viewExtent: number): number {
  const maxOver = RUBBER_PAN_MAX * viewExtent;
  if (center < lo) return lo + rubber(center - lo, maxOver);
  if (center > hi) return hi + rubber(center - hi, maxOver);
  return center;
}

/**
 * Soft analogue of the pan half of {@link clampedTransform}: rather than pin
 * the centre at the content edge, let it drift past with rubber-band
 * resistance, so a drag into the wall still moves a little. Pure: returns the
 * softly-clamped centre without touching `t`.
 */
export function softClampPan(
  t: ViewTransform,
  ctx: ViewGeomContext,
  z: number,
): { centerX: number; centerY: number } {
  const lim = panLimits(ctx, z);
  if (!lim) return { centerX: t.centerX, centerY: t.centerY };
  return {
    centerX: rubberAxis(t.centerX, lim.loX, lim.hiX, lim.viewX),
    centerY: rubberAxis(t.centerY, lim.loY, lim.hiY, lim.viewY),
  };
}

/**
 * Soft analogue of the zoom floor in {@link clampedTransform}: a zoom below
 * the whole-projection fit is allowed but resisted in log space (perceptually
 * even with the rest of zooming), so wheeling out at the edge keeps responding
 * — the projection shrinks a touch more — before the settle springs it back.
 */
export function softFloorZoom(rawZoom: number, ctx: ViewGeomContext, currentZoom: number): number {
  const minZoom = computeFitZoom(ctx, currentZoom);
  if (rawZoom >= minZoom) return rawZoom;
  const over = Math.log(minZoom / rawZoom); // > 0: how far past the floor, in log units
  const damped = rubber(over, RUBBER_ZOOM_MAX);
  return minZoom * Math.exp(-damped);
}

/**
 * Soft analogue of the zoom ceiling in {@link clampedTransform}: a zoom above
 * the finest-level cap ({@link computeMaxZoom}) is allowed but resisted in log
 * space (perceptually even with the rest of zooming), so wheeling in at the
 * edge keeps responding — the bins grow a touch past their normal-browsing max
 * — before the settle springs it back. Mirror of {@link softFloorZoom}.
 */
export function softCeilZoom(rawZoom: number, ctx: ViewGeomContext, currentZoom: number): number {
  const maxZoom = computeMaxZoom(ctx, currentZoom);
  if (rawZoom <= maxZoom) return rawZoom;
  const over = Math.log(rawZoom / maxZoom); // > 0: how far past the ceiling, in log units
  const damped = rubber(over, RUBBER_ZOOM_MAX);
  return maxZoom * Math.exp(damped);
}

/** Projection-space `(px, py)` → screen (canvas CSS px) coords at transform `t`. */
export function projToScreen(
  px: number,
  py: number,
  t: ViewTransform,
  width: number,
  height: number,
): [number, number] {
  const z = t.zoom;
  const sx = (px - t.centerX) * z + width / 2;
  const sy = (py - t.centerY) * z + height / 2;
  return [sx, sy];
}

/** Screen (canvas CSS px) `(sx, sy)` → projection-space coords at transform `t`. */
export function screenToProj(
  sx: number,
  sy: number,
  t: ViewTransform,
  width: number,
  height: number,
): [number, number] {
  const z = t.zoom;
  const px = (sx - width / 2) / z + t.centerX;
  const py = (sy - height / 2) / z + t.centerY;
  return [px, py];
}

/** The projection-space rectangle currently visible in the viewport at
 *  transform `t`, as `[xmin, ymin, xmax, ymax]`. */
export function getVisibleBounds(
  t: ViewTransform,
  width: number,
  height: number,
): [number, number, number, number] {
  const [xmin, ymin] = screenToProj(0, 0, t, width, height);
  const [xmax, ymax] = screenToProj(width, height, t, width, height);
  return [
    Math.min(xmin, xmax),
    Math.min(ymin, ymax),
    Math.max(xmin, xmax),
    Math.max(ymin, ymax),
  ];
}
