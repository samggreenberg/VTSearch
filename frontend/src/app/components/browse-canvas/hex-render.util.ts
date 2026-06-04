// Shared hex-grid rendering primitives for the browse canvas and its minimap.
// Kept in one place so the density colormap and hex geometry can't drift
// between the full view and the overview.

export const SQRT3 = Math.sqrt(3);
const DEG30 = Math.PI / 6;

/** Pointy-top hexagon vertex angles (matches the projection's hex layout). */
export const HEX_ANGLES = Array.from({ length: 6 }, (_, i) => (Math.PI / 3) * i - DEG30);

// Dark-red → yellow density ramp. The low end is a deep red rather than black
// so that pure black stays free to mean "nothing here" (None): empty space on
// the canvas reads as absence, while any occupied hex is at least dark red.
const HEATMAP: [number, number, number][] = [
  [90, 0, 0],
  [140, 12, 0],
  [185, 28, 0],
  [220, 60, 0],
  [240, 105, 0],
  [250, 150, 5],
  [255, 195, 25],
  [255, 235, 70],
];

/** Map a normalized density ``t`` in [0, 1] to a darkred→yellow ``rgb(...)`` string. */
export function densityColor(t: number): string {
  const n = HEATMAP.length - 1;
  const idx = t * n;
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n);
  const frac = idx - lo;
  const r = Math.round(HEATMAP[lo][0] + (HEATMAP[hi][0] - HEATMAP[lo][0]) * frac);
  const g = Math.round(HEATMAP[lo][1] + (HEATMAP[hi][1] - HEATMAP[lo][1]) * frac);
  const b = Math.round(HEATMAP[lo][2] + (HEATMAP[hi][2] - HEATMAP[lo][2]) * frac);
  return `rgb(${r},${g},${b})`;
}

/** Trace a hexagon outline as the current path (no fill/stroke). */
export function traceHexPath(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
): void {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const x = cx + radius * Math.cos(HEX_ANGLES[i]);
    const y = cy + radius * Math.sin(HEX_ANGLES[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

/**
 * A hex's inscribed-circle radius as a fraction of its circumradius
 * (``radius``). A disc of this radius is the largest circle that fits inside
 * the hex, so a singleton drawn as a disc reads slightly smaller than the hex
 * it replaces.
 */
export const HEX_INRADIUS_RATIO = SQRT3 / 2;

/**
 * Trace one cell's outline as the current path. A cell holding a single media
 * item (``single``) is drawn as the hex's inscribed disc — barely smaller than
 * the hex, and visibly so since a disc has less area than the hex around it —
 * so singletons read as distinct dots. Every other cell keeps the full hex.
 */
export function traceCellPath(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  single: boolean,
): void {
  if (single) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * HEX_INRADIUS_RATIO, 0, Math.PI * 2);
    ctx.closePath();
  } else {
    traceHexPath(ctx, cx, cy, radius);
  }
}
