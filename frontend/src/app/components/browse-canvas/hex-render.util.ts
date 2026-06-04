// Shared hex-grid rendering primitives for the browse canvas and its minimap.
// Kept in one place so the density colormap and hex geometry can't drift
// between the full view and the overview.

export const SQRT3 = Math.sqrt(3);
const DEG30 = Math.PI / 6;

/** Pointy-top hexagon vertex angles (matches the projection's hex layout). */
export const HEX_ANGLES = Array.from({ length: 6 }, (_, i) => (Math.PI / 3) * i - DEG30);

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [41, 120, 142],
  [33, 145, 140],
  [42, 168, 131],
  [68, 190, 112],
  [94, 201, 98],
  [128, 213, 79],
  [166, 222, 52],
  [199, 227, 33],
  [229, 228, 32],
  [253, 231, 37],
];

/** Map a normalized density ``t`` in [0, 1] to a viridis ``rgb(...)`` string. */
export function viridisColor(t: number): string {
  const n = VIRIDIS.length - 1;
  const idx = t * n;
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n);
  const frac = idx - lo;
  const r = Math.round(VIRIDIS[lo][0] + (VIRIDIS[hi][0] - VIRIDIS[lo][0]) * frac);
  const g = Math.round(VIRIDIS[lo][1] + (VIRIDIS[hi][1] - VIRIDIS[lo][1]) * frac);
  const b = Math.round(VIRIDIS[lo][2] + (VIRIDIS[hi][2] - VIRIDIS[lo][2]) * frac);
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
