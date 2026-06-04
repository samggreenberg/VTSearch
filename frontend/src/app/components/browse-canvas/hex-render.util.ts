// Shared hex-grid rendering primitives for the browse canvas and its minimap.
// Kept in one place so the density colormap and hex geometry can't drift
// between the full view and the overview.

export const SQRT3 = Math.sqrt(3);
const DEG30 = Math.PI / 6;

/** Pointy-top hexagon vertex angles (matches the projection's hex layout). */
export const HEX_ANGLES = Array.from({ length: 6 }, (_, i) => (Math.PI / 3) * i - DEG30);

type RGB = [number, number, number];

/** The effective theme the canvas is rendered under (no ``'system'``). */
export type CanvasTheme = 'dark' | 'light' | 'highviz';

/**
 * A density colormap, resolved for a concrete theme. ``single`` is the colour
 * for a one-item cell (drawn as a distinct dot); ``ramp`` is the low→high
 * density gradient for multi-item cells. The "nothing here" colour is never
 * part of the map — empty space is painted with the canvas background so it
 * reads as absence — so every ``single``/``ramp`` colour means "at least one".
 */
export interface ResolvedColormap {
  single: RGB;
  ramp: RGB[];
}

/** The colormap presets the user can pick per media type in Settings → Browser. */
export type BrowseColormapId = 'auto' | 'heat' | 'ocean' | 'gray';

/** All selectable colormap ids, in pulldown order (``auto`` first). */
export const BROWSE_COLORMAP_IDS: readonly BrowseColormapId[] = ['auto', 'heat', 'ocean', 'gray'];

// "Heat": dark-red → yellow. The low end is a deep red rather than black so
// that black stays free to mean "nothing here" — empty space reads as absence
// while any occupied cell is at least dark red. Brightness rises with density,
// which reads naturally on a dark background. This is the dark-mode default.
const HEAT_RAMP: RGB[] = [
  [90, 0, 0],
  [140, 12, 0],
  [185, 28, 0],
  [220, 60, 0],
  [240, 105, 0],
  [250, 150, 5],
  [255, 195, 25],
  [255, 235, 70],
];

// "Ocean": light-blue → dark-navy, with a neutral light grey for singletons.
// Darkness rises with density, which reads naturally on a light background
// (more ink = more items) and keeps the hues disjoint from Heat's red/yellow —
// so "this bin is blue" unambiguously means light mode + lots, and "yellow"
// means dark mode + lots. This is the light-mode default.
const OCEAN_RAMP: RGB[] = [
  [222, 235, 247],
  [198, 219, 239],
  [158, 202, 225],
  [107, 174, 214],
  [66, 146, 198],
  [33, 113, 181],
  [8, 81, 156],
  [8, 48, 107],
];
const OCEAN_SINGLE: RGB = [200, 205, 214];

// "Grayscale": a neutral luminance ramp that always moves *away* from the
// background so density stays legible — darker as it grows on a light canvas,
// lighter as it grows on a dark one — so the direction is theme-dependent.
const GRAY_LIGHT: ResolvedColormap = {
  single: [176, 176, 176],
  ramp: [
    [150, 150, 150],
    [122, 122, 122],
    [92, 92, 92],
    [60, 60, 60],
    [28, 28, 28],
  ],
};
const GRAY_DARK: ResolvedColormap = {
  single: [96, 96, 96],
  ramp: [
    [110, 110, 110],
    [150, 150, 150],
    [185, 185, 185],
    [214, 214, 214],
    [240, 240, 240],
  ],
};

const HEAT: ResolvedColormap = { single: HEAT_RAMP[0], ramp: HEAT_RAMP };
const OCEAN: ResolvedColormap = { single: OCEAN_SINGLE, ramp: OCEAN_RAMP };

/**
 * Resolve a colormap id to concrete colours for *theme*. ``auto`` picks the
 * per-theme default — Ocean (blue, darkens with density) in light mode, Heat
 * (red→yellow, brightens with density) in dark/high-viz — so the field always
 * has good contrast against the background without the user choosing a map.
 */
export function resolveColormap(id: BrowseColormapId, theme: CanvasTheme): ResolvedColormap {
  switch (id) {
    case 'heat':
      return HEAT;
    case 'ocean':
      return OCEAN;
    case 'gray':
      return theme === 'light' ? GRAY_LIGHT : GRAY_DARK;
    case 'auto':
    default:
      return theme === 'light' ? OCEAN : HEAT;
  }
}

/** Format an ``[r, g, b]`` triple as a CSS ``rgb(...)`` string. */
export function rgbString(c: RGB): string {
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Map a normalized density ``t`` in [0, 1] to an ``rgb(...)`` string on *ramp*. */
export function densityColor(t: number, ramp: RGB[]): string {
  const n = ramp.length - 1;
  const idx = t * n;
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n);
  const frac = idx - lo;
  const r = Math.round(ramp[lo][0] + (ramp[hi][0] - ramp[lo][0]) * frac);
  const g = Math.round(ramp[lo][1] + (ramp[hi][1] - ramp[lo][1]) * frac);
  const b = Math.round(ramp[lo][2] + (ramp[hi][2] - ramp[lo][2]) * frac);
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
