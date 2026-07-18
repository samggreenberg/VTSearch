// Bin-shape geometry for the browse canvas and its minimap. The projection
// pyramid can tile the 2-D layout as hexagons (default) or squares; both share
// the per-level `radius` scale and pyramid structure, so the only things that
// differ between them are (a) the projection-space grid spacing used to pick
// covering tiles and (b) how a single cell is drawn / hit-tested. This module
// captures exactly those differences behind one `BinGeometry` interface so the
// canvas and minimap stay shape-agnostic.

import {
  SQRT3,
  HEX_INRADIUS_RATIO,
  traceCellPath as traceHexCellPath,
  densityColor,
  resolveColormap,
} from './hex-render.util';
import type { BinShape } from '../../models/projection.models';

export { SQRT3, densityColor, resolveColormap };
export type { BinShape };

/** Offset from a cell centre to a neighbour centre, in units of `radius`. */
export interface NeighborOffset {
  dx: number;
  dy: number;
}

export interface BinGeometry {
  readonly shape: BinShape;
  /** Column spacing in projection units for a cell of the given radius. */
  dx(radius: number): number;
  /** Row spacing in projection units for a cell of the given radius. */
  dy(radius: number): number;
  /**
   * Trace one cell's outline as the current path at screen coords `(cx, cy)`
   * with the given screen `radius`. A multi-item ("pile") cell (`rounded`) is
   * drawn with a soft rounded shape so piles stand out — a disc in hex mode, a
   * rounded-corner rectangle in square mode — while a single-item cell keeps
   * its full sharp hexagon / square, so a lone item reads as a crisp tile.
   */
  traceCell(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    rounded: boolean,
  ): void;
  /**
   * The absolute corner radius (screen px) a pile cell's rounded outline curves
   * with at the given bin `radius` — the inscribed disc's radius in hex mode,
   * the rounded square's corner radius in square mode. The hovered break-out
   * thumbnail reuses this fixed ("total", not proportional) value so an
   * enlarged pile keeps the *same* corner curvature it had in the grid: it
   * still reads as a pile when blown up, while only nibbling the image corners
   * instead of cropping more as the tile grows.
   */
  roundedCornerRadius(radius: number): number;
  /**
   * Whether a projection-space offset `(ox, oy)` from a cell center falls
   * inside a cell of the given `radius` — the hover hit-test predicate.
   */
  contains(ox: number, oy: number, radius: number): boolean;
  /**
   * Offsets, in units of `radius`, from a cell centre to each immediately
   * adjacent cell centre — six for a hex, four for a square. Multiply by a
   * screen `radius` to get neighbour-centre offsets in screen px. Used to size
   * a hovered thumbnail so it grows until its edge just reaches the nearest
   * neighbour centre, never covering one.
   */
  neighborOffsets(): readonly NeighborOffset[];
}

// Pointy-top hex neighbours: two along the row at ±√3·radius, and four in the
// rows above/below offset half a column (±√3/2·radius) and ±1.5·radius — the
// same spacing as `dx`/`dy`. All six sit one centre-distance (√3·radius) away.
const HEX_NEIGHBORS: readonly NeighborOffset[] = [
  { dx: SQRT3, dy: 0 },
  { dx: -SQRT3, dy: 0 },
  { dx: SQRT3 / 2, dy: 1.5 },
  { dx: SQRT3 / 2, dy: -1.5 },
  { dx: -SQRT3 / 2, dy: 1.5 },
  { dx: -SQRT3 / 2, dy: -1.5 },
];

// Square neighbours: the four edge-adjacent cells at ±√3·radius (the diagonal
// corners are deliberately excluded — the thumbnail may grow past a corner so
// long as it doesn't reach an edge neighbour's centre).
const SQUARE_NEIGHBORS: readonly NeighborOffset[] = [
  { dx: SQRT3, dy: 0 },
  { dx: -SQRT3, dy: 0 },
  { dx: 0, dy: SQRT3 },
  { dx: 0, dy: -SQRT3 },
];

// Pointy-top hexagon: column spacing `radius·√3`, row spacing `radius·1.5`.
// Hit-tested against the circumscribed circle (matching the original canvas).
const HEX_GEOMETRY: BinGeometry = {
  shape: 'hex',
  dx: (radius) => radius * SQRT3,
  dy: (radius) => radius * 1.5,
  traceCell: (ctx, cx, cy, radius, rounded) => traceHexCellPath(ctx, cx, cy, radius, rounded),
  roundedCornerRadius: (radius) => radius * HEX_INRADIUS_RATIO,
  contains: (ox, oy, radius) => ox * ox + oy * oy < radius * radius,
  neighborOffsets: () => HEX_NEIGHBORS,
};

// Corner radius of a square-mode pile, as a fraction of its half-side. A pile
// is drawn as a rounded-corner rectangle (vs the singleton's sharp square) so
// the two read as distinct shapes even before the colormap border lands.
const SQUARE_ROUNDED_CORNER_RATIO = 0.35;

// Square cell of side `radius·√3` (matching the hex column spacing so the two
// lattices have a comparable on-screen footprint and share the level picker).
// The backend corner-anchors the lattice on a fixed origin (a quadtree, so reps
// persist across zoom — see vtscore/projection/squarebin.py) and reports each
// cell's true midpoint as `(cx, cy)`; here we just draw a side-`radius·√3`
// square around that centre, so columns and rows are spaced one side apart.
const SQUARE_GEOMETRY: BinGeometry = {
  shape: 'square',
  dx: (radius) => radius * SQRT3,
  dy: (radius) => radius * SQRT3,
  traceCell: (ctx, cx, cy, radius, rounded) => {
    const half = (radius * SQRT3) / 2;
    ctx.beginPath();
    if (rounded) {
      // Rounded-corner rectangle: the rect-mode counterpart of the hex's disc.
      ctx.roundRect(cx - half, cy - half, half * 2, half * 2, half * SQUARE_ROUNDED_CORNER_RATIO);
    } else {
      ctx.rect(cx - half, cy - half, half * 2, half * 2);
    }
    ctx.closePath();
  },
  roundedCornerRadius: (radius) => ((radius * SQRT3) / 2) * SQUARE_ROUNDED_CORNER_RATIO,
  contains: (ox, oy, radius) => {
    const half = (radius * SQRT3) / 2;
    return Math.abs(ox) < half && Math.abs(oy) < half;
  },
  neighborOffsets: () => SQUARE_NEIGHBORS,
};

/** The geometry for a bin shape; defaults to hex for an unset/unknown shape. */
export function binGeometry(shape: BinShape | undefined): BinGeometry {
  return shape === 'square' ? SQUARE_GEOMETRY : HEX_GEOMETRY;
}

/** The minimum a hit-test needs of a cell: its centre in projection space. */
export interface CellCentre {
  cx: number;
  cy: number;
}

/**
 * The nearest cell to projection-space point `(px, py)` among `cells`, returned
 * only if that point falls *inside* it (per `geom.contains` at `radius`) — so it
 * resolves to `null` over blank space between bins. This is the pure core of the
 * canvas hover/click hit-test: the component gathers the candidate cells from the
 * covering tiles around the cursor, then this picks the one under it. Kept
 * framework-free (no tile cache, no transform) so the "nearest-then-contains"
 * rule is unit-testable on its own.
 */
export function pickCell<T extends CellCentre>(
  cells: Iterable<T>,
  px: number,
  py: number,
  geom: BinGeometry,
  radius: number,
): T | null {
  let best: T | null = null;
  let bestDist = Infinity;
  for (const cell of cells) {
    const cdx = cell.cx - px;
    const cdy = cell.cy - py;
    const dist = cdx * cdx + cdy * cdy;
    if (dist < bestDist) {
      bestDist = dist;
      best = cell;
    }
  }
  if (best && geom.contains(best.cx - px, best.cy - py, radius)) return best;
  return null;
}

/**
 * Half-extents (screen px) of a hovered break-out thumbnail with the given
 * `aspect` (image width / height), centred on a cell of screen `radius` and
 * grown as large as possible under one rule: no neighbour cell's centre may be
 * covered. A neighbour at offset `(dx, dy)·radius` stays uncovered while the
 * half-height `hh ≤ max(|dx|·radius / aspect, |dy|·radius)`; the binding
 * neighbour is the tightest of `offsets`, and the rectangle's edge then just
 * touches that centre. Wide images grow until they reach the side neighbours,
 * tall ones until they reach the top/bottom. Pulled out of the canvas component
 * so the neighbour-centre rule is unit-testable without an `HTMLImageElement`.
 */
export function hoverThumbHalfExtents(
  aspect: number,
  radius: number,
  offsets: readonly NeighborOffset[],
): { hw: number; hh: number } {
  let hh = Infinity;
  for (const { dx, dy } of offsets) {
    hh = Math.min(hh, Math.max((Math.abs(dx) * radius) / aspect, Math.abs(dy) * radius));
  }
  return { hw: aspect * hh, hh };
}

/** How a grid thumbnail is fitted into a bin's `2*radius` square. */
export type TileFit = 'cover' | 'balanced';

/**
 * Draw dimensions (screen px) for a thumbnail of intrinsic size `iw × ih`
 * painted over a bin's `2*radius` square, centred and clipped to the cell. The
 * caller draws the image at `(cx - dw/2, cy - dh/2)` with size `dw × dh`.
 *
 * - `'cover'` scales to fill the square (`max` of the two axis ratios), so the
 *   longer axis overflows and is clipped — the historical crop-to-fill fit still
 *   used for video/audio tiles.
 * - `'balanced'` scales by the geometric mean of the cover and contain ratios
 *   (`size / √(iw·ih)`), so the fraction cropped off the long axis exactly
 *   equals the fraction of background gap left on the short axis — the "half
 *   crop, half pad" fit used for image tiles. (The returned box is the geometric
 *   mean of the cover box, `≥ size` on both axes, and the contain box, `≤ size`
 *   on both axes.) A square image is unaffected: both fits return `size × size`.
 *
 * Pulled out of the canvas component so the crop/pad split is unit-testable
 * without a `CanvasRenderingContext2D`.
 */
export function imageTileFitDimensions(
  iw: number,
  ih: number,
  radius: number,
  fit: TileFit,
): { dw: number; dh: number } {
  const size = radius * 2;
  const scale = fit === 'balanced' ? size / Math.sqrt(iw * ih) : Math.max(size / iw, size / ih);
  return { dw: iw * scale, dh: ih * scale };
}
