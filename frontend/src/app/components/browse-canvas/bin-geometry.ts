// Bin-shape geometry for the browse canvas and its minimap. The projection
// pyramid can tile the 2-D layout as hexagons (default) or squares; both share
// the per-level `radius` scale and pyramid structure, so the only things that
// differ between them are (a) the projection-space grid spacing used to pick
// covering tiles and (b) how a single cell is drawn / hit-tested. This module
// captures exactly those differences behind one `BinGeometry` interface so the
// canvas and minimap stay shape-agnostic.

import { SQRT3, traceCellPath as traceHexCellPath, densityColor, resolveColormap } from './hex-render.util';
import type { BinShape } from '../../models/projection.models';

export { SQRT3, densityColor, resolveColormap };
export type { BinShape };

export interface BinGeometry {
  readonly shape: BinShape;
  /** Column spacing in projection units for a cell of the given radius. */
  dx(radius: number): number;
  /** Row spacing in projection units for a cell of the given radius. */
  dy(radius: number): number;
  /**
   * Trace one cell's outline as the current path at screen coords `(cx, cy)`
   * with the given screen `radius`. A `single`-item cell is drawn with a
   * distinct shape so singletons stand out — a disc in hex mode, a
   * rounded-corner rectangle in square mode — while every other (pile) cell
   * keeps its full hexagon / square so it tiles the space.
   */
  traceCell(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    single: boolean,
  ): void;
  /**
   * Whether a projection-space offset `(ox, oy)` from a cell center falls
   * inside a cell of the given `radius` — the hover hit-test predicate.
   */
  contains(ox: number, oy: number, radius: number): boolean;
  /**
   * Half-width / half-height of the cell's bounding box at the given screen
   * `radius`. Used to contain-fit a thumbnail inside a cell: a hex is wider
   * than tall (`√3·radius` × `2·radius`) while a square is `√3·radius` on a
   * side, so the fit box differs by shape.
   */
  contentHalfExtent(radius: number): { hw: number; hh: number };
}

// Pointy-top hexagon: column spacing `radius·√3`, row spacing `radius·1.5`.
// Hit-tested against the circumscribed circle (matching the original canvas).
const HEX_GEOMETRY: BinGeometry = {
  shape: 'hex',
  dx: (radius) => radius * SQRT3,
  dy: (radius) => radius * 1.5,
  traceCell: (ctx, cx, cy, radius, single) => traceHexCellPath(ctx, cx, cy, radius, single),
  contains: (ox, oy, radius) => ox * ox + oy * oy < radius * radius,
  contentHalfExtent: (radius) => ({ hw: (radius * SQRT3) / 2, hh: radius }),
};

// Corner radius of a square-mode singleton, as a fraction of its half-side. A
// singleton is drawn as a rounded-corner rectangle (vs the pile's sharp square)
// so the two read as distinct shapes even before the colormap border lands.
const SQUARE_SINGLE_CORNER_RATIO = 0.35;

// Square cell of side `radius·√3` (matching the hex column spacing so the two
// lattices have a comparable on-screen footprint and share the level picker).
// Centered on lattice points, so columns and rows are spaced one side apart.
const SQUARE_GEOMETRY: BinGeometry = {
  shape: 'square',
  dx: (radius) => radius * SQRT3,
  dy: (radius) => radius * SQRT3,
  traceCell: (ctx, cx, cy, radius, single) => {
    const half = (radius * SQRT3) / 2;
    ctx.beginPath();
    if (single) {
      // Rounded-corner rectangle: the rect-mode counterpart of the hex's disc.
      ctx.roundRect(cx - half, cy - half, half * 2, half * 2, half * SQUARE_SINGLE_CORNER_RATIO);
    } else {
      ctx.rect(cx - half, cy - half, half * 2, half * 2);
    }
    ctx.closePath();
  },
  contains: (ox, oy, radius) => {
    const half = (radius * SQRT3) / 2;
    return Math.abs(ox) < half && Math.abs(oy) < half;
  },
  contentHalfExtent: (radius) => {
    const half = (radius * SQRT3) / 2;
    return { hw: half, hh: half };
  },
};

/** The geometry for a bin shape; defaults to hex for an unset/unknown shape. */
export function binGeometry(shape: BinShape | undefined): BinGeometry {
  return shape === 'square' ? SQUARE_GEOMETRY : HEX_GEOMETRY;
}
