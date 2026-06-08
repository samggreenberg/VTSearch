// Bin-shape geometry for the browse canvas and its minimap. The projection
// pyramid can tile the 2-D layout as hexagons (default) or squares; both share
// the per-level `radius` scale and pyramid structure, so the only things that
// differ between them are (a) the projection-space grid spacing used to pick
// covering tiles and (b) how a single cell is drawn / hit-tested. This module
// captures exactly those differences behind one `BinGeometry` interface so the
// canvas and minimap stay shape-agnostic.

import {
  SQRT3,
  HEX_ANGLES,
  traceCellPath as traceHexCellPath,
  densityColor,
  resolveColormap,
} from './hex-render.util';
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
  /**
   * Trace the pile cell outline clipped to the centered axis-aligned rectangle
   * of half-extents `(hw, hh)` as the current path. This is the cell shape with
   * the corners that stick out past the rectangle lopped off — used to draw a
   * hovered thumbnail tile trimmed to the thumbnail's contain-fit rectangle, so
   * the enlarged tile doesn't paint background over its neighbours. The outer
   * `radius` (and so the shape away from the cut) is unchanged.
   */
  traceTrimmedCell(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    hw: number,
    hh: number,
  ): void;
}

// Sutherland–Hodgman clip of a convex polygon (local coords, centered on the
// origin) against one axis-aligned half-plane, returning the clipped polygon.
type Pt = { x: number; y: number };
function clipHalfPlane(poly: Pt[], axis: 'x' | 'y', value: number, keepGreater: boolean): Pt[] {
  if (poly.length === 0) return poly;
  const inside = (p: Pt) => (keepGreater ? p[axis] >= value : p[axis] <= value);
  const crossing = (a: Pt, b: Pt): Pt => {
    const t = (value - a[axis]) / (b[axis] - a[axis]);
    return axis === 'x'
      ? { x: value, y: a.y + (b.y - a.y) * t }
      : { x: a.x + (b.x - a.x) * t, y: value };
  };
  const out: Pt[] = [];
  for (let i = 0; i < poly.length; i++) {
    const cur = poly[i];
    const prev = poly[(i + poly.length - 1) % poly.length];
    const curIn = inside(cur);
    if (curIn) {
      if (!inside(prev)) out.push(crossing(prev, cur));
      out.push(cur);
    } else if (inside(prev)) {
      out.push(crossing(prev, cur));
    }
  }
  return out;
}

// Trace `verts` (local, origin-centered) clipped to the `±hw × ±hh` rectangle as
// the current path at screen center `(cx, cy)`.
function traceClippedPolygon(
  ctx: CanvasRenderingContext2D,
  verts: Pt[],
  cx: number,
  cy: number,
  hw: number,
  hh: number,
): void {
  let poly = clipHalfPlane(verts, 'x', -hw, true);
  poly = clipHalfPlane(poly, 'x', hw, false);
  poly = clipHalfPlane(poly, 'y', -hh, true);
  poly = clipHalfPlane(poly, 'y', hh, false);
  ctx.beginPath();
  poly.forEach((p, i) => {
    if (i === 0) ctx.moveTo(cx + p.x, cy + p.y);
    else ctx.lineTo(cx + p.x, cy + p.y);
  });
  ctx.closePath();
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
  traceTrimmedCell: (ctx, cx, cy, radius, hw, hh) => {
    const verts = HEX_ANGLES.map((a) => ({
      x: radius * Math.cos(a),
      y: radius * Math.sin(a),
    }));
    traceClippedPolygon(ctx, verts, cx, cy, hw, hh);
  },
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
  traceTrimmedCell: (ctx, cx, cy, radius, hw, hh) => {
    const half = (radius * SQRT3) / 2;
    const verts: Pt[] = [
      { x: -half, y: -half },
      { x: half, y: -half },
      { x: half, y: half },
      { x: -half, y: half },
    ];
    traceClippedPolygon(ctx, verts, cx, cy, hw, hh);
  },
};

/** The geometry for a bin shape; defaults to hex for an unset/unknown shape. */
export function binGeometry(shape: BinShape | undefined): BinGeometry {
  return shape === 'square' ? SQUARE_GEOMETRY : HEX_GEOMETRY;
}
