/* The Browse (VTSBrowse) projection payloads.
 *
 * The wire shapes are the generated ones (`vtsearch/schemas/projection.py`),
 * re-exported here so the Browse code imports them from one place. What stays
 * hand-written is what the spec cannot say:
 *
 * - **`ProjectionLayout`**: one schema, `ProjectionMeta`, answers both a build
 *   in flight (status and progress only) and a ready layout, so the spec marks
 *   every layout field optional. A ready meta always carries them;
 *   {@link readyLayout} narrows a meta to the shape the canvas draws from.
 * - **`BinShape`**: the spec types `bin_shape` as a plain string.
 * - **`ViewTransform`**: the canvas's own pan/zoom state; no endpoint returns it.
 */

import type { LevelMeta } from '../generated/api-client/models/level-meta';
import type { ProjectionMeta } from '../generated/api-client/models/projection-meta';

export type { HexCell } from '../generated/api-client/models/hex-cell';
export type { LevelMeta } from '../generated/api-client/models/level-meta';
export type { ProjectionBuildResponse } from '../generated/api-client/models/projection-build-response';
export type { ProjectionLabelsResponse } from '../generated/api-client/models/projection-labels-response';
export type { ProjectionMeta } from '../generated/api-client/models/projection-meta';
export type { RegionLabel } from '../generated/api-client/models/region-label';
export type { TileResponse } from '../generated/api-client/models/tile-response';

/** How the projection pyramid tiles the 2-D layout. */
export type BinShape = 'hex' | 'square';

/**
 * A built projection's meta: the pyramid fields a `ready` meta always carries,
 * made required. The canvas, the minimap and the view geometry draw from this.
 */
export interface ProjectionLayout extends ProjectionMeta {
  projection_id: string;
  /** `[xmin, ymin, xmax, ymax]` of the frozen 2-D layout. */
  bounds: [number, number, number, number];
  base_radius: number;
  tile_span: number;
  point_count: number;
  levels: LevelMeta[];
}

/** The meta as a {@link ProjectionLayout}, or null while no layout is built (idle, building, error). */
export function readyLayout(meta: ProjectionMeta | null | undefined): ProjectionLayout | null {
  if (!meta || meta.status !== 'ready') return null;
  const { projection_id, bounds, base_radius, tile_span, point_count, levels } = meta;
  if (projection_id == null || bounds?.length !== 4) return null;
  if (base_radius == null || tile_span == null || point_count == null || levels == null) return null;
  return meta as ProjectionLayout;
}

export interface ViewTransform {
  centerX: number;
  centerY: number;
  zoom: number;
}
