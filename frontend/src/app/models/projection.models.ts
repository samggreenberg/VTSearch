/** How the projection pyramid tiles the 2-D layout. */
export type BinShape = 'hex' | 'square';

export interface HexCellPayload {
  q: number;
  r: number;
  cx: number;
  cy: number;
  count: number;
  rep_id: number;
  /**
   * All media ids aggregated in this cell. The canvas uses it to render the
   * cell's selection state (none / partial / full) and to toggle the whole
   * bin's contents. Re-derived server-side from the frozen layout, so it may be
   * absent on a degenerate response; callers fall back to ``[rep_id]``.
   */
  member_ids?: number[];
}

export interface TilePayload {
  level: number;
  tx: number;
  ty: number;
  cells: HexCellPayload[];
}

export interface LevelMeta {
  level: number;
  radius: number;
  n_cells: number;
}

export interface ProjectionMeta {
  projection_id: string;
  /** Which lattice this metadata's pyramid was binned with. */
  bin_shape?: BinShape;
  bounds: [number, number, number, number];
  base_radius: number;
  tile_span: number;
  point_count: number;
  levels: LevelMeta[];
  media_type?: string;
  /**
   * Membership version of this projection. 0 for full-dataset layouts; bumped
   * when items are removed from a subset browse in place. Combined with
   * ``projection_id`` to form the tile cache token, so an in-place edit busts
   * the immutable tile cache without changing the layout identity (which would
   * make the canvas re-frame the viewport).
   */
  content_version?: number;
  // Build lifecycle (idle | building | ready | error) and progress, populated
  // by GET /api/projection/meta while a build is in flight.
  status?: 'idle' | 'building' | 'ready' | 'error';
  current?: number;
  total?: number;
  message?: string;
  error?: string;
}

export interface ProjectionBuildResponse {
  status: string;
  projection_id?: string;
}

export interface ViewTransform {
  centerX: number;
  centerY: number;
  zoom: number;
}
