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

/**
 * One region signpost — a named place on the projection map (see
 * `docs/plans/vtsbrowse-toponymy.md`). The canvas draws it as a text pill
 * anchored at `(x, y)` when the view's zoom is near the sign's `level`.
 */
export interface RegionLabelPayload {
  /**
   * Pyramid zoom level the sign belongs to: 0 = the coarsest layer (continent
   * names), deeper = finer (countries, then states). May be fractional — the
   * canvas interpolates visibility/size on a continuous level axis.
   */
  level: number;
  /** Anchor in projection space (the frozen 2-D layout's coordinates). */
  x: number;
  y: number;
  text: string;
  /** Naming confidence, used as the de-clutter tiebreak (higher wins). */
  score?: number;
  /** Which namer produced the sign (e.g. "keyphrase", "llm"). */
  source?: string;
  /**
   * Whether a coarser sign names this region one zoom band out (a parent in the
   * topic tree). `false` marks a **root** region: the canvas skips the coarse-
   * edge fade-in and keeps the sign visible when zoomed out, so the region
   * isn't left nameless with nothing coarser covering it. Absent ⇒ `true`
   * (treated as having a parent — the pre-flag fading behaviour).
   */
  has_coarser?: boolean;
  /**
   * Whether a finer sign names this region one zoom band in (a child in the
   * topic tree). `false` marks a **leaf** region: the canvas skips the fine-
   * edge fade-out and keeps the sign visible as you zoom in, so an on-screen
   * island's only name doesn't expire with nothing finer to hand off to.
   * Absent ⇒ `true`.
   */
  has_finer?: boolean;
}

/** Response of ``GET /api/projection/labels``. */
export interface ProjectionLabelsResponse {
  /** "ready" when a projection exists (labels may still be empty), "idle" otherwise. */
  status: string;
  projection_id?: string;
  labels: RegionLabelPayload[];
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
   * Whether region signpost labels exist for this projection (see
   * ``GET /api/projection/labels``). Absent/false until a labeler has run,
   * so the client can skip the labels fetch entirely.
   */
  has_labels?: boolean;
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
