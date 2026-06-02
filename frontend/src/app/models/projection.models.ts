export interface HexCellPayload {
  q: number;
  r: number;
  cx: number;
  cy: number;
  count: number;
  rep_id: number;
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
  bounds: [number, number, number, number];
  base_radius: number;
  tile_span: number;
  point_count: number;
  levels: LevelMeta[];
  media_type?: string;
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
