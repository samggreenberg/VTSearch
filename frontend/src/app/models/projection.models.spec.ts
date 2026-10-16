import { readyLayout, type ProjectionMeta } from './projection.models';

describe('readyLayout', () => {
  const ready: ProjectionMeta = {
    status: 'ready',
    projection_id: 'p1',
    bin_shape: 'hex',
    bounds: [0, 0, 10, 10],
    base_radius: 1,
    tile_span: 8,
    point_count: 12,
    levels: [{ level: 0, radius: 1, n_cells: 4 }],
    content_version: 0,
  };

  it('narrows a ready meta to its layout, as the same object', () => {
    expect(readyLayout(ready)).toBe(ready);
  });

  it('is null with no meta, or while no layout is built', () => {
    expect(readyLayout(null)).toBeNull();
    expect(readyLayout(undefined)).toBeNull();
    expect(readyLayout({ status: 'idle' })).toBeNull();
    expect(readyLayout({ status: 'building', current: 3, total: 10 })).toBeNull();
    expect(readyLayout({ status: 'error', error: 'boom' })).toBeNull();
  });

  it('is null for a ready meta missing a layout field', () => {
    expect(readyLayout({ ...ready, bounds: null })).toBeNull();
    expect(readyLayout({ ...ready, bounds: [0, 0, 1] })).toBeNull();
    expect(readyLayout({ ...ready, levels: undefined })).toBeNull();
    expect(readyLayout({ ...ready, projection_id: null })).toBeNull();
  });
});
