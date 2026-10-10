import { describe, expect, it } from 'vitest';

import { buildMediaContextMenuItems } from './media-context-menu-items';

describe('buildMediaContextMenuItems', () => {
  it('offers the crop variants only where a viewer can crop', () => {
    expect(buildMediaContextMenuItems('image').map((i) => i.id)).toEqual(['sort', 'crop-sort', 'seed', 'crop-seed']);
    expect(buildMediaContextMenuItems('text').map((i) => i.id)).toEqual(['sort', 'seed']);
  });

  // #4750: on a dataset whose import also produced an Image dataset.
  it('adds Show in photo last when asked', () => {
    const items = buildMediaContextMenuItems('face', { showInPhoto: true });
    expect(items.map((i) => i.id)).toEqual(['sort', 'seed', 'show-in-photo']);
    expect(items[2].label).toBe('Show in photo');
  });
});
