import type { MediaContextMenuItem } from '../left-panel/media-item/media-context-menu.component';

/**
 * Build the right-click context-menu items for a media item in
 * `vt-label-view`.  The crop variants are only emitted for media types
 * whose viewer can produce a sub-region selection (`audio` spectrograms
 * and `image` raster regions).
 */
export function buildMediaContextMenuItems(mediaType: string): MediaContextMenuItem[] {
  const cropAble = mediaType === 'audio' || mediaType === 'image';
  const items: MediaContextMenuItem[] = [
    {
      id: 'sort',
      label: 'Sort by similarity to this',
      title: 'Sort all loaded items by similarity to this item, using its existing embedding.',
    },
  ];
  if (cropAble) {
    items.push({
      id: 'crop-sort',
      label: 'Crop, then sort by similarity…',
      title: 'Open the crop tool to pick a sub-region, then sort by similarity.',
    });
  }
  items.push({
    id: 'seed',
    label: 'Use as detector seed',
    title: 'Open the New Detector form with this item pre-selected as the example.',
  });
  if (cropAble) {
    items.push({
      id: 'crop-seed',
      label: 'Crop, then use as detector seed…',
      title: 'Open the crop tool to pick a sub-region, then seed a new detector.',
    });
  }
  return items;
}
