import type { ContextMenuItem } from '../context-menu/context-menu.component';

/**
 * Build the right-click context-menu items for a media item in
 * `vt-label-view`.  The crop variants are only emitted for media types
 * whose viewer can produce a sub-region selection (`audio` spectrograms
 * and `image` raster regions).  `showInPhoto` adds Show in photo (#4750),
 * for a dataset whose import also produced an Image dataset
 * (`SourcePhotoService.offered`).
 */
export function buildMediaContextMenuItems(
  mediaType: string,
  { showInPhoto = false }: { showInPhoto?: boolean } = {},
): ContextMenuItem[] {
  const cropAble = mediaType === 'audio' || mediaType === 'image';
  const items: ContextMenuItem[] = [
    {
      id: 'sort',
      label: 'Sort by similarity to this',
      title: 'Sort all loaded items by similarity to this item, using this item as the example.',
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
  if (showInPhoto) {
    items.push({
      id: 'show-in-photo',
      label: 'Show in photo',
      title: 'Open the photo this item was cut from, with the item outlined.',
    });
  }
  return items;
}
