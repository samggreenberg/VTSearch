import type { ContextMenuItem } from '../context-menu/context-menu.component';

/**
 * Builds the right-click context-menu items for the dashboard's dataset and
 * detector rows.  Each item mirrors a button in the row's Actions column: same
 * icon, same label, same availability rules (Load only shows when the item is
 * unloaded; Edit-access only shows in multi-user mode and is disabled for
 * non-owners).  Keep this list and ordering in sync with the Actions column
 * markup in `dataset-card.component.html` / `detector-card.component.html`.
 */

const svg = (body: string): string =>
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
  'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
  'stroke-linejoin="round">' +
  body +
  '</svg>';

/** Action icons, matching the inline SVGs used by the Actions-column buttons. */
const ICON = {
  load: svg(
    '<path d="M13 3h5a3 3 0 0 1 3 3v12a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3v-5"/>' +
      '<polyline points="12 6 12 12 6 12"/><line x1="3" y1="3" x2="12" y2="12"/>',
  ),
  browse: svg(
    '<path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/>' +
      '<circle cx="12" cy="12" r="3"/>',
  ),
  security: svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
  rename: svg(
    '<path d="M16.5 2.5a2.828 2.828 0 0 1 4 4L7.5 19.5L3 20L3.5 15.5L16.5 2.5z"/>' +
      '<line x1="7.5" y1="19.5" x2="3.5" y2="15.5"/>' +
      '<line x1="14.5" y1="4.5" x2="18.5" y2="8.5"/>',
  ),
  addLabels: svg(
    '<path d="M22 10l-10-5L2 10l10 5 10-5z"/>' +
      '<path d="M6 12v5c0 1.66 2.69 3 6 3s6-1.34 6-3v-5"/>' +
      '<line x1="22" y1="10" x2="22" y2="16"/>',
  ),
  export: svg(
    '<path d="M11 3H6a3 3 0 0 0-3 3v12a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-5"/>' +
      '<polyline points="15 3 21 3 21 9"/><line x1="12" y1="12" x2="21" y2="3"/>',
  ),
  stats: svg(
    '<circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20"/>' +
      '<line x1="12" y1="12" x2="12" y2="2"/>' +
      '<line x1="12" y1="12" x2="20.66" y2="17"/>' +
      '<line x1="12" y1="12" x2="2" y2="12"/>',
  ),
  delete: svg(
    '<polyline points="3 6 5 6 21 6"/>' +
      '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>' +
      '<path d="M10 11v6"/><path d="M14 11v6"/>' +
      '<path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
  ),
};

export interface CardMenuAccess {
  /** True when running without real auth; access-control actions are hidden. */
  isDefaultLogin: boolean;
  /** True when the current user created the item; only owners may edit access. */
  isOwner: boolean;
}

function securityItem(access: CardMenuAccess): ContextMenuItem {
  return {
    id: 'security',
    label: 'Edit access list',
    title: access.isOwner
      ? 'Edit access list'
      : 'Only the creator can edit access',
    iconSvg: ICON.security,
    disabled: !access.isOwner,
  };
}

export function buildDatasetCardMenuItems(
  dataset: { loaded?: boolean } | undefined,
  access: CardMenuAccess,
): ContextMenuItem[] {
  const items: ContextMenuItem[] = [];
  if (!dataset?.loaded) {
    items.push({ id: 'load', label: 'Load dataset', title: 'Load dataset', iconSvg: ICON.load });
  }
  items.push({ id: 'browse', label: 'Browse dataset', title: 'Browse dataset', iconSvg: ICON.browse });
  if (!access.isDefaultLogin) {
    items.push(securityItem(access));
  }
  items.push({ id: 'rename', label: 'Rename', title: 'Rename', iconSvg: ICON.rename });
  items.push({ id: 'stats', label: 'Stats', title: 'Stats', iconSvg: ICON.stats });
  items.push({ id: 'delete', label: 'Delete', title: 'Delete', iconSvg: ICON.delete });
  return items;
}

export function buildDetectorCardMenuItems(
  detector: { detector_loaded?: boolean } | undefined,
  access: CardMenuAccess,
): ContextMenuItem[] {
  const items: ContextMenuItem[] = [];
  if (!detector?.detector_loaded) {
    items.push({ id: 'load', label: 'Load detector', title: 'Load detector', iconSvg: ICON.load });
  }
  items.push({ id: 'browse', label: 'Browse positives', title: "Browse this detector's positives", iconSvg: ICON.browse });
  if (!access.isDefaultLogin) {
    items.push(securityItem(access));
  }
  items.push({ id: 'rename', label: 'Rename', title: 'Rename', iconSvg: ICON.rename });
  items.push({ id: 'add-labels', label: 'Import Labels', title: 'Import Labels', iconSvg: ICON.addLabels });
  items.push({ id: 'export', label: 'Export', title: 'Export', iconSvg: ICON.export });
  items.push({ id: 'stats', label: 'Stats', title: 'Stats', iconSvg: ICON.stats });
  items.push({ id: 'delete', label: 'Delete', title: 'Delete', iconSvg: ICON.delete });
  return items;
}
