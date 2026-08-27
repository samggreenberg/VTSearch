import type { ContextMenuItem } from '../context-menu/context-menu.component';

/**
 * Builds the action menu for the dashboard's dataset and detector rows. The
 * full list backs the right-click context menu, where being complete is the
 * point. The Actions column renders only the two universal verbs inline (Load
 * when unloaded, Delete); the ⋯ overflow button reuses this same list but drops
 * those inline verbs (see ``overflowMenuItems``) so it reads as "more" — Browse,
 * Rename, Stats, and detector-only Import Labels / Export labels, plus
 * Edit-access in multi-user mode — rather than repeating icons already visible
 * in the row.
 * Rename is additionally surfaced as a pencil next to the row name. Availability
 * rules: Load only shows when the item is unloaded; Edit-access only shows in
 * multi-user mode and is disabled for non-owners.
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
  lock: svg(
    '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>' +
      '<path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  ),
  unlock: svg(
    '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>' +
      '<path d="M7 11V7a5 5 0 0 1 9.9-1"/>',
  ),
};

/**
 * Min width of the shared context menu, mirroring `.context-menu`'s `min-width`
 * in `context-menu.component.scss`. Used to right-align the ⋯ overflow menu
 * under its button so a menu opened at the rightmost Actions column never
 * spills off the viewport's right edge.
 */
export const CARD_MENU_MIN_WIDTH = 200;

/**
 * Action ids the Actions column already renders as inline icon buttons (Load
 * only when unloaded, Delete). ``overflowMenuItems`` strips these from the ⋯
 * overflow menu so it shows only what isn't already one click away in the row.
 * The right-click context menu keeps the complete list.
 */
const INLINE_CARD_ACTION_IDS: ReadonlySet<string> = new Set(['load', 'delete']);

/** The ⋯ overflow subset of a card menu: the full list minus the inline verbs. */
export function overflowMenuItems(items: ContextMenuItem[]): ContextMenuItem[] {
  return items.filter((item) => !INLINE_CARD_ACTION_IDS.has(item.id));
}

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

/**
 * Detector menu. An AutoRun detector (``autofind``) is frozen: the editing
 * verbs (Rename, Import Labels, Delete) are omitted entirely — the only way
 * to change it is "Move to Drafts" first — while the read/use verbs (Load,
 * Browse, Export labels, Stats) stay. A draft detector instead offers
 * "Move to AutoRun" to finalize it.
 */
export function buildDetectorCardMenuItems(
  detector: { detector_loaded?: boolean; autofind?: boolean } | undefined,
  access: CardMenuAccess,
): ContextMenuItem[] {
  const frozen = !!detector?.autofind;
  const items: ContextMenuItem[] = [];
  if (!detector?.detector_loaded) {
    items.push({ id: 'load', label: 'Load detector', title: 'Load detector', iconSvg: ICON.load });
  }
  items.push({ id: 'browse', label: 'Browse positives', title: "Browse this detector's positives", iconSvg: ICON.browse });
  if (!access.isDefaultLogin) {
    items.push(securityItem(access));
  }
  if (!frozen) {
    items.push({ id: 'rename', label: 'Rename', title: 'Rename', iconSvg: ICON.rename });
    items.push({ id: 'add-labels', label: 'Import Labels', title: 'Import Labels', iconSvg: ICON.addLabels });
  }
  // Exporting the detector's *labels* is the only export offered here. The
  // portable ONNX bundle (POST /api/detectors/<id>/portable-bundle) is a
  // deliberate expert affordance called directly against the API, not a menu
  // item: it sat one line below this one under a near-identical name and icon,
  // and the two are not variants of one action — one moves the detector between
  // VTSearch instances, the other hands a frozen scorer to someone who does not
  // run VTSearch at all.
  items.push({
    id: 'export',
    label: 'Export labels',
    title: "Export this detector's labeled items; the full set is what re-imports as the detector",
    iconSvg: ICON.export,
  });
  items.push({ id: 'stats', label: 'Stats', title: 'Stats', iconSvg: ICON.stats });
  if (frozen) {
    items.push({
      id: 'move-to-drafts',
      label: 'Move to Drafts',
      title: 'Unfreeze this detector: stop auto-running it and allow editing again',
      iconSvg: ICON.unlock,
    });
  } else {
    items.push({
      id: 'move-to-autorun',
      label: 'Move to AutoRun',
      title: 'Finalize this detector: freeze it against edits and auto-run it on every dataset as it is imported',
      iconSvg: ICON.lock,
    });
    items.push({ id: 'delete', label: 'Delete', title: 'Delete', iconSvg: ICON.delete });
  }
  return items;
}
