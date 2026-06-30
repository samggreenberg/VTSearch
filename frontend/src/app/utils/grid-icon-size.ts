const ICON_SIZE_GOAL_WIDTH: Record<string, number> = {
  XS: 25,
  S: 50,
  M: 80,
  L: 130,
  XL: 200,
};

export function iconSizeToGoalWidth(size: string): number {
  return ICON_SIZE_GOAL_WIDTH[size] ?? ICON_SIZE_GOAL_WIDTH['M'];
}

const GRID_GAP = 4;

/**
 * When the user releases the panel resize divider, snap the panel width down to
 * the minimum width that still shows the same number of grid columns.
 *
 * Finds the active grid container (.media-list.grid-layout, the virtualized
 * .media-list.grid-virtual viewport, .vote-list, or .bsp-list) inside panelEl, reads the
 * current column count from the live DOM (so scrollbar width and all structural offsets
 * are automatically accounted for), and returns the snapped panel width. Returns null if
 * no grid container is found.
 *
 * The right panel's `.vote-list` and the browse panel's `.bsp-list` are always grids
 * (the old list view mode was removed), so they carry no `.grid-layout` marker class;
 * they are matched by their base class alone. The left `.media-list` still carries
 * `.grid-layout` (and `.grid-virtual` when virtualized) because that selector also has
 * to exclude its skeleton/non-grid states.
 *
 * The left panel switches to a CDK virtual-scroll viewport (.grid-virtual) once a view
 * grows past its grid-virtualization threshold; that viewport carries the same
 * --grid-goal-width and padding-inline as the plain grid, so the column math below is
 * identical and snapping works in both modes.
 */
export function snapPanelWidthToGridColumns(panelEl: HTMLElement, currentPanelWidth: number): number | null {
  const gridEl = (
    panelEl.querySelector('.media-list.grid-layout') ??
    panelEl.querySelector('.media-list.grid-virtual') ??
    panelEl.querySelector('.vote-list') ??
    panelEl.querySelector('.bsp-list')
  ) as HTMLElement | null;
  if (!gridEl) return null;

  const goalWidthStr = gridEl.style.getPropertyValue('--grid-goal-width');
  const goalWidth = parseFloat(goalWidthStr);
  if (!goalWidth || isNaN(goalWidth)) return null;

  // clientWidth excludes the scrollbar but includes padding
  const clientWidth = gridEl.clientWidth;
  if (clientWidth <= 0) return null;

  const style = window.getComputedStyle(gridEl);
  const paddingH = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);

  const contentWidth = clientWidth - paddingH;
  if (contentWidth <= 0) return null;

  const cols = Math.floor((contentWidth + GRID_GAP) / (goalWidth + GRID_GAP));
  if (cols <= 0) return null;

  // Minimum content width to still show `cols` columns
  const minContentWidth = cols * (goalWidth + GRID_GAP) - GRID_GAP;
  const minClientWidth = minContentWidth + paddingH;

  // Re-add scrollbar width (getBoundingClientRect includes it, clientWidth does not)
  const boundingWidth = gridEl.getBoundingClientRect().width;
  const scrollbarWidth = Math.max(0, boundingWidth - clientWidth);
  const minBoundingWidth = minClientWidth + scrollbarWidth;

  // Preserve any structural offset between the panel edge and the grid element edge
  const offset = currentPanelWidth - boundingWidth;

  // Round up and add a 1px safety margin so sub-pixel layout rounding can't drop
  // the snapped panel width below the threshold for `cols` columns (which would
  // leave the user with cols-1 icons and an empty gap).
  return Math.ceil(minBoundingWidth + offset) + 1;
}
