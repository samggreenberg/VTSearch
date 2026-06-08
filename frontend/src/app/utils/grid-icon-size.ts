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
 * Finds the active grid container (.media-list.grid-layout or .vote-list.grid-layout)
 * inside panelEl, reads the current column count from the live DOM (so scrollbar width
 * and all structural offsets are automatically accounted for), and returns the snapped
 * panel width. Returns null if the panel is not in grid mode or the element is not found.
 */
export function snapPanelWidthToGridColumns(panelEl: HTMLElement, currentPanelWidth: number): number | null {
  const gridEl = (
    panelEl.querySelector('.media-list.grid-layout') ??
    panelEl.querySelector('.vote-list.grid-layout') ??
    panelEl.querySelector('.bsp-list.grid-layout')
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
