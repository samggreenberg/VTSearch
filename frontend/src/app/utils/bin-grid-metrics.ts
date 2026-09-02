/**
 * Geometry shared by the bin details' member grid and the floating popup shell
 * that positions itself around it.
 *
 * These live outside both components because each needs the same numbers for a
 * different reason: {@link BrowseBinMemberGridComponent} lays the grid out and
 * drives its virtual viewport with them, while ``BrowseBinPopupComponent``
 * models the rendered popup's height/width from them when it clamps itself onto
 * the visible canvas. A private copy in either place would let the clamp drift
 * from what the grid actually renders — which shows up as a popup whose floor
 * sits just off-screen, the exact class of bug the clamp exists to prevent.
 */

/** Vertical padding (px) inside every grid cell (``.bin-popup-entry`` 2px top +
 *  2px bottom), always present regardless of icon size. Reserved in {@link
 *  binGridRowSize} so the cell's real rendered height never exceeds its virtual
 *  slot. */
export const GRID_CELL_PADDING = 4;

/** Gap (px) between grid cells (and grid rows); matches ``--space-2xs``-ish. */
export const GRID_GAP = 4;

/** Width (px) available to lay out cells inside the popup's scroll column (≈ its
 *  width minus padding and the scrollbar). Columns are derived from this. */
export const GRID_CONTENT_WIDTH = 256;

/** Width (px) of the scrolling grid column; mirrors the historic popup width. */
export const GRID_COLUMN_WIDTH = 280;

/** Vertical room (px) the member-count label takes above the scrolling grid. */
export const COUNT_LABEL_HEIGHT = 22;

/**
 * Pixel stride of one virtual grid row: the thumbnail plus its always-present
 * vertical cell padding and an inter-row gap. The grid prints no name under each
 * thumbnail, so only the thumbnail and padding contribute; accounting for the
 * cell padding keeps the row's rendered content from overflowing its virtual
 * slot by a sub-pixel, which would otherwise force a stray scrollbar on even a
 * single row.
 */
export function binGridRowSize(cellWidth: number): number {
  return cellWidth + GRID_CELL_PADDING + GRID_GAP;
}

/**
 * How many fixed-width cells fit across ``contentWidth``. At least one, so a
 * panel narrower than a single thumbnail still chunks into one-item rows rather
 * than dividing by zero.
 */
export function binGridColumns(contentWidth: number, cellWidth: number): number {
  return Math.max(1, Math.floor((contentWidth + GRID_GAP) / (cellWidth + GRID_GAP)));
}
