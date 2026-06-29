import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { iconSizeToGoalWidth, snapPanelWidthToGridColumns } from './grid-icon-size';

describe('iconSizeToGoalWidth', () => {
  it('maps known sizes to their goal widths', () => {
    expect(iconSizeToGoalWidth('XS')).toBe(25);
    expect(iconSizeToGoalWidth('M')).toBe(80);
    expect(iconSizeToGoalWidth('XL')).toBe(200);
  });

  it('falls back to M for unknown sizes', () => {
    expect(iconSizeToGoalWidth('???')).toBe(80);
    expect(iconSizeToGoalWidth('')).toBe(80);
  });
});

describe('snapPanelWidthToGridColumns', () => {
  // jsdom does no layout, so the grid container's geometry is stubbed to a known
  // shape: a content box wide enough for exactly 3 columns of an 80px goal width
  // (3 * 84 - 4 = 248 content + 16 padding = 264 client), plus a 15px scrollbar.
  const GOAL = 80;
  const COLS = 3;
  const SCROLLBAR = 15;
  const contentWidth = COLS * (GOAL + 4) - 4; // 248
  const clientWidth = contentWidth + 16; // 264 (8px padding each side)
  const boundingWidth = clientWidth + SCROLLBAR; // 279

  let panel: HTMLElement;

  function makeGrid(className: string): HTMLElement {
    const grid = document.createElement('div');
    grid.className = className;
    grid.style.setProperty('--grid-goal-width', `${GOAL}px`);
    // Inline padding is reflected by jsdom's getComputedStyle.
    grid.style.paddingLeft = '8px';
    grid.style.paddingRight = '8px';
    // Stub the layout-derived geometry jsdom can't compute.
    Object.defineProperty(grid, 'clientWidth', { value: clientWidth, configurable: true });
    grid.getBoundingClientRect = () =>
      ({ width: boundingWidth, height: 0, top: 0, left: 0, right: 0, bottom: 0, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    return grid;
  }

  beforeEach(() => {
    panel = document.createElement('div');
    document.body.appendChild(panel);
  });

  afterEach(() => {
    panel.remove();
  });

  it('snaps a right-panel .vote-list (regression: no .grid-layout class)', () => {
    panel.appendChild(makeGrid('vote-list'));
    // currentPanelWidth carries a 20px structural offset over the grid bounding box.
    const snapped = snapPanelWidthToGridColumns(panel, boundingWidth + 20);
    expect(snapped).not.toBeNull();
    // Tight bounding width (279, already exactly 3 cols) + offset (20) + 1px margin.
    expect(snapped).toBe(300);
  });

  it('snaps a browse .bsp-list (regression: no .grid-layout class)', () => {
    panel.appendChild(makeGrid('bsp-list'));
    expect(snapPanelWidthToGridColumns(panel, boundingWidth)).not.toBeNull();
  });

  it('still snaps a left-panel .media-list.grid-layout', () => {
    panel.appendChild(makeGrid('media-list grid-layout'));
    expect(snapPanelWidthToGridColumns(panel, boundingWidth)).not.toBeNull();
  });

  it('returns null when the panel has no grid container', () => {
    panel.appendChild(document.createElement('div'));
    expect(snapPanelWidthToGridColumns(panel, 300)).toBeNull();
  });
});
