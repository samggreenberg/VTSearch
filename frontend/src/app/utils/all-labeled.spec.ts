import { allItemsLabeled } from './all-labeled';

/**
 * The "is this list finished?" rule, shared by the centre pane's
 * dataset-exhausted message and the left panel's "all labeled" chip (#4028).
 */
describe('allItemsLabeled', () => {
  const items = (...ids: number[]) => ids.map((id) => ({ id }));

  it('is false for an empty list — nothing loaded is not nothing left', () => {
    expect(allItemsLabeled([], new Set([1]), new Set())).toBe(false);
  });

  it('is false while one item is still unlabeled', () => {
    expect(allItemsLabeled(items(1, 2, 3), new Set([1]), new Set([2]))).toBe(false);
  });

  it('is true once every item carries a label, of either polarity', () => {
    expect(allItemsLabeled(items(1, 2, 3), new Set([1, 3]), new Set([2]))).toBe(true);
  });

  /**
   * The size comparison is only a cheap reject. A labelset carries elements
   * from wherever they were labeled, so a vote set can be larger than this
   * dataset while still missing one of its items — the scan is what answers.
   */
  it('is false when the votes are plentiful but name other items', () => {
    const good = new Set([1, 90, 91, 92]);
    expect(allItemsLabeled(items(1, 2), good, new Set())).toBe(false);
  });
});
