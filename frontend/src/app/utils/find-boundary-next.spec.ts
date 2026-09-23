import { type BoundaryPick, peekBoundaryQueue, pickBoundaryNext } from './find-boundary-next';

/**
 * The Find boundary walk as a pure rule (#3896). Pulled out of
 * `FindViewComponent.advanceToBoundary` so the image prefetch can ask what the
 * next advances will be through the same rule that takes them.
 */
describe('find boundary walk', () => {
  // Descending by score; the cutoff at 0.5 sits between ids 3 and 4.
  const order = [
    { id: 1, score: 0.9 },
    { id: 2, score: 0.7 },
    { id: 3, score: 0.55 },
    { id: 4, score: 0.45 },
    { id: 5, score: 0.3 },
    { id: 6, score: 0.1 },
  ];
  const none = (): boolean => false;

  describe('pickBoundaryNext', () => {
    it('takes the nearest item above the cut when serving `above`', () => {
      expect(pickBoundaryNext(order, 0.5, none, 'above')).toEqual({ id: 3, took: 'above' });
    });

    it('takes the nearest item below the cut when serving `below`', () => {
      expect(pickBoundaryNext(order, 0.5, none, 'below')).toEqual({ id: 4, took: 'below' });
    });

    it('skips verified items', () => {
      const verified = new Set([3, 4]);
      expect(pickBoundaryNext(order, 0.5, (id) => verified.has(id), 'above')).toEqual({
        id: 2,
        took: 'above',
      });
      expect(pickBoundaryNext(order, 0.5, (id) => verified.has(id), 'below')).toEqual({
        id: 5,
        took: 'below',
      });
    });

    it('falls back to the other side when the served one is exhausted', () => {
      const above = new Set([1, 2, 3]);
      expect(pickBoundaryNext(order, 0.5, (id) => above.has(id), 'above')).toEqual({
        id: 4,
        took: 'below',
      });
    });

    it('is null when both sides are verified — the done state', () => {
      expect(pickBoundaryNext(order, 0.5, () => true, 'above')).toBeNull();
    });
  });

  describe('peekBoundaryQueue', () => {
    it('alternates sides past the item on screen', () => {
      // On screen: 3 (the walk just took `above`, so `below` is served next).
      expect(peekBoundaryQueue(order, 0.5, new Set(), 'below', 3, 2)).toEqual([4, 2]);
    });

    it('matches walking the rule one advance at a time', () => {
      const verified = new Set<number>([3]);
      const predicted = peekBoundaryQueue(order, 0.5, verified, 'below', 3, 3);
      const walked: number[] = [];
      let side: 'above' | 'below' = 'below';
      for (let i = 0; i < 3; i++) {
        const pick: BoundaryPick = pickBoundaryNext(order, 0.5, (id) => verified.has(id), side)!;
        walked.push(pick.id);
        verified.add(pick.id);
        side = pick.took === 'above' ? 'below' : 'above';
      }
      expect(predicted).toEqual(walked);
    });

    it('treats the item on screen as verified even before its vote lands', () => {
      expect(peekBoundaryQueue(order, 0.5, new Set(), 'above', 3, 1)).toEqual([2]);
    });

    it('is empty before a score or with nothing on screen', () => {
      expect(peekBoundaryQueue(null, 0.5, new Set(), 'above', 3, 2)).toEqual([]);
      expect(peekBoundaryQueue(order, null, new Set(), 'above', 3, 2)).toEqual([]);
      expect(peekBoundaryQueue(order, 0.5, new Set(), 'above', null, 2)).toEqual([]);
    });

    it('stops short when the walk runs out', () => {
      const allBut = new Set([1, 2, 4, 5, 6]);
      expect(peekBoundaryQueue(order, 0.5, allBut, 'above', 3, 2)).toEqual([]);
    });
  });
});
