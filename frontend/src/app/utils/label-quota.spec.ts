import type { LabelQuota } from '../generated/api-client/models/label-quota';
import { centroidNote, labelsOwed } from './label-quota';

function quota(over: Partial<LabelQuota> = {}): LabelQuota {
  return {
    tier: 'centroid',
    n_good: 1,
    n_bad: 0,
    goods_owed: 2,
    bads_owed: 4,
    good_quota: 3,
    bad_quota: 4,
    dry_bad_quota: 16,
    ...over,
  };
}

describe('label-quota (#4643, #4731)', () => {
  describe('labelsOwed', () => {
    it('names both classes, singular and plural', () => {
      expect(labelsOwed({ goods_owed: 2, bads_owed: 1 })).toBe('2 more Goods and 1 more Bad');
    });

    it('leaves out a class that is met', () => {
      expect(labelsOwed({ goods_owed: 0, bads_owed: 3 })).toBe('3 more Bads');
      expect(labelsOwed({ goods_owed: 1, bads_owed: 0 })).toBe('1 more Good');
      expect(labelsOwed({ goods_owed: 0, bads_owed: 0 })).toBe('');
    });
  });

  describe('centroidNote', () => {
    it('says nothing for a trained detector or before a pass', () => {
      expect(centroidNote(null)).toBe('');
      expect(centroidNote(undefined)).toBe('');
      expect(centroidNote(quota({ tier: 'trained', goods_owed: 0, bads_owed: 0 }))).toBe('');
    });

    it("says the detector is the Goods' centroid, the quota, and what is owed", () => {
      const note = centroidNote(quota());
      expect(note).toContain("Goods' centroid");
      expect(note).toContain('3 Goods and 4 Bads, or a Good and 16 Bads');
      expect(note).toContain('2 more Goods and 4 more Bads, or 16 more Bads in Train');
      expect(note).toContain('the Threshold moves it');
    });

    it('offers the second quota, Bads alone, once a Good is in hand (#4731)', () => {
      const note = centroidNote(quota({ n_good: 2, n_bad: 10, goods_owed: 1, bads_owed: 0 }));
      expect(note).toContain('1 more Good, or 6 more Bads in Train give a trained one.');
    });

    it('owes nothing when only an unresolved label kept the trained detector away', () => {
      const note = centroidNote(quota({ n_good: 3, n_bad: 4, goods_owed: 0, bads_owed: 0 }));
      expect(note).toContain("Goods' centroid");
      expect(note).not.toContain('more');
    });
  });
});
