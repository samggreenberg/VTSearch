import type { LabelQuota } from '../generated/api-client/models/label-quota';
import { centroidNote, labelsOwed, queryNote, quotaNote, typedQuery } from './label-quota';

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
    hold_labels: 30,
    hold_owed: 0,
    ...over,
  };
}

describe('label-quota (#4643, #4731, #4758, #4889)', () => {
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
      expect(centroidNote(quota({ tier: 'query' }))).toBe('');
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

  describe('queryNote (#4758)', () => {
    it('says nothing for any other detector', () => {
      expect(queryNote(null)).toBe('');
      expect(queryNote(quota())).toBe('');
      expect(queryNote(quota({ tier: 'trained', goods_owed: 0, bads_owed: 0 }))).toBe('');
    });

    it("says the detector is its typed query's own set, the quota, and what is owed", () => {
      const note = queryNote(quota({ tier: 'query' }));
      expect(note).toContain("the detector's typed query, keeping the images its text sort keeps in Train");
      expect(note).toContain('3 Goods and 4 Bads, or a Good and 16 Bads');
      expect(note).toContain('2 more Goods and 4 more Bads, or 16 more Bads in Train give a trained one.');
      expect(note).toContain('the Threshold moves it');
      expect(note).not.toContain('centroid');
    });

    it('owes the first quota only, before the first Good', () => {
      const note = queryNote(quota({ tier: 'query', n_good: 0, n_bad: 0, goods_owed: 3, bads_owed: 4 }));
      expect(note).toContain(' 3 more Goods and 4 more Bads in Train give a trained one.');
    });

    it('past the quota, names the hold rather than too few labels (#4736, #4889)', () => {
      const note = queryNote(
        quota({ tier: 'query', n_good: 12, n_bad: 15, goods_owed: 0, bads_owed: 0, hold_owed: 3 }),
      );
      expect(note).not.toContain('Too few labels');
      expect(note).toContain('keeps its query\'s set until it has 30 labels (this one has 27)');
      expect(note).toContain("the detector's typed query, keeping the images its text sort keeps in Train");
      expect(note).toContain('3 more labels, Good or Bad, in Train give a trained one.');
      expect(quotaNote(quota({ tier: 'query', n_good: 12, n_bad: 17, goods_owed: 0, bads_owed: 0, hold_owed: 1 }))).toContain(
        '1 more label, Good or Bad,',
      );
    });
  });

  describe('quotaNote', () => {
    it('is the note for whichever detector the quota gave', () => {
      expect(quotaNote(quota({ tier: 'query' }))).toBe(queryNote(quota({ tier: 'query' })));
      expect(quotaNote(quota())).toBe(centroidNote(quota()));
      expect(quotaNote(quota({ tier: 'trained', goods_owed: 0, bads_owed: 0 }))).toBe('');
      expect(quotaNote(null)).toBe('');
    });
  });

  describe('typedQuery (#4758)', () => {
    it("is the detector's text query, else its first text example, as the server reads it", () => {
      expect(typedQuery({ text_query: '  a dog ' })).toBe('a dog');
      expect(
        typedQuery({
          text_query: '',
          examples: [
            { type: 'media', value: 'x.png' },
            { type: 'text', value: 'a cat' },
          ],
        }),
      ).toBe('a cat');
      expect(typedQuery({ text_query: '', examples: [{ type: 'media', value: 'x.png' }] })).toBe('');
      expect(typedQuery({})).toBe('');
      expect(typedQuery({ text_query: null, examples: null })).toBe('');
    });
  });
});
