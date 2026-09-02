import { describe, expect, it } from 'vitest';
import { ListSortMode, SortableListEntry, sortListEntries } from './sort-list-entries';

const entries: SortableListEntry[] = [
  { id: 10, name: 'beta', time: 3, confidence: 0.1 },
  { id: 2, name: 'Alpha', time: 1, confidence: 0.9 },
  { id: 9, name: 'gamma', time: 2, confidence: -1 },
];

const idsFor = (mode: ListSortMode) => sortListEntries(entries, mode).map((e) => e.id);

describe('sortListEntries', () => {
  it('orders by each mode', () => {
    expect(idsFor('time-desc')).toEqual([10, 9, 2]);
    expect(idsFor('time-asc')).toEqual([2, 9, 10]);
    expect(idsFor('name-asc')).toEqual([2, 10, 9]);
    expect(idsFor('name-desc')).toEqual([9, 10, 2]);
    expect(idsFor('confidence-desc')).toEqual([2, 10, 9]);
    expect(idsFor('confidence-asc')).toEqual([9, 10, 2]);
  });

  it('compares numeric ids numerically, not lexically', () => {
    // The bug a String() compare would reintroduce: 10 sorting before 2.
    expect(idsFor('id-asc')).toEqual([2, 9, 10]);
  });

  it('compares string ids as strings', () => {
    const strIds = [
      { id: 'lbl-c', name: 'c', time: 0 },
      { id: 'lbl-a', name: 'a', time: 0 },
    ];
    expect(sortListEntries(strIds, 'id-asc').map((e) => e.id)).toEqual(['lbl-a', 'lbl-c']);
  });

  it('treats a missing confidence as the unscored sentinel', () => {
    const mixed = [
      { id: 1, name: 'a', time: 0 },
      { id: 2, name: 'b', time: 0, confidence: 0.5 },
    ];
    expect(sortListEntries(mixed, 'confidence-desc').map((e) => e.id)).toEqual([2, 1]);
  });

  it('returns a new array and leaves the input untouched', () => {
    const input = [...entries];
    const out = sortListEntries(input, 'name-asc');
    expect(out).not.toBe(input);
    expect(input.map((e) => e.id)).toEqual([10, 2, 9]);
  });
});
