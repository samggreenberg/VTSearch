import { describe, expect, it } from 'vitest';
import { clusterImportSiblings, importSiblingNames } from './import-siblings';

// Already sorted by name, as the Dashboard hands them over: "Pets" falls
// between the two "Photos" siblings, and a third dataset joins the group from
// the far end of the list.
const rows = [
  { id: 'a', name: 'Birds', import_group: null },
  { id: 'b', name: 'Photos – Face', import_group: 'g1' },
  { id: 'c', name: 'Pets' },
  { id: 'd', name: 'Photos – Image', import_group: 'g1' },
  { id: 'e', name: 'Tracks – Audio', import_group: 'g2' },
  { id: 'f', name: 'Zebra pages', import_group: 'g1' },
];

describe('clusterImportSiblings', () => {
  it('pulls each group to where its first member stood, keeping member order', () => {
    expect(clusterImportSiblings(rows).map((r) => r.id)).toEqual(['a', 'b', 'd', 'f', 'c', 'e']);
  });

  it('leaves a list with no groups in its order, as a copy', () => {
    const plain = [{ name: 'b' }, { name: 'a' }];
    const out = clusterImportSiblings(plain);
    expect(out).toEqual(plain);
    expect(out).not.toBe(plain);
  });
});

describe('importSiblingNames', () => {
  it('names every other member of a row\'s group', () => {
    const names = importSiblingNames(rows);
    expect(names.get('b')).toEqual(['Photos – Image', 'Zebra pages']);
    expect(names.get('d')).toEqual(['Photos – Face', 'Zebra pages']);
  });

  it('has no entry for an ungrouped row or a group of one', () => {
    const names = importSiblingNames(rows);
    expect(names.has('a')).toBe(false);
    expect(names.has('c')).toBe(false);
    expect(names.has('e')).toBe(false);
  });
});
