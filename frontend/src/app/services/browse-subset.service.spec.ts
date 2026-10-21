import { runBrowseIds, runBrowseSidesParam } from './browse-subset.service';
import type { AutoDetectResultsData } from '../models/api.models';

describe('runBrowseIds', () => {
  const run: AutoDetectResultsData = {
    results: {
      owl: { hits: [{ id: 1 }, { id: 2 }], negative_hits: [{ id: 3 }] },
      wren: { hits: [{ id: 2 }], negative_hits: [{ id: 1 }] },
    },
  };

  it('lays out the picked rows, a media two detectors share once', () => {
    expect(runBrowseIds(run, 'good')).toEqual([1, 2]);
    expect(runBrowseIds(run, 'bad')).toEqual([3, 1]);
    expect(runBrowseIds(run, 'both')).toEqual([1, 2, 3]);
  });

  it('skips rows with no media id', () => {
    expect(runBrowseIds({ results: { owl: { hits: [{ md5: 'a' }, { id: 4 }] } } }, 'good')).toEqual([4]);
  });
});

describe('runBrowseSidesParam', () => {
  // A link from before the param existed browsed Good, the dialog's default.
  it('reads Good when the param is absent or unknown', () => {
    expect(runBrowseSidesParam(null)).toBe('good');
    expect(runBrowseSidesParam('everything')).toBe('good');
    expect(runBrowseSidesParam('bad')).toBe('bad');
    expect(runBrowseSidesParam('both')).toBe('both');
  });
});
