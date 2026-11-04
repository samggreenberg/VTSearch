import { runBrowseIds, runBrowseSidesParam, subsetBrowseQueryParams } from './browse-subset.service';
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

  it('splits each detector at the notch of the Threshold it is given (#5540)', () => {
    const scored: AutoDetectResultsData = {
      results: {
        owl: {
          beta: 1,
          threshold: 0.5,
          lines: [
            { beta: 4, threshold: 0.1 },
            { beta: 1, threshold: 0.5 },
            { beta: 0.25, threshold: 0.9 },
          ],
          hits: [{ id: 1, score: 0.95 }, { id: 2, score: 0.6 }],
          negative_hits: [{ id: 3, score: 0.2 }],
        },
      },
    };
    expect(runBrowseIds(scored, 'good', 4)).toEqual([1, 2, 3]);
    expect(runBrowseIds(scored, 'bad', 0.25)).toEqual([2, 3]);
    expect(runBrowseIds(scored, 'good', null)).toEqual([1, 2]);
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

describe('subsetBrowseQueryParams', () => {
  it('names the notch only when the list was split at another than the run\'s (#5540)', () => {
    expect(subsetBrowseQueryParams({ kind: 'results', runId: 'r1', sides: 'good' })).toEqual({
      subset: 1,
      from: 'results',
      run: 'r1',
      sides: 'good',
    });
    expect(subsetBrowseQueryParams({ kind: 'results', runId: 'r1', sides: 'bad', notch: 0.25 })).toEqual({
      subset: 1,
      from: 'results',
      run: 'r1',
      sides: 'bad',
      notch: 0.25,
    });
    expect(subsetBrowseQueryParams({ kind: 'test' })).toEqual({ subset: 1 });
  });
});
