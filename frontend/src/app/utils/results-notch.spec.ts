import type { AutoDetectDetectorResult, AutoDetectResultsData } from '../models/api.models';
import { cutAt, notchMoves, notchParam, ownNotch, runNotch, splitAtNotch } from './results-notch';

const lines = (four: number, one: number, quarter: number) => [
  { beta: 4, threshold: four },
  { beta: 1, threshold: one },
  { beta: 0.25, threshold: quarter },
];

const cats: AutoDetectDetectorResult = {
  threshold: 0.5,
  beta: 1,
  lines: lines(0.3, 0.5, 0.9),
  hits: [
    { id: 1, score: 0.95 },
    { id: 2, score: 0.6 },
  ],
  negative_hits: [
    { id: 3, score: 0.4 },
    { id: 4, score: 0.1 },
  ],
};

const ids = (hits: { id?: number }[]) => hits.map((h) => h.id);

describe('splitAtNotch', () => {
  it("splits the run's scores at another notch's cut, highest first", () => {
    const toward = splitAtNotch(cats, 4);
    expect([ids(toward.good), ids(toward.bad)]).toEqual([[1, 2, 3], [4]]);
    expect([toward.threshold, toward.moved]).toEqual([0.3, true]);

    const away = splitAtNotch(cats, 0.25);
    expect([ids(away.good), ids(away.bad)]).toEqual([[1], [2, 3, 4]]);
  });

  it("keeps the run's own split at its own notch, or with none picked", () => {
    for (const notch of [1, null]) {
      const split = splitAtNotch(cats, notch);
      expect(split.good).toBe(cats.hits);
      expect(split.bad).toBe(cats.negative_hits);
      expect([split.threshold, split.moved]).toEqual([0.5, false]);
    }
  });

  it('keeps the run\'s split where it carries no cut at the notch', () => {
    const { lines: _none, ...bare } = cats;
    expect(splitAtNotch(bare, 4).moved).toBe(false);
  });

  it('counts an item scored exactly at the cut as Good, and one with no score as Bad', () => {
    const edge: AutoDetectDetectorResult = { ...cats, hits: [{ id: 1, score: 0.3 }], negative_hits: [{ id: 2 }] };
    const split = splitAtNotch(edge, 4);
    expect([ids(split.good), ids(split.bad)]).toEqual([[1], [2]]);
  });
});

describe('ownNotch and cutAt', () => {
  it('reads a balance off the presets as the notch the Threshold shows it at', () => {
    expect(ownNotch(cats)).toBe(1);
    expect(ownNotch({ ...cats, beta: 2.5 })).toBe(4);
    expect(ownNotch({ ...cats, beta: null })).toBeNull();
    expect(cutAt(cats, 0.25)).toBe(0.9);
    expect(cutAt({ hits: [] }, 4)).toBeNull();
  });
});

describe('runNotch and notchMoves', () => {
  const run = (results: Record<string, AutoDetectDetectorResult>): AutoDetectResultsData => ({ results });

  it("is the notch every detector shares, or none when they differ", () => {
    expect(runNotch(run({ cats, dogs: { ...cats, beta: 1 } }))).toBe(1);
    expect(runNotch(run({ cats, dogs: { ...cats, beta: 4 } }))).toBeNull();
  });

  it('moves only when some detector carries cuts that differ', () => {
    expect(notchMoves(run({ cats }))).toBe(true);
    expect(notchMoves(run({ cats: { ...cats, lines: lines(0.5, 0.5, 0.5) } }))).toBe(false);
    expect(notchMoves(run({ cats: { hits: [] } }))).toBe(false);
  });
});

describe('notchParam', () => {
  it('reads a preset back, and anything else as the run\'s own', () => {
    expect(notchParam('4')).toBe(4);
    expect(notchParam('0.25')).toBe(0.25);
    expect([notchParam(null), notchParam(''), notchParam('2'), notchParam('x')]).toEqual([null, null, null, null]);
  });
});
