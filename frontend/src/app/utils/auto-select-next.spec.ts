import { describe, expect, it } from 'vitest';

import { autoSelectNext } from './auto-select-next';
import type { AutoSelectInput } from './auto-select-next';
import type { SortedItem } from '../services/sort-state.service';

/**
 * Unit coverage for the auto-advance pick rule extracted out of
 * `LabelViewComponent.autoSelectNext`. The Python eval harness reproduces this
 * rule (`vtscore/eval/al_strategies._hard_pick_by_index`) and
 * `scripts/check-eval-app-sync.py` pins it, so these cases double as the
 * executable statement of what the harness has to keep matching — in
 * particular that `hard` measures distance in **rank space**, not score space.
 */
describe('autoSelectNext', () => {
  function rank(scores: number[]): SortedItem[] {
    return scores.map((score, i) => ({ id: i + 1, score }));
  }

  function input(partial: Partial<AutoSelectInput> = {}): AutoSelectInput {
    return {
      sortOrder: rank([0.9, 0.8, 0.7, 0.6, 0.5]),
      selectMode: 'top',
      acqThreshold: null,
      goodVotes: new Set<number>(),
      badVotes: new Set<number>(),
      ...partial,
    };
  }

  describe('no ranking loaded', () => {
    it('picks nothing when the sort order is null', () => {
      expect(autoSelectNext(input({ sortOrder: null }))).toEqual({ kind: 'none' });
    });

    it('picks nothing when the sort order is empty', () => {
      expect(autoSelectNext(input({ sortOrder: [] }))).toEqual({ kind: 'none' });
    });

    it('does not fire a diversity probe without a ranking', () => {
      // The `new` mode still needs a loaded sort: the atlas probe is steered by
      // the current scores and the acquisition cut.
      expect(autoSelectNext(input({ sortOrder: null, selectMode: 'new' }))).toEqual({
        kind: 'none',
      });
    });
  });

  describe('top', () => {
    it('takes the highest-ranked item when nothing is voted', () => {
      expect(autoSelectNext(input())).toEqual({ kind: 'media', id: 1 });
    });

    it('skips voted items on both sides', () => {
      const pick = autoSelectNext(input({ goodVotes: new Set([1]), badVotes: new Set([2]) }));
      expect(pick).toEqual({ kind: 'media', id: 3 });
    });

    it('skips the excluded id (the vote that has not landed yet)', () => {
      expect(autoSelectNext(input({ excludeId: 1 }))).toEqual({ kind: 'media', id: 2 });
    });

    it('picks nothing once every item is voted', () => {
      expect(autoSelectNext(input({ goodVotes: new Set([1, 2, 3, 4, 5]) }))).toEqual({
        kind: 'none',
      });
    });
  });

  describe('hard', () => {
    it('picks nothing when the sort carried no acquisition cut', () => {
      // Neither an acquisition cut nor a reporting threshold to fall back to:
      // there is no line to sample around, so the pick declines rather than
      // guessing at one.
      expect(autoSelectNext(input({ selectMode: 'hard', acqThreshold: null }))).toEqual({
        kind: 'none',
      });
    });

    it('takes the item at the cutoff index', () => {
      // 0.7 is the first score at or below 0.75, so thresholdIndex = 2 → id 3.
      expect(autoSelectNext(input({ selectMode: 'hard', acqThreshold: 0.75 }))).toEqual({
        kind: 'media',
        id: 3,
      });
    });

    it('falls to the nearest unvoted neighbour when the cutoff item is voted', () => {
      // thresholdIndex = 2; id 3 is voted, so ids 2 and 4 both sit at distance
      // 1 and the earlier (higher-ranked) one wins the strict `<` comparison.
      const pick = autoSelectNext(
        input({ selectMode: 'hard', acqThreshold: 0.75, goodVotes: new Set([3]) }),
      );
      expect(pick).toEqual({ kind: 'media', id: 2 });
    });

    it('measures distance in rank space, not score space', () => {
      // The separating case. Scores cluster tightly on each side of a wide gap
      // that straddles the cut, so the two metrics disagree outright:
      //
      //   index:  0     1     2     3     4
      //   score:  0.99  0.98  0.10  0.09  0.08     cut = 0.5, id 3 voted
      //
      // thresholdIndex = 2. In rank space ids 2 and 4 tie at distance 1 and the
      // earlier index wins → id 2. In score space id 4 (|0.09-0.5| = 0.41) beats
      // id 2 (|0.98-0.5| = 0.48) → id 4. Getting id 2 is what proves the rule is
      // the rank-space one the harness mirrors.
      const pick = autoSelectNext(
        input({
          sortOrder: rank([0.99, 0.98, 0.1, 0.09, 0.08]),
          selectMode: 'hard',
          acqThreshold: 0.5,
          goodVotes: new Set([3]),
        }),
      );
      expect(pick).toEqual({ kind: 'media', id: 2 });
    });

    it('uses the full ranking for the cutoff index, voted rows included', () => {
      // Voting out the whole head must not slide the cutoff position: the
      // threshold index is computed over every row, and voted rows are merely
      // skipped when choosing among the candidates.
      const pick = autoSelectNext(
        input({ selectMode: 'hard', acqThreshold: 0.75, goodVotes: new Set([1, 2, 3]) }),
      );
      // thresholdIndex stays 2; the nearest unvoted row is index 3 (id 4).
      expect(pick).toEqual({ kind: 'media', id: 4 });
    });

    it('takes the last item when every score is above the cut', () => {
      // No score falls at or below the cut, so thresholdIndex = length and the
      // final row is the nearest.
      expect(autoSelectNext(input({ selectMode: 'hard', acqThreshold: 0.1 }))).toEqual({
        kind: 'media',
        id: 5,
      });
    });

    it('takes the first item when every score is at or below the cut', () => {
      expect(autoSelectNext(input({ selectMode: 'hard', acqThreshold: 0.95 }))).toEqual({
        kind: 'media',
        id: 1,
      });
    });

    it('picks nothing once every item is voted', () => {
      const pick = autoSelectNext(
        input({
          selectMode: 'hard',
          acqThreshold: 0.75,
          goodVotes: new Set([1, 2, 3, 4, 5]),
        }),
      );
      expect(pick).toEqual({ kind: 'none' });
    });

    it('honours excludeId alongside the vote sets', () => {
      const pick = autoSelectNext(input({ selectMode: 'hard', acqThreshold: 0.75, excludeId: 3 }));
      expect(pick).toEqual({ kind: 'media', id: 2 });
    });
  });

  describe('new', () => {
    it('defers to the coverage atlas rather than picking from the ranking', () => {
      expect(autoSelectNext(input({ selectMode: 'new' }))).toEqual({ kind: 'diversity' });
    });

    it('defers even when every loaded item is already voted', () => {
      // The atlas probes the whole dataset, not just the loaded window, so an
      // exhausted window is not a reason to decline.
      const pick = autoSelectNext(
        input({ selectMode: 'new', goodVotes: new Set([1, 2, 3, 4, 5]) }),
      );
      expect(pick).toEqual({ kind: 'diversity' });
    });
  });
});
