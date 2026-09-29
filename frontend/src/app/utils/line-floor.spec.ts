import type { FloorState } from '../generated/api-client/models/floor-state';
import { SAMPLE_RANGE, lineFloor } from '../testing/line-floor';
import {
  FLOOR_PRESETS,
  checkLabel,
  checkTitle,
  floorExplanation,
  floorPercent,
  floorSummary,
  lineFloorFrom,
  rangePercent,
  rangeTitle,
  staleNote,
  type FloorStatus,
} from './line-floor';

describe('line-floor (#4272, #4273)', () => {
  describe('lineFloorFrom', () => {
    it('reads the wire object, schedule included', () => {
      const wire: FloorState = {
        min_precision: 0.25,
        status: 'confirmed',
        count: 64,
        range: { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: true },
        schedule: { candidate: 64, rounds: 2, picks: 5 },
      };
      expect(lineFloorFrom(wire)).toEqual({
        minPrecision: 0.25,
        status: 'confirmed',
        count: 64,
        range: { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: true },
        schedule: { candidate: 64, rounds: 2, picks: 5 },
      });
    });

    it('reads an unchecked line with no range', () => {
      const wire = { min_precision: 0.1, status: 'unchecked', count: 128, range: null, schedule: { candidate: 128, rounds: 3, picks: 5 } };
      expect(lineFloorFrom(wire as unknown as FloorState)).toEqual({
        minPrecision: 0.1,
        status: 'unchecked',
        count: 128,
        range: null,
        schedule: { candidate: 128, rounds: 3, picks: 5 },
      });
    });

    it('reads a range with no stale flag as current', () => {
      const wire = { min_precision: 0.5, status: 'short', count: 32, range: { lo: 0.11, hi: 0.73, labelled: 5, right: 2 }, schedule: null };
      const floor = lineFloorFrom(wire as unknown as FloorState)!;
      expect(floor.range!.stale).toBe(false);
      expect(floor.schedule).toBeNull();
    });

    it('is null for a response with no detector behind it', () => {
      expect(lineFloorFrom(null)).toBeNull();
      expect(lineFloorFrom(undefined)).toBeNull();
    });

    it('reads a status outside the three states as no verdict', () => {
      const wire = { min_precision: 0.5, status: 'promised', count: 32, range: null, schedule: { candidate: 32, rounds: 1, picks: 5 } };
      expect(lineFloorFrom(wire as unknown as FloorState)).toBeNull();
    });
  });

  describe('the floor control copy (#4246, #4273)', () => {
    it('offers five floors, symmetric about the 50% default', () => {
      expect(FLOOR_PRESETS).toEqual([0.1, 0.25, 0.5, 0.75, 0.9]);
      expect(FLOOR_PRESETS.map(floorPercent)).toEqual(['10%', '25%', '50%', '75%', '90%']);
      expect(rangePercent(SAMPLE_RANGE)).toBe('11–73%');
    });

    it.each<[FloorStatus, string]>([
      ['confirmed', 'At least 50% right · likely 55–100% (checked 5) · 32 kept'],
      ['short', 'Aimed at 50%: likely 11–73% right (checked 5) · top 32 kept'],
      ['unchecked', 'Top 32 kept, unchecked · aiming at 50%'],
    ])('summarises %s', (status, expected) => {
      expect(floorSummary(lineFloor(status))).toBe(expected);
    });

    it('reads the count from the result, never from the preset', () => {
      expect(floorSummary(lineFloor('confirmed', { minPrecision: 0.1, count: 64 }))).toContain('64 kept');
      expect(floorSummary(lineFloor('confirmed', { minPrecision: 0.1, count: 128 }))).toContain('128 kept');
      expect(floorSummary(lineFloor('unchecked', { minPrecision: 0.1, count: 128 }))).toBe(
        'Top 128 kept, unchecked · aiming at 10%',
      );
    });

    it('carries the range wide as the check left it: a five-pick range at 50% is 0.62 wide', () => {
      const text = floorSummary(lineFloor('short'))!;
      expect(text).toContain('11–73%');
      expect(text).toContain('(checked 5)');
    });

    it('has nothing to summarise, explain or check without a verdict', () => {
      expect(floorSummary(null)).toBeNull();
      expect(floorExplanation(null)).toBeNull();
      expect(checkLabel(null)).toBeNull();
    });

    it('explains a confirmed line by its check, with the range from the picks alone', () => {
      const why = floorExplanation(lineFloor('confirmed'))!;
      expect(why).toContain('5 random picks from the 32 items the line keeps found 5 right');
      expect(why).toContain('likely 55–100% of them are: at least 50%');
      expect(why).not.toMatch(/estimate/i);
    });

    it('explains an unchecked line as unmeasured, and what a check costs', () => {
      const why = floorExplanation(lineFloor('unchecked', { count: 128, schedule: { candidate: 128, rounds: 3, picks: 5 } }))!;
      expect(why).toContain('Unchecked: the line keeps the top 128');
      expect(why).toContain('5 random picks a round, in up to 3 rounds');
      expect(why).not.toMatch(/likely/);
    });

    it('explains a short check by how close it got, naming no cause', () => {
      const why = floorExplanation(lineFloor('short'))!;
      expect(why).toContain('Aimed at 50%');
      expect(why).toContain('5 random picks from the top 32');
      expect(why).toContain('likely 11–73% of them are, short of the floor');
      // The app can't tell a sparse corpus from a weak model (owner, 2026-09-29).
      expect(why).not.toMatch(/sparse|weak|evidence|too few|model/i);
    });

    it.each<FloorStatus>(['short', 'confirmed'])('says a stale %s range is stale in the tooltip, and only there', (status) => {
      const fresh = lineFloor(status);
      const stale = lineFloor(status, { range: { ...fresh.range!, stale: true } });
      expect(floorSummary(stale)).toBe(floorSummary(fresh));
      expect(floorExplanation(stale)).toBe(floorExplanation(fresh) + staleNote(stale.range));
      expect(floorExplanation(stale)).toContain('Measured before your later votes');
      expect(floorExplanation(fresh)).not.toContain('later votes');
    });
  });

  describe('the check affordance (#4273)', () => {
    it.each<[number, number, string]>([
      [0.1, 5, 'Check 5 picks'],
      [0.5, 5, 'Check 5 picks'],
      [0.75, 11, 'Check 11 picks'],
      [0.9, 29, 'Check 29 picks'],
    ])('at %s reads "%s picks" off the schedule', (x, picks, label) => {
      const floor = lineFloor('unchecked', { minPrecision: x, schedule: { candidate: 32, rounds: 1, picks } });
      expect(checkLabel(floor)).toBe(label);
    });

    it('is offered in every state: after a finished check it runs a fresh one', () => {
      for (const status of ['unchecked', 'confirmed', 'short'] as const) {
        expect(checkLabel(lineFloor(status))).toBe('Check 5 picks');
      }
    });

    it('says what a check does and that its votes are votes', () => {
      const title = checkTitle(lineFloor('unchecked', { schedule: { candidate: 64, rounds: 2, picks: 5 } }));
      expect(title).toContain('5 random picks a round, in up to 2 rounds');
      expect(title).toContain('ordinary votes');
    });
  });

  describe('rangeTitle', () => {
    it('names the picks the range comes from', () => {
      expect(rangeTitle(SAMPLE_RANGE)).toBe('Likely 11–73% right, from 5 random picks (2 right).');
    });

    it('adds the stale sentence for a stale range', () => {
      expect(rangeTitle({ ...SAMPLE_RANGE, stale: true })).toContain('Measured before your later votes');
    });
  });
});
