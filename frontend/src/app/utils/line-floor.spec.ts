import type { FloorState } from '../generated/api-client/models/floor-state';
import { SAMPLE_RANGE, lineFloor } from '../testing/line-floor';
import {
  FLOOR_PRESETS,
  floorExplanation,
  floorPercent,
  floorSummary,
  isUnpromised,
  lineFloorFrom,
  rangePercent,
  unpromisedReason,
  type FloorStatus,
} from './line-floor';

describe('line-floor (#4247, #4272)', () => {
  describe('lineFloorFrom', () => {
    it('reads the wire object', () => {
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
      });
    });

    it('reads an unchecked line with no range', () => {
      const wire = { min_precision: 0.1, status: 'unchecked', count: 128, range: null, schedule: { candidate: 128, rounds: 3, picks: 5 } };
      expect(lineFloorFrom(wire as unknown as FloorState)).toEqual({ minPrecision: 0.1, status: 'unchecked', count: 128, range: null });
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

  describe('isUnpromised', () => {
    it.each<[FloorStatus, boolean]>([
      ['confirmed', false],
      ['unchecked', true],
      ['short', true],
    ])('%s -> %s', (status, expected) => {
      expect(isUnpromised(lineFloor(status))).toBe(expected);
    });

    it('is false with no floor object at all', () => {
      expect(isUnpromised(null)).toBe(false);
    });
  });

  describe('unpromisedReason', () => {
    it('says an unchecked line has not been measured, and how to', () => {
      const why = unpromisedReason(lineFloor('unchecked', { count: 128 }));
      expect(why).toContain('Unchecked');
      expect(why).toContain('top 128 the line keeps');
      expect(why).toContain('Run a check');
    });

    it('says how close a short check got, naming no cause', () => {
      const why = unpromisedReason(lineFloor('short'))!;
      expect(why).toContain('Aimed at 50%');
      expect(why).toContain('5 random picks');
      expect(why).toContain('likely 11–73% right');
      expect(why).not.toMatch(/sparse|weak|evidence/);
    });

    it('notes a stale range in the tooltip only', () => {
      const why = unpromisedReason(lineFloor('short', { range: { ...SAMPLE_RANGE, stale: true } }))!;
      expect(why).toContain('Measured before your later votes');
      expect(floorSummary(lineFloor('short', { range: { ...SAMPLE_RANGE, stale: true } }))).toBe(
        floorSummary(lineFloor('short')),
      );
    });

    it('has nothing to say about a confirmed line or no verdict', () => {
      expect(unpromisedReason(lineFloor('confirmed'))).toBeNull();
      expect(unpromisedReason(null)).toBeNull();
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
      expect(floorSummary(lineFloor('unchecked', { minPrecision: 0.1, count: 128 }))).toBe(
        'Top 128 kept, unchecked · aiming at 10%',
      );
    });

    it('has nothing to summarise or explain without a verdict', () => {
      expect(floorSummary(null)).toBeNull();
      expect(floorExplanation(null)).toBeNull();
    });

    it('explains a confirmed line by its check, with the range from the picks alone', () => {
      const why = floorExplanation(lineFloor('confirmed'))!;
      expect(why).toContain('5 random picks from the 32 items the line keeps found 5 right');
      expect(why).toContain('likely 55–100% of them are: at least 50%');
      expect(why).not.toMatch(/estimate/i);
    });

    it('explains an unpromised line with its reason', () => {
      expect(floorExplanation(lineFloor('short'))).toBe(unpromisedReason(lineFloor('short')));
      expect(floorExplanation(lineFloor('unchecked'))).toBe(unpromisedReason(lineFloor('unchecked')));
    });
  });
});
