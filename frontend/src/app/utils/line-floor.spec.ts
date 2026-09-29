import type { FloorState } from '../generated/api-client/models/floor-state';
import { SAMPLE_RANGE, lineFloor } from '../testing/line-floor';
import {
  FLOOR_PRESETS,
  floorExplanation,
  floorName,
  floorSummary,
  isFloorPreset,
  isUnpromised,
  lineFloorFrom,
  nearestFloorPreset,
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
      expect(why).toContain('Aimed at Centered');
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

  describe('the floor presets (#4298)', () => {
    it('offers three floors, named, symmetric about the Centered default', () => {
      expect(FLOOR_PRESETS.map((p) => p.value)).toEqual([0.1, 0.5, 0.9]);
      expect(FLOOR_PRESETS.map((p) => p.name)).toEqual(['Complete', 'Centered', 'Correct']);
    });

    it('knows a preset from a floor off the list', () => {
      expect(FLOOR_PRESETS.every((p) => isFloorPreset(p.value))).toBe(true);
      expect(isFloorPreset(0.25)).toBe(false);
      expect(isFloorPreset(0.75)).toBe(false);
    });

    it.each<[number, number]>([
      [0.01, 0.1],
      [0.25, 0.1],
      [0.29, 0.1],
      [0.31, 0.5],
      [0.6, 0.5],
      [0.75, 0.9],
      [1, 0.9],
    ])('snaps %s to the nearest preset, %s', (stored, snapped) => {
      expect(nearestFloorPreset(stored).value).toBe(snapped);
    });

    it('names a floor off the list after the preset it snaps to', () => {
      expect(floorName(0.5)).toBe('Centered');
      expect(floorName(0.25)).toBe('Complete');
      expect(floorName(0.75)).toBe('Correct');
    });
  });

  describe('the floor control copy (#4246, #4273)', () => {
    it('shows a check\'s range as a number', () => {
      expect(rangePercent(SAMPLE_RANGE)).toBe('11–73%');
    });

    it.each<[FloorStatus, string]>([
      ['confirmed', 'Confirmed · likely 55–100% right (checked 5) · 32 kept'],
      ['short', 'Aimed at Centered: likely 11–73% right (checked 5) · top 32 kept'],
      ['unchecked', 'Top 32 kept, unchecked · aiming at Centered'],
    ])('summarises %s', (status, expected) => {
      expect(floorSummary(lineFloor(status))).toBe(expected);
    });

    it.each<FloorStatus>(['confirmed', 'short', 'unchecked'])(
      'names the floor, never numbers it (%s, #4298)',
      (status) => {
        for (const minPrecision of [0.1, 0.25, 0.5, 0.9]) {
          const floor = lineFloor(status, { minPrecision });
          const target = `${Math.round(minPrecision * 100)}%`;
          expect(floorSummary(floor)).not.toContain(target);
          expect(floorExplanation(floor)).not.toContain(target);
        }
      },
    );

    it('reads the count from the result, never from the preset', () => {
      expect(floorSummary(lineFloor('confirmed', { minPrecision: 0.1, count: 64 }))).toContain('64 kept');
      expect(floorSummary(lineFloor('unchecked', { minPrecision: 0.1, count: 128 }))).toBe(
        'Top 128 kept, unchecked · aiming at Complete',
      );
    });

    it('has nothing to summarise or explain without a verdict', () => {
      expect(floorSummary(null)).toBeNull();
      expect(floorExplanation(null)).toBeNull();
    });

    it('explains a confirmed line by its check, with the range from the picks alone', () => {
      const why = floorExplanation(lineFloor('confirmed'))!;
      expect(why).toContain('5 random picks from the 32 items the line keeps found 5 right');
      expect(why).toContain('likely 55–100% of them are: enough for Centered');
      expect(why).not.toMatch(/estimate/i);
    });

    it('explains an unpromised line with its reason', () => {
      expect(floorExplanation(lineFloor('short'))).toBe(unpromisedReason(lineFloor('short')));
      expect(floorExplanation(lineFloor('unchecked'))).toBe(unpromisedReason(lineFloor('unchecked')));
    });
  });
});
