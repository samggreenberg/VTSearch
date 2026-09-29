import type { FloorState } from '../generated/api-client/models/floor-state';
import { lineFloor } from '../testing/line-floor';
import {
  FLOOR_PRESETS,
  floorExplanation,
  floorPercent,
  floorSummary,
  isUnpromised,
  lineFloorFrom,
  unpromisedReason,
  type FloorStatus,
} from './line-floor';

describe('line-floor (#4247)', () => {
  describe('lineFloorFrom', () => {
    it('reads the wire object', () => {
      const wire: FloorState = {
        min_precision: 0.8,
        status: 'unreachable',
        calibration_positives: 14,
        min_calibration_positives: 10,
      };
      expect(lineFloorFrom(wire)).toEqual({
        minPrecision: 0.8,
        status: 'unreachable',
        calibrationPositives: 14,
        minCalibrationPositives: 10,
      });
    });

    it('is null for a response with no detector behind it', () => {
      expect(lineFloorFrom(null)).toBeNull();
      expect(lineFloorFrom(undefined)).toBeNull();
    });

    it('reads a status outside the three states as no verdict', () => {
      const wire = { min_precision: 0.5, status: 'bogus', calibration_positives: 0, min_calibration_positives: 10 };
      expect(lineFloorFrom(wire as unknown as FloorState)).toBeNull();
    });
  });

  describe('isUnpromised', () => {
    it.each<[FloorStatus, boolean]>([
      ['promised', false],
      ['unreachable', true],
      ['insufficient_evidence', true],
    ])('%s -> %s', (status, expected) => {
      expect(isUnpromised(lineFloor(status))).toBe(expected);
    });

    it('is false with no floor object at all', () => {
      expect(isUnpromised(null)).toBe(false);
    });
  });

  describe('unpromisedReason', () => {
    it('names the missing evidence and what adds it', () => {
      const why = unpromisedReason(lineFloor('insufficient_evidence'));
      expect(why).toContain('50% precision');
      expect(why).toContain('3 of the 10');
      expect(why).toContain('default cut');
    });

    it('says no cut reaches an unreachable floor', () => {
      const why = unpromisedReason(lineFloor('unreachable', { minPrecision: 0.9, calibrationPositives: 20 }));
      expect(why).toContain('No cut reaches 90% precision');
      expect(why).toContain('default cut');
    });

    it('has nothing to say about a promised line or no verdict', () => {
      expect(unpromisedReason(lineFloor('promised'))).toBeNull();
      expect(unpromisedReason(null)).toBeNull();
    });
  });

  describe('the floor control copy (#4246)', () => {
    it('offers the four priced floors', () => {
      expect(FLOOR_PRESETS).toEqual([0.25, 0.5, 0.75, 0.9]);
      expect(FLOOR_PRESETS.map(floorPercent)).toEqual(['25%', '50%', '75%', '90%']);
    });

    it.each<[FloorStatus, string]>([
      ['promised', 'At least 50% right · 1,200 returned'],
      ['unreachable', "Can't reach 50% on this dataset · showing the default cut"],
      ['insufficient_evidence', 'Not enough evidence yet (3 of 10 Good votes) · showing the default cut'],
    ])('summarises %s', (status, expected) => {
      expect(floorSummary(lineFloor(status), 1200)).toBe(expected);
    });

    it('has nothing to summarise or explain without a verdict', () => {
      expect(floorSummary(null, 10)).toBeNull();
      expect(floorExplanation(null, 10)).toBeNull();
    });

    it('explains a promise without the estimate behind it', () => {
      const why = floorExplanation(lineFloor('promised'), 12)!;
      expect(why).toContain('At least 50% of the 12 items the line returns');
    });

    it('explains an unpromised line with its reason', () => {
      expect(floorExplanation(lineFloor('unreachable'), 12)).toBe(unpromisedReason(lineFloor('unreachable')));
    });
  });
});
