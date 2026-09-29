import type { FloorState } from '../generated/api-client/models/floor-state';
import { lineFloor } from '../testing/line-floor';
import {
  FLOOR_PRESETS,
  floorExplanation,
  floorName,
  floorSummary,
  isFloorPreset,
  isUnpromised,
  lineFloorFrom,
  nearestFloorPreset,
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
      expect(why).toContain('promise Centered');
      expect(why).toContain('3 of the 10');
      expect(why).toContain('default cut');
    });

    it('says no cut reaches an unreachable floor', () => {
      const why = unpromisedReason(lineFloor('unreachable', { minPrecision: 0.9, calibrationPositives: 20 }));
      expect(why).toContain('No cut on this dataset reaches Correct');
      expect(why).toContain('default cut');
    });

    it('has nothing to say about a promised line or no verdict', () => {
      expect(unpromisedReason(lineFloor('promised'))).toBeNull();
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

  describe('the floor control copy (#4246)', () => {
    it.each<[FloorStatus, string]>([
      ['promised', 'Promise kept · 1,200 returned'],
      ['unreachable', "Can't reach Centered on this dataset · showing the default cut"],
      ['insufficient_evidence', 'Not enough evidence yet (3 of 10 Good votes) · showing the default cut'],
    ])('summarises %s', (status, expected) => {
      expect(floorSummary(lineFloor(status), 1200)).toBe(expected);
    });

    it.each<FloorStatus>(['promised', 'unreachable', 'insufficient_evidence'])(
      'never shows the floor as a number (%s, #4298)',
      (status) => {
        for (const minPrecision of [0.1, 0.25, 0.5, 0.9]) {
          const floor = lineFloor(status, { minPrecision });
          expect(floorSummary(floor, 1200)).not.toContain('%');
          expect(floorExplanation(floor, 1200)).not.toContain('%');
        }
      },
    );

    it('has nothing to summarise or explain without a verdict', () => {
      expect(floorSummary(null, 10)).toBeNull();
      expect(floorExplanation(null, 10)).toBeNull();
    });

    it('explains a promise without the estimate behind it', () => {
      const why = floorExplanation(lineFloor('promised'), 12)!;
      expect(why).toContain('keeps its Centered promise on the 12 items it returns');
    });

    it('explains an unpromised line with its reason', () => {
      expect(floorExplanation(lineFloor('unreachable'), 12)).toBe(unpromisedReason(lineFloor('unreachable')));
    });
  });
});
