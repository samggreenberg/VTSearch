import type { FloorState } from '../generated/api-client/models/floor-state';
import { lineFloor } from '../testing/line-floor';
import { isUnpromised, lineFloorFrom, unpromisedReason, type FloorStatus } from './line-floor';

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

    it('reads a null status as no floor, whatever the generated type spells it', () => {
      const wire = { min_precision: null, status: null, calibration_positives: 0, min_calibration_positives: 10 };
      expect(lineFloorFrom(wire as unknown as FloorState)?.status).toBeNull();
      expect(lineFloorFrom({ ...wire, status: 'null' } as FloorState)?.status).toBeNull();
    });
  });

  describe('isUnpromised', () => {
    it.each<[FloorStatus | null, boolean]>([
      ['promised', false],
      ['unreachable', true],
      ['insufficient_evidence', true],
      [null, false],
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
      expect(why).toContain('Inclusion 0');
    });

    it('says no cut reaches an unreachable floor', () => {
      const why = unpromisedReason(lineFloor('unreachable', { minPrecision: 0.9, calibrationPositives: 20 }));
      expect(why).toContain('No cut reaches 90% precision');
      expect(why).toContain('Inclusion 0');
    });

    it('has nothing to say about a promised line or no floor', () => {
      expect(unpromisedReason(lineFloor('promised'))).toBeNull();
      expect(unpromisedReason(lineFloor(null))).toBeNull();
      expect(unpromisedReason(null)).toBeNull();
    });
  });
});
