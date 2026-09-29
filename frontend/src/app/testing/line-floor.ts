import type { FloorStatus, LineFloor } from '../utils/line-floor';

/** The two states in which the floor promises nothing and the line is the Inclusion 0 cut (#4247). */
export const NO_PROMISE_STATES: FloorStatus[] = ['unreachable', 'insufficient_evidence'];

/** A floor verdict as the sort state holds it: a 50% floor with 3 of 10 calibration positives. */
export function lineFloor(status: FloorStatus, overrides: Partial<LineFloor> = {}): LineFloor {
  return {
    minPrecision: 0.5,
    status,
    calibrationPositives: 3,
    minCalibrationPositives: 10,
    ...overrides,
  };
}

/** The same verdict as the wire `floor` object a response carries. */
export function wireFloor(status: FloorStatus, overrides: Partial<LineFloor> = {}) {
  const f = lineFloor(status, overrides);
  return {
    min_precision: f.minPrecision,
    status: f.status,
    calibration_positives: f.calibrationPositives,
    min_calibration_positives: f.minCalibrationPositives,
  };
}
