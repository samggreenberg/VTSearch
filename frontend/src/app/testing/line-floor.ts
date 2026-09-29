import type { FloorStatus, LikelyRange, LineFloor } from '../utils/line-floor';

/** The two states in which the line is not a confirmed set and reads unpromised (#4247, #4272). */
export const NO_PROMISE_STATES: FloorStatus[] = ['unchecked', 'short'];

/** The range a five-pick check at 50% typically leaves: 2 of 5 right. */
export const SAMPLE_RANGE: LikelyRange = { lo: 0.11, hi: 0.73, labelled: 5, right: 2, stale: false };

/** A floor verdict as the sort state holds it: a 50% floor keeping the top 32, with a check's range unless unchecked. */
export function lineFloor(status: FloorStatus, overrides: Partial<LineFloor> = {}): LineFloor {
  const range: LikelyRange | null =
    status === 'unchecked' ? null : status === 'confirmed' ? { ...SAMPLE_RANGE, lo: 0.55, hi: 1, right: 5 } : SAMPLE_RANGE;
  return {
    minPrecision: 0.5,
    status,
    count: 32,
    range,
    ...overrides,
  };
}

/** The same verdict as the wire `floor` object a response carries. */
export function wireFloor(status: FloorStatus, overrides: Partial<LineFloor> = {}) {
  const f = lineFloor(status, overrides);
  return {
    min_precision: f.minPrecision,
    status: f.status,
    count: f.count,
    range: f.range,
    schedule: { candidate: 32, rounds: 1, picks: 5 },
  };
}
