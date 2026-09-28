import type { FloorState } from '../generated/api-client/models/floor-state';

/**
 * What the precision floor says about the line a detector draws (#4247).
 *
 * - `promised`: the line is the floor's own cut, and at least `minPrecision`
 *   of what it returns is estimated right.
 * - `unreachable` / `insufficient_evidence`: the floor promised nothing, and
 *   the line is the **Inclusion 0 cut**, the one the app drew before there was
 *   a floor. Every match, count and action keeps working on it; the line is
 *   only labelled *unpromised*.
 */
export type FloorStatus = 'promised' | 'unreachable' | 'insufficient_evidence';

/** The floor's verdict on the current line, as the sort state holds it. */
export interface LineFloor {
  /** The detector's floor, or null when none is set and Inclusion drew the line. */
  minPrecision: number | null;
  /** Null when no floor is set. */
  status: FloorStatus | null;
  /** Positives among the held-out votes that may calibrate a promise. */
  calibrationPositives: number;
  /** How many the floor needs before it promises anything. */
  minCalibrationPositives: number;
}

const STATUSES: readonly FloorStatus[] = ['promised', 'unreachable', 'insufficient_evidence'];

/**
 * The wire `floor` object, as a {@link LineFloor}; null when the response
 * carried none (a sort with no detector behind it).
 *
 * The generated type spells a null `status` as the string `'null'` (the
 * generator's reading of a nullable enum); the JSON carries a real null, and
 * anything outside the three states is read as no floor.
 */
export function lineFloorFrom(wire: FloorState | null | undefined): LineFloor | null {
  if (!wire) return null;
  const status = STATUSES.find((s) => s === (wire.status as string | null)) ?? null;
  return {
    minPrecision: wire.min_precision ?? null,
    status,
    calibrationPositives: wire.calibration_positives ?? 0,
    minCalibrationPositives: wire.min_calibration_positives ?? 0,
  };
}

/** True when a floor is set and promises nothing: the line is the Inclusion 0 fallback. */
export function isUnpromised(floor: LineFloor | null): boolean {
  return floor?.status === 'unreachable' || floor?.status === 'insufficient_evidence';
}

/** "50%" for 0.5. */
function pct(p: number | null): string {
  return `${Math.round((p ?? 0) * 100)}%`;
}

/**
 * Why the line is unpromised, in a sentence for a tooltip; null when it is not.
 * Both states end the same way: the line is where Inclusion 0 puts it, so what
 * the user sees is what they saw before there was a floor.
 */
export function unpromisedReason(floor: LineFloor | null): string | null {
  if (!floor || !isUnpromised(floor)) return null;
  const target = pct(floor.minPrecision);
  const why =
    floor.status === 'unreachable'
      ? `No cut reaches ${target} precision on this dataset, so no promise is made.`
      : `Not enough evidence yet to promise ${target} precision: it has ` +
        `${floor.calibrationPositives} of the ${floor.minCalibrationPositives} held-back Good votes it needs. ` +
        'Votes on Autopilot’s Hard picks, or down a learned sort, add evidence.';
  return `${why} The line stays at the Inclusion 0 cut.`;
}
