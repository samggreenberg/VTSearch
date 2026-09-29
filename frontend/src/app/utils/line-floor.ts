import type { FloorState } from '../generated/api-client/models/floor-state';

/**
 * What the precision floor says about the line a detector draws (#4247).
 *
 * - `promised`: the line is the floor's own cut, and at least `minPrecision`
 *   of what it returns is estimated right.
 * - `unreachable` / `insufficient_evidence`: the floor promised nothing, and
 *   the line is the **default cut** (Inclusion 0 on the backend), the one the
 *   app drew before there was a floor. Every match, count and action keeps
 *   working on it; the line is only labelled *unpromised*.
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

/**
 * The floors the control offers (owner, 2026-09-29): symmetric about the 50%
 * default, with 10% for a user willing to dig through a long list. The backend
 * takes any value in `[0.01, 1]`; a stored value outside this list still shows,
 * as its own option.
 */
export const FLOOR_PRESETS: readonly number[] = [0.1, 0.25, 0.5, 0.75, 0.9];

/**
 * The floor the control shows before the detector's own value arrives from
 * `GET /api/min-precision`. It is the backend's default
 * (`vtscore.config.runtime.DEFAULT_MIN_PRECISION`), and only a placeholder:
 * nothing is sent until the user picks a floor.
 */
export const DEFAULT_MIN_PRECISION = 0.5;

/** "50%" for 0.5. */
export function floorPercent(p: number | null): string {
  return `${Math.round((p ?? 0) * 100)}%`;
}

/**
 * Why the line is unpromised, in a sentence for a tooltip; null when it is not.
 * Both states end the same way: the line is the default cut, so what the user
 * sees is what they saw before there was a floor.
 */
export function unpromisedReason(floor: LineFloor | null): string | null {
  if (!floor || !isUnpromised(floor)) return null;
  const target = floorPercent(floor.minPrecision);
  const why =
    floor.status === 'unreachable'
      ? `No cut reaches ${target} precision on this dataset, so no promise is made.`
      : `Not enough evidence yet to promise ${target} precision: it has ` +
        `${floor.calibrationPositives} of the ${floor.minCalibrationPositives} held-back Good votes it needs. ` +
        'Votes on Autopilot’s Hard picks, or down a learned sort, add evidence.';
  return `${why} The line stays at the default cut, where it sat before there was a floor.`;
}

/**
 * The floor's state in one short line, for the control under the picker
 * (#4246): the floor and whether the line keeps it, never the estimate behind
 * it (owner, 2026-09-28). `returned` is how many items the line returns; null
 * when unknown. Null when there is no verdict to report (no line yet, or a
 * sort with no detector behind it).
 */
export function floorSummary(floor: LineFloor | null, returned: number | null): string | null {
  if (!floor) return null;
  const target = floorPercent(floor.minPrecision);
  switch (floor.status) {
    case 'promised':
      return returned === null
        ? `At least ${target} right`
        : `At least ${target} right · ${returned.toLocaleString()} returned`;
    case 'unreachable':
      return `Can't reach ${target} on this dataset · showing the default cut`;
    case 'insufficient_evidence':
      return (
        `Not enough evidence yet (${floor.calibrationPositives} of ${floor.minCalibrationPositives} Good votes)` +
        ' · showing the default cut'
      );
    default:
      return 'No floor set · the line makes no promise';
  }
}

/**
 * The same state at tooltip length: what the short line means, and for an
 * unpromised line, why. Null when {@link floorSummary} is.
 */
export function floorExplanation(floor: LineFloor | null, returned: number | null): string | null {
  if (!floor) return null;
  const target = floorPercent(floor.minPrecision);
  if (floor.status === 'promised') {
    const what = returned === null ? 'what the line returns' : `the ${returned.toLocaleString()} items the line returns`;
    return (
      `At least ${target} of ${what} is estimated to be right. The estimate is cautious: ` +
      'the line returns as much as it can while keeping that promise.'
    );
  }
  if (floor.status === null) {
    return 'This detector has no precision floor, so its line makes no promise. Pick a floor to set one.';
  }
  return unpromisedReason(floor);
}
