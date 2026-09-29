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
  /** The detector's floor. Every detector has one (#4269). */
  minPrecision: number;
  status: FloorStatus;
  /** Positives among the held-out votes that may calibrate a promise. */
  calibrationPositives: number;
  /** How many the floor needs before it promises anything. */
  minCalibrationPositives: number;
}

const STATUSES: readonly FloorStatus[] = ['promised', 'unreachable', 'insufficient_evidence'];

/**
 * The wire `floor` object, as a {@link LineFloor}; null when the response
 * carried none (a sort with no detector behind it), or a status outside the
 * three states.
 */
export function lineFloorFrom(wire: FloorState | null | undefined): LineFloor | null {
  if (!wire) return null;
  const status = STATUSES.find((s) => s === wire.status);
  if (!status) return null;
  return {
    minPrecision: wire.min_precision,
    status,
    calibrationPositives: wire.calibration_positives ?? 0,
    minCalibrationPositives: wire.min_calibration_positives ?? 0,
  };
}

/** True when the floor promises nothing: the line is the Inclusion 0 fallback. */
export function isUnpromised(floor: LineFloor | null): boolean {
  return floor?.status === 'unreachable' || floor?.status === 'insufficient_evidence';
}

/** One floor the control offers, and the word the user sees for it. */
export interface FloorPreset {
  /** The floor, a fraction. */
  value: number;
  /** Its name: the control never shows the number (#4298). */
  name: string;
}

/**
 * The floors the control offers (owner, 2026-09-29, #4298): three, named
 * rather than numbered. The spot check rarely delivers a floor exactly, so
 * five percentages claimed a precision it could not keep; three words on one
 * Complete - Correct scale promise only a direction. The backend takes any
 * value in `[0.01, 1]`; the control snaps one outside this list to the
 * nearest (see {@link nearestFloorPreset}).
 */
export const FLOOR_PRESETS: readonly FloorPreset[] = [
  { value: 0.1, name: 'Complete' },
  { value: 0.5, name: 'Centered' },
  { value: 0.9, name: 'Correct' },
];

/**
 * The floor the control shows before the detector's own value arrives from
 * `GET /api/min-precision`. It is the backend's default
 * (`vtscore.config.runtime.DEFAULT_MIN_PRECISION`), and only a placeholder:
 * nothing is sent until the user picks a floor.
 */
export const DEFAULT_MIN_PRECISION = 0.5;

/** True when `p` is one of the floors the control offers. */
export function isFloorPreset(p: number): boolean {
  return FLOOR_PRESETS.some((preset) => preset.value === p);
}

/**
 * The preset closest to `p`, for a stored floor the control does not offer (a
 * pick from before #4298, or one set through the CLI or the API). A tie goes
 * to the lower preset.
 */
export function nearestFloorPreset(p: number): FloorPreset {
  return FLOOR_PRESETS.reduce((best, preset) =>
    Math.abs(preset.value - p) < Math.abs(best.value - p) ? preset : best,
  );
}

/** "Centered" for 0.5; a floor off the list reads as the preset nearest it. */
export function floorName(p: number): string {
  return nearestFloorPreset(p).name;
}

/**
 * Why the line is unpromised, in a sentence for a tooltip; null when it is not.
 * Both states end the same way: the line is the default cut, so what the user
 * sees is what they saw before there was a floor.
 */
export function unpromisedReason(floor: LineFloor | null): string | null {
  if (!floor || !isUnpromised(floor)) return null;
  const target = floorName(floor.minPrecision);
  const why =
    floor.status === 'unreachable'
      ? `No cut on this dataset reaches ${target}, so no promise is made.`
      : `Not enough evidence yet to promise ${target}: it has ` +
        `${floor.calibrationPositives} of the ${floor.minCalibrationPositives} held-back Good votes it needs. ` +
        'Votes on Autopilot’s Hard picks, or down a learned sort, add evidence.';
  return `${why} The line stays at the default cut, where it sat before there was a floor.`;
}

/**
 * The floor's state in one short line, for the control under the picker
 * (#4246): whether the line keeps the floor, never the estimate behind it
 * (owner, 2026-09-28), and never the floor as a number (#4298). `returned` is
 * how many items the line returns; null when unknown. Null when there is no
 * verdict to report (no line yet, or a sort with no detector behind it).
 */
export function floorSummary(floor: LineFloor | null, returned: number | null): string | null {
  if (!floor) return null;
  switch (floor.status) {
    case 'promised':
      return returned === null ? 'Promise kept' : `Promise kept · ${returned.toLocaleString()} returned`;
    case 'unreachable':
      return `Can't reach ${floorName(floor.minPrecision)} on this dataset · showing the default cut`;
    case 'insufficient_evidence':
      return (
        `Not enough evidence yet (${floor.calibrationPositives} of ${floor.minCalibrationPositives} Good votes)` +
        ' · showing the default cut'
      );
  }
}

/**
 * The same state at tooltip length: what the short line means, and for an
 * unpromised line, why. Null when {@link floorSummary} is.
 */
export function floorExplanation(floor: LineFloor | null, returned: number | null): string | null {
  if (!floor) return null;
  if (floor.status === 'promised') {
    const what = returned === null ? 'everything it returns' : `the ${returned.toLocaleString()} items it returns`;
    return (
      `The line keeps its ${floorName(floor.minPrecision)} promise on ${what}. The estimate behind it is ` +
      'cautious: the line returns as much as it can while keeping that promise.'
    );
  }
  return unpromisedReason(floor);
}
