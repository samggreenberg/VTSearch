import type { FloorState } from '../generated/api-client/models/floor-state';

/**
 * What the precision floor says about the line a detector draws (#4247, #4272).
 *
 * The line always keeps a set: the top `count` unvoted items of the ranking.
 *
 * - `unchecked`: no spot check has run at this floor. The set is the floor's
 *   starting candidate (the top 128 at 10%, 64 at 25%, 32 at 50% and above),
 *   and nothing has measured how much of it is right.
 * - `confirmed`: the last check's likely range clears the floor.
 * - `short`: the check ended below the floor; the line keeps the top 32 it
 *   ended on, and the range says how close it got.
 *
 * Every match, count and action keeps working on the line in every state; an
 * `unchecked` or `short` line is only labelled *unpromised*.
 */
export type FloorStatus = 'unchecked' | 'confirmed' | 'short';

/** How much of the kept set is likely right, from the check's picks alone. */
export interface LikelyRange {
  lo: number;
  hi: number;
  /** How many of the set's items the check labelled, and how many were right. */
  labelled: number;
  right: number;
  /** True once later votes moved the list under the result: the range describes the list as it was. */
  stale: boolean;
}

/** The floor's verdict on the current line, as the sort state holds it. */
export interface LineFloor {
  /** The detector's floor. Every detector has one (#4269). */
  minPrecision: number;
  status: FloorStatus;
  /** How many unvoted items the line keeps. */
  count: number;
  /** The check's likely range for those items; null while unchecked. */
  range: LikelyRange | null;
}

const STATUSES: readonly FloorStatus[] = ['unchecked', 'confirmed', 'short'];

/**
 * The wire `floor` object, as a {@link LineFloor}; null when the response
 * carried none (a sort with no detector behind it), or a status outside the
 * three states.
 */
export function lineFloorFrom(wire: FloorState | null | undefined): LineFloor | null {
  if (!wire) return null;
  const status = STATUSES.find((s) => s === wire.status);
  if (!status) return null;
  const range = wire.range;
  return {
    minPrecision: wire.min_precision,
    status,
    count: wire.count ?? 0,
    range: range
      ? { lo: range.lo, hi: range.hi, labelled: range.labelled, right: range.right, stale: range.stale ?? false }
      : null,
  };
}

/** True when the line is not a confirmed set: unchecked, or a check that fell short. */
export function isUnpromised(floor: LineFloor | null): boolean {
  return floor?.status === 'unchecked' || floor?.status === 'short';
}

/** One floor the control offers, and the word the user sees for it. */
export interface FloorPreset {
  /** The floor, a fraction. */
  value: number;
  /** Its name: the control never shows the floor as a number (#4298). */
  name: string;
}

/**
 * The floors the control offers (owner, 2026-09-29, #4298): three, named
 * rather than numbered. The spot check rarely delivers a floor exactly, so
 * five percentages claimed a precision it could not keep; three words on one
 * Complete - Correct scale promise only a direction. The floor is a setting
 * and is named; what a check measures (its likely range) stays a number. The
 * backend takes any value in `[0.01, 1]`; the control snaps one outside this
 * list to the nearest (see {@link nearestFloorPreset}).
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

/** "11–73%" for a range. */
export function rangePercent(range: LikelyRange): string {
  return `${Math.round(range.lo * 100)}–${Math.round(range.hi * 100)}%`;
}

/** The sentence a stale range's tooltip adds; empty when the range is current. */
function staleNote(range: LikelyRange | null): string {
  return range?.stale ? ' Measured before your later votes: the list at the line has changed since.' : '';
}

/**
 * Why the line is unpromised, in a sentence for a tooltip; null when it is not.
 * A short check names no cause: the app can't tell a sparse corpus from a
 * weak model, so the copy is true in both cases.
 */
export function unpromisedReason(floor: LineFloor | null): string | null {
  if (!floor || !isUnpromised(floor)) return null;
  const target = floorName(floor.minPrecision);
  const kept = floor.count.toLocaleString();
  if (floor.status === 'unchecked' || !floor.range) {
    return (
      `Unchecked: nobody has measured how much of the top ${kept} the line keeps is right. ` +
      `Run a check of a few random picks from it to find out.`
    );
  }
  const r = floor.range;
  return (
    `Aimed at ${target}: a check of ${r.labelled} random picks found the top ${kept} the line keeps ` +
    `likely ${rangePercent(r)} right, so it fell short.` +
    staleNote(r)
  );
}

/**
 * The floor's state in one short line, for the control under the picker
 * (#4246, #4273): the floor, what the line keeps, and how close the check got.
 * The floor goes by its name, never its number (#4298); the range comes only
 * from the check's picks, never from the model. Null when there is no verdict
 * to report (no line yet, or a sort with no detector behind it).
 */
export function floorSummary(floor: LineFloor | null): string | null {
  if (!floor) return null;
  const target = floorName(floor.minPrecision);
  const kept = floor.count.toLocaleString();
  const r = floor.range;
  switch (floor.status) {
    case 'confirmed':
      return r
        ? `Confirmed · likely ${rangePercent(r)} right (checked ${r.labelled}) · ${kept} kept`
        : `Confirmed · ${kept} kept`;
    case 'short':
      return r
        ? `Aimed at ${target}: likely ${rangePercent(r)} right (checked ${r.labelled}) · top ${kept} kept`
        : `Aimed at ${target}: fell short · top ${kept} kept`;
    case 'unchecked':
      return `Top ${kept} kept, unchecked · aiming at ${target}`;
  }
}

/**
 * The same state at tooltip length: what the short line means, and for an
 * unpromised line, why. Null when {@link floorSummary} is.
 */
export function floorExplanation(floor: LineFloor | null): string | null {
  if (!floor) return null;
  if (floor.status !== 'confirmed') return unpromisedReason(floor);
  const target = floorName(floor.minPrecision);
  const kept = floor.count.toLocaleString();
  const r = floor.range;
  if (!r) return `A check confirmed the ${kept} items the line keeps at ${target}.`;
  return (
    `A check of ${r.labelled} random picks from the ${kept} items the line keeps found ${r.right} right, ` +
    `so likely ${rangePercent(r)} of them are: enough for ${target}.` +
    staleNote(r)
  );
}
