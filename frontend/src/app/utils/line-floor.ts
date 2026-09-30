import type { FloorState } from '../generated/api-client/models/floor-state';

/**
 * What the precision floor says about the line a detector draws (#4272, #4273).
 *
 * The line always keeps a set: the top `count` unvoted items of the ranking.
 * It never falls back to a default cut.
 *
 * - `unchecked`: no spot check has run at this floor. The set is the floor's
 *   starting candidate (the top 128 at Complete, 32 at Centered and Correct),
 *   and nothing has measured how much of it is right.
 * - `confirmed`: the last check's likely range clears the floor.
 * - `short`: the check ended below the floor; the line keeps the top 32 it
 *   ended on, and the range says how close it got.
 *
 * Every match, count and action works on the line in every state. The state
 * and its range show in the floor control; the line itself is drawn the same
 * in all three.
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

/**
 * What a spot check at a floor costs: the count the walk starts from, the
 * bands it audits before its first verdict, and the picks each band draws
 * (#4388). Beyond that the walk goes as deep as the list stays right.
 */
export interface CheckSchedule {
  candidate: number;
  rounds: number;
  picks: number;
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
  /** What a check at this floor would cost; null when the response carried none. */
  schedule: CheckSchedule | null;
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
  const schedule = wire.schedule;
  return {
    minPrecision: wire.min_precision,
    status,
    count: wire.count ?? 0,
    range: range
      ? { lo: range.lo, hi: range.hi, labelled: range.labelled, right: range.right, stale: range.stale ?? false }
      : null,
    schedule: schedule ? { candidate: schedule.candidate, rounds: schedule.rounds, picks: schedule.picks } : null,
  };
}

/** One floor the control offers, and what its radio says when pointed at. */
export interface FloorPreset {
  /** The floor, a fraction. */
  value: number;
  /**
   * Its radio's tooltip. The control names no floor and shows no number
   * (#4317): a floor is where its radio sits on the False Positives - False
   * Negatives spectrum, and only this says it in words.
   */
  hint: string;
}

/**
 * The floors the control offers, left to right along its spectrum (#4298,
 * #4317): three radios under the thirds of a False Positives - False Negatives
 * bar, with no word or number on any of them. The spot check rarely delivers a
 * floor exactly, so a percentage would claim a precision it could not keep; a
 * place on the spectrum promises only a direction. What a check measures (its
 * likely range) stays a number. The backend takes any value in `[0.01, 1]`;
 * the control snaps one outside this list to the nearest (see
 * {@link nearestFloorPreset}).
 */
export const FLOOR_PRESETS: readonly FloorPreset[] = [
  { value: 0.1, hint: 'Toward false positives: return the most, with more wrong ones in it' },
  { value: 0.5, hint: 'Between the two' },
  { value: 0.9, hint: 'Toward false negatives: return only the surest, and miss more' },
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

/** "11–73%" for a range: what a check measured stays a number (#4298). */
export function rangePercent(range: Pick<LikelyRange, 'lo' | 'hi'>): string {
  return `${Math.round(range.lo * 100)}–${Math.round(range.hi * 100)}%`;
}

/**
 * The sentence a stale range's tooltip adds; empty when the range is current.
 * A stale range is drawn exactly as a current one (owner, 2026-09-29): this
 * sentence is the only difference.
 */
export function staleNote(range: LikelyRange | null): string {
  return range?.stale ? ' Measured before your later votes: the list at the line has changed since.' : '';
}

/** The picks a check deals, in words: "5 random picks a band, walking the list from the top 32". */
function checkCost(schedule: CheckSchedule | null): string {
  if (!schedule) return 'a few random picks';
  return `${schedule.picks} random picks a band, walking the list from the top ${schedule.candidate.toLocaleString()}`;
}

/**
 * The floor's state in one short line, for the control under the spectrum
 * (#4246, #4273): what the line keeps, and how close the check got. The floor
 * itself goes unnamed (#4317): the radio above the line already shows it. The
 * range comes only from the check's picks, never from the model. Null when
 * there is no verdict to report (no line yet, or a sort with no detector
 * behind it). A stale range reads exactly as a current one: only
 * {@link floorExplanation} says so.
 */
export function floorSummary(floor: LineFloor | null): string | null {
  if (!floor) return null;
  const kept = floor.count.toLocaleString();
  const r = floor.range;
  switch (floor.status) {
    case 'confirmed':
      return r
        ? `Confirmed · likely ${rangePercent(r)} right (checked ${r.labelled}) · ${kept} kept`
        : `Confirmed · ${kept} kept`;
    case 'short':
      // A short check names no cause (owner, 2026-09-29): the app can't tell a
      // sparse corpus from a weak model, so the words are true of both.
      return r
        ? `Fell short · likely ${rangePercent(r)} right (checked ${r.labelled}) · top ${kept} kept`
        : `Fell short · top ${kept} kept`;
    case 'unchecked':
      return `Top ${kept} kept, unchecked`;
  }
}

/**
 * The same state at tooltip length: what the short line means, how the range
 * was measured, and whether later votes have moved the list since. Null when
 * {@link floorSummary} is. It points at no check: Find, where it also shows,
 * offers none (#4317).
 */
export function floorExplanation(floor: LineFloor | null): string | null {
  if (!floor) return null;
  const kept = floor.count.toLocaleString();
  const r = floor.range;
  if (floor.status === 'unchecked') {
    return `Unchecked: the line keeps the top ${kept}, and nothing has measured how much of it is right.`;
  }
  if (!r) {
    return floor.status === 'confirmed'
      ? `A check confirmed the ${kept} items the line keeps.`
      : `The check fell short, and the line keeps the top ${kept} it ended on.`;
  }
  if (floor.status === 'confirmed') {
    return (
      `A check of ${r.labelled} random picks from the ${kept} items the line keeps found ${r.right} right, ` +
      `so likely ${rangePercent(r)} of them are: enough for the threshold.` +
      staleNote(r)
    );
  }
  return (
    `A check of ${r.labelled} random picks from the top ${kept} the line keeps ` +
    `found ${r.right} right, so likely ${rangePercent(r)} of them are: short of the threshold.` +
    staleNote(r)
  );
}

/**
 * The floor control's check affordance (#4273): "Check 5 picks", with the
 * picks a round draws at this floor. It starts a check in every state; after
 * a finished one it runs a fresh check. Null with no line to check.
 */
export function checkLabel(floor: LineFloor | null): string | null {
  if (!floor) return null;
  return floor.schedule ? `Check ${floor.schedule.picks} picks` : 'Check the line';
}

/** The check affordance's tooltip: what a check does, and what it costs at this floor. */
export function checkTitle(floor: LineFloor | null): string {
  const cost = checkCost(floor?.schedule ?? null);
  return (
    `Vote on ${cost}: the check goes deeper while the list stays right enough and shorter while it does not, ` +
    `and the line keeps the deepest set that was. Your votes count as ordinary votes.`
  );
}

/** A range's own tooltip, for a chart that draws it (#4273): how it was measured, and whether it is stale. */
export function rangeTitle(range: LikelyRange): string {
  return (
    `Likely ${rangePercent(range)} right, from ${range.labelled} random picks (${range.right} right).` +
    staleNote(range)
  );
}
