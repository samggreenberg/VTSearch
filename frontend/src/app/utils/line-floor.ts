import type { FloorState } from '../generated/api-client/models/floor-state';

/**
 * What the precision floor says about the line a detector draws (#4272, #4273).
 *
 * The line always keeps a set: the top `count` unvoted items of the ranking.
 * It never falls back to a default cut.
 *
 * - `unchecked`: no spot check has run at this floor. The set is the floor's
 *   starting candidate (the top 128 at 10%, 64 at 25%, 32 at 50% and above),
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

/** What a spot check at a floor costs: its starting candidate, its rounds, and the picks a round draws. */
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
export function floorPercent(p: number): string {
  return `${Math.round(p * 100)}%`;
}

/** "11–73%" for a range. */
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

/** The picks a check deals, in words: "5 random picks", plus its rounds when it can take more than one. */
function checkCost(schedule: CheckSchedule | null): string {
  if (!schedule) return 'a few random picks';
  const picks = `${schedule.picks} random picks`;
  return schedule.rounds > 1 ? `${picks} a round, in up to ${schedule.rounds} rounds` : picks;
}

/**
 * The floor's state in one short line, for the control under the picker
 * (#4246, #4273): the floor, what the line keeps, and how close the check got.
 * The range comes only from the check's picks, never from the model. Null
 * when there is no verdict to report (no line yet, or a sort with no detector
 * behind it). A stale range reads exactly as a current one: only
 * {@link floorExplanation} says so.
 */
export function floorSummary(floor: LineFloor | null): string | null {
  if (!floor) return null;
  const target = floorPercent(floor.minPrecision);
  const kept = floor.count.toLocaleString();
  const r = floor.range;
  switch (floor.status) {
    case 'confirmed':
      return r
        ? `At least ${target} right · likely ${rangePercent(r)} (checked ${r.labelled}) · ${kept} kept`
        : `At least ${target} right · ${kept} kept`;
    case 'short':
      // A short check names no cause (owner, 2026-09-29): the app can't tell a
      // sparse corpus from a weak model, so the words are true of both.
      return r
        ? `Aimed at ${target}: likely ${rangePercent(r)} right (checked ${r.labelled}) · top ${kept} kept`
        : `Aimed at ${target}: fell short · top ${kept} kept`;
    case 'unchecked':
      return `Top ${kept} kept, unchecked · aiming at ${target}`;
  }
}

/**
 * The same state at tooltip length: what the short line means, how the range
 * was measured, and whether later votes have moved the list since. Null when
 * {@link floorSummary} is.
 */
export function floorExplanation(floor: LineFloor | null): string | null {
  if (!floor) return null;
  const target = floorPercent(floor.minPrecision);
  const kept = floor.count.toLocaleString();
  const r = floor.range;
  if (floor.status === 'unchecked' || !r) {
    if (floor.status === 'unchecked') {
      return (
        `Unchecked: the line keeps the top ${kept}, and nothing has measured how much of it is right. ` +
        `A check of ${checkCost(floor.schedule)} from it finds out.`
      );
    }
    return floor.status === 'confirmed'
      ? `At least ${target} of the ${kept} items the line keeps is right.`
      : `Aimed at ${target}: the check fell short, and the line keeps the top ${kept} it ended on.`;
  }
  if (floor.status === 'confirmed') {
    return (
      `A check of ${r.labelled} random picks from the ${kept} items the line keeps found ${r.right} right, ` +
      `so likely ${rangePercent(r)} of them are: at least ${target}.` +
      staleNote(r)
    );
  }
  return (
    `Aimed at ${target}: a check of ${r.labelled} random picks from the top ${kept} the line keeps ` +
    `found ${r.right} right, so likely ${rangePercent(r)} of them are, short of the floor.` +
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
    `Vote on ${cost} from the set the line keeps, to measure how much of it is right. ` +
    `Your votes count as ordinary votes.`
  );
}

/** A range's own tooltip, for a chart that draws it (#4273): how it was measured, and whether it is stale. */
export function rangeTitle(range: LikelyRange): string {
  return (
    `Likely ${rangePercent(range)} right, from ${range.labelled} random picks (${range.right} right).` +
    staleNote(range)
  );
}
