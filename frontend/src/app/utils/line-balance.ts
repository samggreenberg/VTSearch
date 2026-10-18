import type { BalanceState } from '../generated/api-client/models/balance-state';

/**
 * What the balance says about the line a detector draws (#4413; the line and
 * its check are #4272's and #4273's).
 *
 * The line always keeps a set: the top `count` unvoted items of the ranking.
 * It never falls back to a default cut. A structural detector's line is the
 * exception (`gate`, below): it is drawn by the verification gate, not on a
 * ranking.
 *
 * - `unchecked`: no spot check has run at this balance. The set is the
 *   balance's starting candidate, and nothing has measured how much of it is
 *   right.
 * - `checked`: a check walked the ranking and ended on the set where its
 *   estimate of the balance (F-beta) peaked. The two ranges say what its
 *   picks found in the set the line kept when it started (`keptAtCheck`,
 *   #4483).
 * - `gate`: a structural detector's line (#4505), the verification gate's
 *   boundary. The set is the `count` unvoted items the gate passes; a check
 *   needs a ranking to sample, so none applies and there are no ranges.
 *
 * Nothing is met or fallen short of: a check just says what it estimated.
 * Every match, count and action works on the line in both states. The state
 * and its ranges show in the balance control; the line itself is drawn the
 * same in both.
 */
export type BalanceStatus = 'unchecked' | 'checked' | 'gate';

/**
 * How a check treats the line (#4427, #4452): `advisory` - the walk audits and
 * reports, its votes stay votes, and the line stays where the labels put it.
 * Since #4452 the server sends `advisory` at every balance, because the line
 * comes from the labels alone; `trim` (the walk may only step shallower, and
 * the line keeps its end) is kept for a server that still sends it.
 */
export type BalanceShape = 'advisory' | 'trim';

/**
 * A likely share, from the check's picks alone: of the set the line kept when
 * the check started that is right (precision), or of all the matches that are
 * in it (recall); #4483, before which they described the audited set. Both
 * cover only the items the user had not voted on when the check ran (#4914):
 * the walk samples the unvoted ranking, and recall's denominator is the
 * mixture's count of the positives left in it, so the user's own Goods are in
 * neither.
 */
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
 * What a spot check at a balance costs: the count the walk starts from, the
 * bands it audits before its first step, and the picks each band draws
 * (#4388). Beyond that the walk goes as deep as the balance keeps improving.
 */
export interface CheckSchedule {
  candidate: number;
  rounds: number;
  picks: number;
}

/** The balance's state on the current line, as the sort state holds it. */
export interface LineBalance {
  /** The detector's balance, F-beta's beta. Every detector has one. */
  beta: number;
  status: BalanceStatus;
  /** How many unvoted items the line keeps. */
  count: number;
  /** The check's likely share of the set the line kept when it started that is right; null while unchecked. */
  precision: LikelyRange | null;
  /** The check's likely share of all the unvoted matches that are in that set; null while unchecked. */
  recall: LikelyRange | null;
  /** The check's F-beta estimate for the audited set; null while unchecked. */
  fbeta: number | null;
  /** What a check at this balance would cost; null when the response carried none. */
  schedule: CheckSchedule | null;
  /** How a check treats the line at this balance. */
  shape: BalanceShape;
  /** The set the last check audited (the walk's end); under `advisory` not the set kept. Null while unchecked. */
  audited: number | null;
  /**
   * How many unvoted items the line kept when the last check started: the set the ranges describe (#4483).
   * Null while unchecked, and from a server before #4483, whose ranges describe `audited`.
   */
  keptAtCheck: number | null;
  /**
   * Whether a spot check can start on this line (#4489). False when the
   * detector has no ranking to walk - a structural detector, whose line is the
   * verification gate's boundary rather than a cut on a ranking - or nothing in
   * it is left unvoted. The server refuses a check there, so none is offered.
   */
  checkable: boolean;
  /** How far apart the labels' Good and Bad scores sit, in spreads (d'); null before a retrain has drawn the labels' line. */
  separation: number | null;
  /**
   * The labels separate weakly enough that a spot check is due (#4496): Autopilot runs one. The
   * server decides (`weak_check_due`): d' below 1.5, from 10 votes on, and only while no check has
   * ended, so once a session (#4909).
   */
  checkDue: boolean;
  /**
   * Autopilot's check before Done should run (`done_check_due`): no check has ended in the last 25
   * votes. Read once Autopilot reaches the step (every quality light green, Done's level met, or
   * the click cap); false skips it.
   */
  doneCheckDue: boolean;
}

const STATUSES: readonly BalanceStatus[] = ['unchecked', 'checked', 'gate'];

function likelyRangeFrom(range: BalanceState['precision']): LikelyRange | null {
  return range
    ? { lo: range.lo, hi: range.hi, labelled: range.labelled, right: range.right, stale: range.stale ?? false }
    : null;
}

/**
 * The wire `balance` object, as a {@link LineBalance}; null when the response
 * carried none (a sort with no detector behind it), or a status outside the
 * three states.
 */
export function lineBalanceFrom(wire: BalanceState | null | undefined): LineBalance | null {
  if (!wire) return null;
  const status = STATUSES.find((s) => s === wire.status);
  if (!status) return null;
  const schedule = wire.schedule;
  return {
    beta: wire.beta,
    status,
    count: wire.count ?? 0,
    precision: likelyRangeFrom(wire.precision),
    recall: likelyRangeFrom(wire.recall),
    fbeta: wire.fbeta ?? null,
    schedule: schedule ? { candidate: schedule.candidate, rounds: schedule.rounds, picks: schedule.picks } : null,
    shape: wire.shape === 'advisory' ? 'advisory' : 'trim',
    audited: wire.audited ?? null,
    keptAtCheck: wire.kept_at_check ?? null,
    // A server from before #4489 sends no flag, and offered a check everywhere.
    checkable: wire.checkable !== false,
    separation: wire.separation ?? null,
    checkDue: wire.check_due ?? false,
    doneCheckDue: wire.done_check_due ?? false,
  };
}

/** One balance the control offers, and what its radio says when pointed at. */
export interface BalancePreset {
  /** The balance, F-beta's beta. */
  value: number;
  /**
   * Its radio's tooltip. The control names no balance and shows no number
   * (#4317): a balance is where its radio sits on the False Positives - False
   * Negatives spectrum, and only this says it in words.
   */
  hint: string;
}

/**
 * The balances the control offers, left to right along its spectrum (#4413;
 * the spectrum is #4298's and #4317's): three radios under the thirds of a
 * False Positives - False Negatives bar, with no word or number on any of
 * them. Beta 4 leans to recall (toward False Positives: the most returned,
 * with more wrong ones in it), 1 is balanced, 1/4 leans to precision (toward
 * False Negatives: only the surest, and more missed). The ends are the owner's
 * pick of 2026-10-03 on #4448, priced on #4452's line: past 1/3 and 3 each end
 * buys at most 0.02 more of what it leans toward, and 1/4 and 4 are the ends
 * of the range the backend accepts (they were 0.5 and 2 before). A place on the spectrum
 * promises only a direction; what a check measures (its likely ranges) stays
 * a number. The backend takes any positive beta; the control snaps one
 * outside this list to the nearest (see {@link nearestBalancePreset}).
 */
export const BALANCE_PRESETS: readonly BalancePreset[] = [
  { value: 4, hint: 'Toward false positives: return the most, with more wrong ones in it' },
  { value: 1, hint: 'Between the two' },
  { value: 0.25, hint: 'Toward false negatives: return only the surest, and miss more' },
];

/**
 * The balance the control shows before the detector's own value arrives from
 * `GET /api/balance`. It is the backend's default (balanced, F1), and only a
 * placeholder: nothing is sent until the user picks a balance.
 */
export const DEFAULT_BETA = 1;

/** True when `b` is one of the balances the control offers. */
export function isBalancePreset(b: number): boolean {
  return BALANCE_PRESETS.some((preset) => preset.value === b);
}

/**
 * The preset closest to `b` in log space (beta is a ratio: 4 is as far from 1
 * as 1/4 is), for a stored balance the control does not offer (one set
 * through the CLI or the API, or a preset from before #4448). A tie goes to
 * the preset farther from the balanced middle, so a stored lean keeps its
 * side: 2 and 0.5, the presets before #4448, sit exactly halfway in log space
 * and show as 4 and 1/4, not as balanced. A beta that is not a positive number
 * has no log, and shows as the balanced default.
 */
export function nearestBalancePreset(b: number): BalancePreset {
  if (!(b > 0)) return BALANCE_PRESETS.find((preset) => preset.value === DEFAULT_BETA)!;
  const target = Math.log(b);
  const distance = (preset: BalancePreset) => Math.abs(Math.log(preset.value) - target);
  const lean = (preset: BalancePreset) => Math.abs(Math.log(preset.value));
  return BALANCE_PRESETS.reduce((best, preset) => {
    const closer = distance(preset) - distance(best);
    // A tie (within floating point) goes to the preset that leans further.
    return closer < -1e-9 || (closer <= 1e-9 && lean(preset) > lean(best)) ? preset : best;
  });
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

/**
 * The recall range in words, from its midpoint: how many of all the matches
 * the checked set likely holds. Words rather than a second percentage, so the
 * state line reads as a sentence; the number is in the tooltip
 * ({@link balanceExplanation}).
 */
export function foundWords(recall: LikelyRange): string {
  const mid = (recall.lo + recall.hi) / 2;
  if (mid < 0.15) return 'few of them found';
  if (mid < 0.375) return 'about a quarter of them found';
  if (mid < 0.625) return 'about half of them found';
  if (mid < 0.875) return 'about three quarters of them found';
  return 'nearly all of them found';
}

/**
 * The balance's state in one short line, for the control under the spectrum
 * (#4413; the line is #4246's and #4273's): what the line keeps, and what the
 * check found there. The balance itself goes unnamed (#4317): the radio above
 * the line already shows it. The ranges come only from the check's picks,
 * never from the model. Null when there is no state to report (no line yet,
 * or a sort with no detector behind it). A stale range reads exactly as a
 * current one: only {@link balanceExplanation} says so.
 *
 * Nothing in it counts the user's own votes, and it says so (#4914): the line
 * keeps "N more" beside them, and a check names the set its ranges cover:
 * since #4483 the ones the line kept when it started (`keptAtCheck`), which
 * the line's count has moved from once the check's own votes are cast; from a
 * server before it, the top `audited`. Late in a session, once the matches
 * near the top are voted, that set holds few of them, so the ranges can read
 * low after a session that went well.
 */
export function balanceSummary(balance: LineBalance | null): string | null {
  if (!balance) return null;
  const kept = balance.count.toLocaleString();
  if (balance.status === 'unchecked') return `${kept} more kept, unchecked`;
  if (balance.status === 'gate') return `${kept} pass the verification gate`;
  const p = balance.precision;
  const r = balance.recall;
  if (!p || !r) return `Checked · ${kept} more kept`;
  return (
    `Checked ${rangedSetWords(balance)} · likely ${rangePercent(p)} right, ` +
    `${foundWords(r)} (checked ${p.labelled}) · ${kept} more kept`
  );
}

/** The set a check's ranges describe, named: the ones the line kept when it started (#4483), or the top it audited. */
export function rangedSetWords(balance: LineBalance): string {
  if (balance.keptAtCheck !== null) return `the ${balance.keptAtCheck.toLocaleString()} kept you hadn't voted on`;
  return `the top ${(balance.audited ?? balance.count).toLocaleString()} you hadn't voted on`;
}

/**
 * The same state at tooltip length: what the short line means, how the
 * ranges were measured, and whether later votes have moved the list since.
 * Null when {@link balanceSummary} is. It points at no check: Find, where it
 * also shows, offers none (#4317).
 */
export function balanceExplanation(balance: LineBalance | null): string | null {
  if (!balance) return null;
  const kept = balance.count.toLocaleString();
  if (balance.status === 'unchecked') {
    return `Unchecked: the line keeps the top ${kept} you haven't voted on, and nothing has measured how many of them are right.`;
  }
  if (balance.status === 'gate') {
    return (
      `The verification gate draws this line: ${kept} items match one of your Good examples geometrically, ` +
      `not counting the ones you voted on. A spot check samples a ranked list, and this line has none, so none applies.`
    );
  }
  const p = balance.precision;
  const r = balance.recall;
  const audited = (balance.audited ?? balance.count).toLocaleString();
  if (balance.shape === 'advisory' && balance.keptAtCheck !== null) {
    const ranged = balance.keptAtCheck.toLocaleString();
    if (balance.keptAtCheck === 0) return `When a check started, the line kept none you hadn't voted on; it keeps the top ${kept} now.`;
    if (!p || !r) return `When a check started, the line kept the top ${ranged} you hadn't voted on; it keeps the top ${kept} now.`;
    return (
      `When the check started, the line kept the top ${ranged} you hadn't voted on. ${p.labelled} of the check's random picks ` +
      `fell among them and ${p.right} were right, so likely ${rangePercent(p)} of them are, ` +
      `with likely ${rangePercent(r)} of all the matches you hadn't voted on among them. ` +
      `The line keeps the top ${kept} you haven't voted on now, where your labels put it: a check informs the line and does not move it.` +
      staleNote(p)
    );
  }
  if (balance.shape === 'advisory') {
    if (!p || !r) return `A check ended on the top ${audited} you hadn't voted on; the line keeps the top ${kept}.`;
    return (
      `A check of ${p.labelled} random picks from the top ${audited} you hadn't voted on found ${p.right} right, ` +
      `so likely ${rangePercent(p)} of them are, with likely ${rangePercent(r)} of all the matches you hadn't voted on among them. ` +
      `The line keeps the top ${kept} you haven't voted on, where your labels put it: a check informs the line and does not move it.` +
      staleNote(p)
    );
  }
  if (!p || !r) return `A check ended on the top ${kept} you hadn't voted on, which the line keeps.`;
  return (
    `A check of ${p.labelled} random picks from the top ${kept} you hadn't voted on, which the line keeps, found ${p.right} right, ` +
    `so likely ${rangePercent(p)} of them are, with likely ${rangePercent(r)} of all the matches you hadn't voted on among them: ` +
    `the set where the check's balance peaked.` +
    staleNote(p)
  );
}

/**
 * Why a check is due, when the server says one is (#4496): the labels separate weakly, and a check's
 * picks, drawn evenly down the list, are the votes that teach the detector where its line falls.
 * Only Autopilot runs a check, so the note points there. Null when none is due.
 */
export function checkDueNote(balance: LineBalance | null): string | null {
  if (!balance?.checkDue) return null;
  return 'Your Good and Bad labels still overlap. Autopilot runs a spot check now: its picks teach the detector where its line falls.';
}

/** A range's own tooltip, for a chart that draws it (#4273): how it was measured, and whether it is stale. */
export function rangeTitle(range: LikelyRange): string {
  return (
    `Likely ${rangePercent(range)} right, from ${range.labelled} random picks (${range.right} right).` +
    staleNote(range)
  );
}
