import type { BalanceStatus, CheckSchedule, LikelyRange, LineBalance } from '../utils/line-balance';

/** The two states the balance reports (#4413). */
export const BALANCE_STATES: BalanceStatus[] = ['unchecked', 'checked'];

/** A five-pick precision range with 2 of 5 right. */
export const SAMPLE_RANGE: LikelyRange = { lo: 0.11, hi: 0.73, labelled: 5, right: 2, stale: false };

/** The precision range a checked line typically carries: 5 of 5 right. */
export const CHECKED_PRECISION: LikelyRange = { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: false };

/** The recall range beside it: likely about half of all the matches are in the kept set. */
export const CHECKED_RECALL: LikelyRange = { lo: 0.3, hi: 0.7, labelled: 5, right: 5, stale: false };

/** The schedule at the balanced default: the top 32, one round of 5 picks. */
export const SCHEDULE_DEFAULT: CheckSchedule = { candidate: 32, rounds: 1, picks: 5 };

/**
 * A balance state as the sort state holds it: the balanced default (beta 1)
 * keeping the top 32, with a check's ranges once checked.
 */
export function lineBalance(status: BalanceStatus, overrides: Partial<LineBalance> = {}): LineBalance {
  const checked = status === 'checked';
  return {
    beta: 1,
    status,
    count: 32,
    precision: checked ? CHECKED_PRECISION : null,
    recall: checked ? CHECKED_RECALL : null,
    fbeta: checked ? 0.67 : null,
    schedule: SCHEDULE_DEFAULT,
    ...overrides,
  };
}

/** The same state as the wire `balance` object a response carries. */
export function wireBalance(status: BalanceStatus, overrides: Partial<LineBalance> = {}) {
  const b = lineBalance(status, overrides);
  return {
    beta: b.beta,
    status: b.status,
    count: b.count,
    precision: b.precision,
    recall: b.recall,
    fbeta: b.fbeta,
    schedule: b.schedule ?? SCHEDULE_DEFAULT,
  };
}
