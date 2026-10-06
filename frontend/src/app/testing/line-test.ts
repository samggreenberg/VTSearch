import type { LineTestEdge } from '../generated/api-client/models/line-test-edge';
import type { LineTestEstimate } from '../generated/api-client/models/line-test-estimate';
import type { LineTestResponse } from '../generated/api-client/models/line-test-response';
import type { LineTestState } from '../generated/api-client/models/line-test-state';
import type { LineTestVerdict } from '../generated/api-client/models/line-test-verdict';
import type { BalanceState } from '../generated/api-client/models/balance-state';
import { wireBalance } from './line-balance';

/** An estimate as the wire carries it. */
export function estimate(point: number, lo: number, hi: number): LineTestEstimate {
  return { point, lo, hi };
}

/** The line re-estimated at a band edge. */
export function edge(count: number, side: 'above' | 'below', precision: LineTestEstimate, recall = estimate(0.5, 0.3, 0.7)): LineTestEdge {
  return { count, side, precision, recall, fbeta: estimate(0.6, 0.4, 0.8), found: 'about half of them found' };
}

/**
 * A test's wire state (#4524): a 200-item ranking whose line keeps the top
 * 64, in the `matches` phase with round one dealt from the band holding the
 * line (ranks 33–64). Override what a case needs.
 */
export function wireTest(overrides: Partial<LineTestState> = {}): LineTestState {
  const picks = overrides.picks ?? [51, 36, 41, 44, 48];
  const phase = overrides.phase ?? 'matches';
  return {
    phase,
    report: {
      phase,
      matches_stop: null,
      misses_stop: null,
      matches_width: 0.6,
      misses_width: 0.5,
      picks_above: 0,
      picks_below: 0,
    },
    beta: 1,
    line_count: 64,
    size: 200,
    round: 1,
    picks_per_round: 5,
    band: { index: 3, side: 'above', lo: 33, hi: 64 },
    picks,
    labelled: 0,
    bands: [
      { index: 0, side: 'above', lo: 1, hi: 8, labelled: 0, right: 0, range: null },
      { index: 1, side: 'above', lo: 9, hi: 16, labelled: 0, right: 0, range: null },
      { index: 2, side: 'above', lo: 17, hi: 32, labelled: 0, right: 0, range: null },
      { index: 3, side: 'above', lo: 33, hi: 64, labelled: 0, right: 0, range: null },
      { index: 4, side: 'below', lo: 65, hi: 72, labelled: 0, right: 0, range: null },
      { index: 5, side: 'below', lo: 73, hi: 80, labelled: 0, right: 0, range: null },
    ],
    estimates: {
      beta: 1,
      precision: estimate(0.75, 0.55, 0.95),
      recall: estimate(0.5, 0.3, 0.7),
      fbeta: estimate(0.6, 0.45, 0.78),
      found: 'about half of them found',
      positives_above: estimate(48, 35, 61),
      positives_below: estimate(48, 20, 80),
      tail_positives: 12,
      tail_from_model: true,
      labelled: 0,
      at_edges: [
        edge(8, 'above', estimate(0.9, 0.7, 1)),
        edge(16, 'above', estimate(0.85, 0.65, 0.98)),
        edge(32, 'above', estimate(0.8, 0.6, 0.96)),
        edge(64, 'above', estimate(0.75, 0.55, 0.95)),
        edge(72, 'below', estimate(0.7, 0.5, 0.9)),
        edge(80, 'below', estimate(0.65, 0.45, 0.85)),
      ],
    },
    budgets: {
      matches_width: 0.2,
      misses_width: 0.25,
      matches_picks: 40,
      misses_picks: 40,
      picks_per_round: 5,
      dry_run_share: 0.05,
      model_weight: 5,
      alpha: 0.05,
    },
    kept_at: null,
    class_model: true,
    ...overrides,
  } as LineTestState;
}

/** A finished test: Done, nothing pending, 65 picks behind the ranges (the walk below the line ran to its budget). */
export function wireDone(overrides: Partial<LineTestState> = {}): LineTestState {
  const base = wireTest({ phase: 'done', picks: [], labelled: 65, round: 13, ...overrides });
  return {
    ...base,
    band: null as unknown as LineTestState['band'],
    report: { ...base.report, phase: 'done', matches_stop: 'width', misses_stop: 'budget', matches_width: 0.18, misses_width: 0.22, picks_above: 25, picks_below: 40 },
    estimates: { ...base.estimates, labelled: 65 },
  };
}

/** A verdict the detector keeps (#4526): drawings-new, 34 picks, likely 70–85% right. */
export function wireVerdict(overrides: Partial<LineTestVerdict> = {}): LineTestVerdict {
  return {
    dataset_id: 'ds-new',
    dataset_name: 'drawings-new',
    tested_at: new Date(2026, 9, 5, 12, 0).getTime() / 1000,
    beta: 1,
    line_count: 64,
    size: 200,
    labelled: 34,
    precision: estimate(0.78, 0.7, 0.85),
    recall: estimate(0.5, 0.38, 0.62),
    fbeta: estimate(0.6, 0.5, 0.7),
    found: 'about half of them found',
    stale: false,
    ...overrides,
  };
}

/** The whole `/api/line-test` body around a test (or none). */
export function wireLineTest(test: LineTestState | null, overrides: Partial<LineTestResponse> = {}): LineTestResponse {
  return {
    balance: wireBalance('unchecked', { count: 64 }) as unknown as BalanceState,
    threshold: 0.68,
    line_count: 64,
    test: test as LineTestState,
    stale: false,
    moved: false,
    presets: [],
    ...overrides,
  };
}
