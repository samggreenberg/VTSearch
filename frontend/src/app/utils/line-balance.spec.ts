import type { BalanceState } from '../generated/api-client/models/balance-state';
import { CHECKED_PRECISION, CHECKED_RECALL, SAMPLE_RANGE, lineBalance } from '../testing/line-balance';
import {
  BALANCE_PRESETS,
  DEFAULT_BETA,
  balanceExplanation,
  balanceSummary,
  checkDueNote,
  checkLabel,
  checkTitle,
  foundWords,
  isBalancePreset,
  lineBalanceFrom,
  nearestBalancePreset,
  rangePercent,
  rangeTitle,
  staleNote,
  type BalanceStatus,
} from './line-balance';

describe('line-balance (#4413)', () => {
  describe('lineBalanceFrom', () => {
    it('reads the wire object, both ranges and the schedule included', () => {
      const wire: BalanceState = {
        beta: 0.5,
        status: 'checked',
        count: 64,
        precision: { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: true },
        recall: { lo: 0.3, hi: 0.7, labelled: 5, right: 5 },
        fbeta: 0.71,
        schedule: { candidate: 64, rounds: 2, picks: 5 },
        shape: 'advisory',
        audited: 128,
      };
      expect(lineBalanceFrom(wire)).toEqual({
        beta: 0.5,
        status: 'checked',
        count: 64,
        precision: { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: true },
        recall: { lo: 0.3, hi: 0.7, labelled: 5, right: 5, stale: false },
        fbeta: 0.71,
        schedule: { candidate: 64, rounds: 2, picks: 5 },
        shape: 'advisory',
        audited: 128,
        separation: null,
        checkDue: false,
      });
    });

    it('reads an unchecked line with no ranges and no estimate', () => {
      const wire = { beta: 2, status: 'unchecked', count: 128, precision: null, recall: null, fbeta: null, schedule: { candidate: 128, rounds: 3, picks: 5 }, shape: 'trim', audited: null };
      expect(lineBalanceFrom(wire as unknown as BalanceState)).toEqual({
        beta: 2,
        status: 'unchecked',
        count: 128,
        precision: null,
        recall: null,
        fbeta: null,
        schedule: { candidate: 128, rounds: 3, picks: 5 },
        shape: 'trim',
        audited: null,
        separation: null,
        checkDue: false,
      });
    });

    it('reads the labels\' separation and whether a check is due (#4496)', () => {
      const wire = { beta: 1, status: 'unchecked', count: 40, precision: null, recall: null, fbeta: null, schedule: null, shape: 'advisory', audited: null, separation: 0.83, check_due: true };
      const balance = lineBalanceFrom(wire as unknown as BalanceState)!;
      expect(balance.separation).toBe(0.83);
      expect(balance.checkDue).toBe(true);
      expect(checkDueNote(balance)).toContain('still overlap');
      expect(checkDueNote({ ...balance, checkDue: false })).toBeNull();
      expect(checkDueNote(null)).toBeNull();
    });

    it('reads a range with no stale flag as current, and no schedule as none', () => {
      const wire = { beta: 1, status: 'checked', count: 32, precision: { lo: 0.11, hi: 0.73, labelled: 5, right: 2 }, recall: null, fbeta: 0.4, schedule: null };
      const balance = lineBalanceFrom(wire as unknown as BalanceState)!;
      expect(balance.precision!.stale).toBe(false);
      expect(balance.recall).toBeNull();
      expect(balance.schedule).toBeNull();
      expect(balance.shape).toBe('trim');
      expect(balance.audited).toBeNull();
    });

    it('is null for a response with no detector behind it', () => {
      expect(lineBalanceFrom(null)).toBeNull();
      expect(lineBalanceFrom(undefined)).toBeNull();
    });

    it('reads a status outside the two states as no state: the floor-era verdicts are gone', () => {
      for (const status of ['confirmed', 'short', 'promised']) {
        const wire = { beta: 1, status, count: 32, precision: null, recall: null, fbeta: null, schedule: { candidate: 32, rounds: 1, picks: 5 } };
        expect(lineBalanceFrom(wire as unknown as BalanceState)).toBeNull();
      }
    });
  });

  describe('the balance presets (#4413, #4298, #4317, #4448)', () => {
    it('offers three balances, left to right from false positives to false negatives, symmetric about the balanced middle', () => {
      expect(BALANCE_PRESETS.map((p) => p.value)).toEqual([4, 1, 0.25]);
      expect(BALANCE_PRESETS[0].hint).toMatch(/^Toward false positives/);
      expect(BALANCE_PRESETS[1].hint).toBe('Between the two');
      expect(BALANCE_PRESETS[2].hint).toMatch(/^Toward false negatives/);
      expect(DEFAULT_BETA).toBe(1);
    });

    it('knows a preset from a balance off the list', () => {
      expect(BALANCE_PRESETS.every((p) => isBalancePreset(p.value))).toBe(true);
      expect(isBalancePreset(0.1)).toBe(false);
      expect(isBalancePreset(0.9)).toBe(false);
      expect(isBalancePreset(1.5)).toBe(false);
      // The presets before #4448.
      expect(isBalancePreset(0.5)).toBe(false);
      expect(isBalancePreset(2)).toBe(false);
    });

    it.each<[number, number]>([
      [8, 4],
      [4, 4],
      [2.4, 4],
      [1.9, 1],
      [1, 1],
      [0.6, 1],
      [0.45, 0.25],
      [0.25, 0.25],
      [0.01, 0.25],
    ])('snaps %s to the nearest preset in log space, %s', (stored, snapped) => {
      expect(nearestBalancePreset(stored).value).toBe(snapped);
    });

    it('measures in log space: 2.4 is nearer 4 than 1, and 0.6 nearer 1 than 1/4', () => {
      // In linear terms each is nearer the other preset; as a ratio it is not.
      expect(nearestBalancePreset(2.4).value).toBe(4);
      expect(nearestBalancePreset(0.6).value).toBe(1);
    });

    it('breaks a tie toward the preset that leans further, so the old presets keep their side', () => {
      // The log-space midpoints are 2 (between 4 and 1) and 0.5 (between 1 and 1/4): the presets before #4448.
      expect(nearestBalancePreset(2).value).toBe(4);
      expect(nearestBalancePreset(0.5).value).toBe(0.25);
    });

    it('shows a beta with no log as the balanced default', () => {
      expect(nearestBalancePreset(0).value).toBe(1);
      expect(nearestBalancePreset(-1).value).toBe(1);
      expect(nearestBalancePreset(NaN).value).toBe(1);
    });
  });

  describe('the found words', () => {
    it.each<[number, number, string]>([
      [0, 0.2, 'few of them found'],
      [0.1, 0.4, 'about a quarter of them found'],
      [0.3, 0.7, 'about half of them found'],
      [0.6, 0.9, 'about three quarters of them found'],
      [0.8, 1, 'nearly all of them found'],
    ])('reads a %s–%s recall range at its midpoint as "%s"', (lo, hi, words) => {
      expect(foundWords({ lo, hi, labelled: 5, right: 3, stale: false })).toBe(words);
    });

    it('cuts the bands at 15, 37.5, 62.5 and 87.5 percent', () => {
      const at = (mid: number) => foundWords({ lo: mid, hi: mid, labelled: 1, right: 1, stale: false });
      expect(at(0.149)).toBe('few of them found');
      expect(at(0.15)).toBe('about a quarter of them found');
      expect(at(0.374)).toBe('about a quarter of them found');
      expect(at(0.375)).toBe('about half of them found');
      expect(at(0.624)).toBe('about half of them found');
      expect(at(0.625)).toBe('about three quarters of them found');
      expect(at(0.874)).toBe('about three quarters of them found');
      expect(at(0.875)).toBe('nearly all of them found');
    });
  });

  describe('the balance control copy', () => {
    it("shows a check's range as a number", () => {
      expect(rangePercent(SAMPLE_RANGE)).toBe('11–73%');
    });

    it.each<[BalanceStatus, string]>([
      ['checked', 'Checked · likely 55–100% right, about half of them found (checked 5) · 32 kept'],
      ['unchecked', 'Top 32 kept, unchecked'],
    ])('summarises %s', (status, expected) => {
      expect(balanceSummary(lineBalance(status))).toBe(expected);
    });

    it('summarises a checked line with no ranges by its count alone', () => {
      expect(balanceSummary(lineBalance('checked', { precision: null, recall: null }))).toBe('Checked · 32 kept');
      expect(balanceSummary(lineBalance('checked', { recall: null }))).toBe('Checked · 32 kept');
    });

    it.each<BalanceStatus>(['checked', 'unchecked'])('neither names nor numbers the balance (%s, #4298, #4317)', (status) => {
      for (const beta of [2, 1, 0.5, 1.5]) {
        const balance = lineBalance(status, { beta });
        for (const text of [balanceSummary(balance)!, balanceExplanation(balance)!]) {
          expect(text).not.toMatch(/beta|F-?beta|F1|F2|balance[d]? (at|of) \d/i);
          expect(text).not.toMatch(/Centered|Complete|Correct|floor|threshold/i);
          expect(text).not.toMatch(/\b0\.5\b/);
        }
      }
    });

    it('reads the count from the result, never from the preset', () => {
      expect(balanceSummary(lineBalance('checked', { beta: 2, count: 64 }))).toContain('64 kept');
      expect(balanceSummary(lineBalance('checked', { beta: 2, count: 128 }))).toContain('128 kept');
      expect(balanceSummary(lineBalance('unchecked', { beta: 2, count: 128 }))).toBe('Top 128 kept, unchecked');
    });

    it('carries the precision range wide as the check left it, and the recall in words', () => {
      const text = balanceSummary(lineBalance('checked', { precision: SAMPLE_RANGE }))!;
      expect(text).toContain('11–73%');
      expect(text).toContain('(checked 5)');
      expect(text).toContain('about half of them found');
      expect(text).not.toContain('30–70%');
    });

    it('has nothing to summarise, explain or check without a state', () => {
      expect(balanceSummary(null)).toBeNull();
      expect(balanceExplanation(null)).toBeNull();
      expect(checkLabel(null)).toBeNull();
    });

    it('explains a trimmed line (beta above 1) by its check, both ranges as numbers, as the set where the balance peaked', () => {
      const why = balanceExplanation(lineBalance('checked', { beta: 2 }))!;
      expect(why).toBe(
        'A check of 5 random picks from the 32 items the line keeps found 5 right, so likely 55–100% of them are, ' +
          'with likely 30–70% of all the matches among them: the set where the check\'s balance peaked.',
      );
      expect(why).not.toMatch(/estimate|enough|short/i);
    });

    it('explains an advisory check (beta 1 and below) by the set it audited, and says the line keeps its own count', () => {
      const why = balanceExplanation(lineBalance('checked', { count: 16, audited: 64 }))!;
      expect(why).toContain('A check of 5 random picks from the top 64 found 5 right');
      expect(why).toContain('The line keeps its 16, where your labels put it: a check informs the line and does not move it.');
      expect(why).not.toContain('peaked');
      expect(balanceExplanation(lineBalance('checked', { count: 16, audited: 64, precision: null, recall: null }))).toBe(
        'A check ended on the top 64; the line keeps its 16.',
      );
    });

    it('explains a checked line with no ranges as where the check ended', () => {
      expect(balanceExplanation(lineBalance('checked', { beta: 2, precision: null, recall: null }))).toBe(
        'A check ended on the 32 items the line keeps.',
      );
    });

    it('explains an unchecked line as unmeasured, pointing at no check (Find offers none, #4317)', () => {
      const why = balanceExplanation(lineBalance('unchecked', { count: 128, schedule: { candidate: 128, rounds: 3, picks: 5 } }))!;
      expect(why).toBe('Unchecked: the line keeps the top 128, and nothing has measured how much of it is right.');
      expect(why).not.toMatch(/likely|a check of/i);
    });

    it('says a stale precision range is stale in the tooltip, and only there', () => {
      const fresh = lineBalance('checked');
      const stale = lineBalance('checked', { precision: { ...CHECKED_PRECISION, stale: true }, recall: { ...CHECKED_RECALL, stale: true } });
      expect(balanceSummary(stale)).toBe(balanceSummary(fresh));
      expect(balanceExplanation(stale)).toBe(balanceExplanation(fresh) + staleNote(stale.precision));
      expect(balanceExplanation(stale)).toContain('Measured before your later votes');
      expect(balanceExplanation(fresh)).not.toContain('later votes');
    });
  });

  describe('the check affordance (#4273)', () => {
    it.each<[number, number, string]>([
      [2, 5, 'Check 5 picks'],
      [1, 5, 'Check 5 picks'],
      [0.5, 5, 'Check 5 picks'],
    ])('at beta %s reads "%s picks" off the schedule', (beta, picks, label) => {
      const balance = lineBalance('unchecked', { beta, schedule: { candidate: 32, rounds: 3, picks } });
      expect(checkLabel(balance)).toBe(label);
    });

    it('is offered in both states: after a finished check it runs a fresh one', () => {
      for (const status of ['unchecked', 'checked'] as const) {
        expect(checkLabel(lineBalance(status))).toBe('Check 5 picks');
      }
      expect(checkLabel(lineBalance('unchecked', { schedule: null }))).toBe('Check the line');
    });

    it('says what a check does at this balance (#4427), and that its votes are votes', () => {
      const trimmed = checkTitle(lineBalance('unchecked', { beta: 2, schedule: { candidate: 128, rounds: 5, picks: 5 } }));
      expect(trimmed).toBe(
        'Vote on 5 random picks a band, walking the list from the top 128: the check steps to a shorter list while the balance ' +
          'does not fall, and the line keeps the set where it ends. Your votes count as ordinary votes.',
      );
      const title = checkTitle(lineBalance('unchecked', { schedule: { candidate: 64, rounds: 4, picks: 5 } }));
      expect(title).toBe(
        'Vote on 5 random picks a band, walking the list from the top 64: the check goes deeper while the balance keeps ' +
          'improving and shorter while it does not, and reports what it found; it informs the line and does not move it. ' +
          'Your votes count as ordinary votes.',
      );
      expect(checkTitle(lineBalance('unchecked', { schedule: null }))).toContain('Vote on a few random picks:');
      expect(checkTitle(null)).toContain('Vote on a few random picks:');
    });
  });

  describe('rangeTitle', () => {
    it('names the picks the range comes from', () => {
      expect(rangeTitle(SAMPLE_RANGE)).toBe('Likely 11–73% right, from 5 random picks (2 right).');
    });

    it('adds the stale sentence for a stale range', () => {
      expect(rangeTitle({ ...SAMPLE_RANGE, stale: true })).toContain('Measured before your later votes');
    });
  });
});
