import { estimate, wireDone, wireLineTest, wireTest, wireVerdict } from '../testing/line-test';
import { estimatePercent, fbetaHeadline, lineTestPhase, testLineState, verdictLine, verdictTitle, widthLight } from './line-test';

describe('line-test helpers (#4524)', () => {
  describe('widthLight', () => {
    it('is red beyond twice the target, yellow within twice, green at or under', () => {
      expect(widthLight(0.5, 0.2)).toBe('red');
      expect(widthLight(0.41, 0.2)).toBe('red');
      expect(widthLight(0.4, 0.2)).toBe('yellow');
      expect(widthLight(0.25, 0.2)).toBe('yellow');
      expect(widthLight(0.2, 0.2)).toBe('green');
      expect(widthLight(0.1, 0.2)).toBe('green');
    });

    it('reads red before the server has reported a width', () => {
      expect(widthLight(null, 0.2)).toBe('red');
      expect(widthLight(undefined, 0.2)).toBe('red');
    });
  });

  it('writes a range as whole percents and F-beta as a point with its range', () => {
    expect(estimatePercent(estimate(0.75, 0.554, 0.946))).toBe('55–95%');
    expect(fbetaHeadline(estimate(0.6, 0.45, 0.78))).toBe('0.60 (0.45–0.78)');
  });

  it('reads the phase off the test, and score before one exists', () => {
    expect(lineTestPhase(null)).toBe('score');
    expect(lineTestPhase(wireLineTest(null))).toBe('score');
    expect(lineTestPhase(wireLineTest(wireTest()))).toBe('matches');
    expect(lineTestPhase(wireLineTest(wireTest({ phase: 'misses' })))).toBe('misses');
    expect(lineTestPhase(wireLineTest(wireDone()))).toBe('done');
  });

  describe('the balance control\'s state line in Test', () => {
    it('is absent before a line is drawn', () => {
      expect(testLineState(null, null)).toBeNull();
    });

    it('reads untested before a test, with a yellow dot', () => {
      const st = testLineState(null, 32)!;
      expect(st.text).toBe('Untested · top 32 kept');
      expect(st.dot).toBe('yellow');
      expect(st.title).toContain('No test has measured');
    });

    it('reads testing while a round is on the table', () => {
      const st = testLineState(wireLineTest(wireTest({ labelled: 7 })), 64)!;
      expect(st.text).toBe('Testing · top 64 kept');
      expect(st.title).toContain('7 random picks so far');
    });

    it('reads the result once done, green, with the share right as a number and the found share in words', () => {
      const st = testLineState(wireLineTest(wireDone()), 64)!;
      expect(st.text).toBe('Tested · likely 55–95% right, about half of them found (checked 65) · 64 kept');
      expect(st.dot).toBe('green');
      expect(st.title).toContain('65 random picks from both sides of the line');
      expect(st.title).toContain('30–70% of all the matches');
    });

    it('says the line moved, and keeps the measured count apart from the kept one', () => {
      const st = testLineState(wireLineTest(wireDone(), { moved: true, line_count: 128 }), 128)!;
      expect(st.text).toContain('Tested at another line');
      expect(st.text).toContain('· 128 kept');
      expect(st.title).toContain('keeps the top 128 now');
      expect(st.dot).toBe('yellow');
    });

    it('says the result is out of date once corrections were added', () => {
      const st = testLineState(wireLineTest(wireDone(), { stale: true }), 64)!;
      expect(st.text).toContain('Tested, out of date');
      expect(st.title).toContain('seen this test set');
      expect(st.dot).toBe('yellow');
    });

    it('reads untested when there is nothing to test', () => {
      const st = testLineState(wireLineTest(wireTest({ phase: 'nothing', picks: [] })), 3)!;
      expect(st.text).toBe('Untested · top 3 kept');
    });
  });

  describe('a kept verdict (#4526)', () => {
    it('reads in one line: the dataset, the share right, the share found, the picks and the date', () => {
      expect(verdictLine(wireVerdict())).toBe('drawings-new: likely 70–85% right, about half of them found (34 picks, 2026-10-05)');
      expect(verdictLine(wireVerdict({ labelled: 1, dataset_name: '' }))).toBe(
        'a dataset: likely 70–85% right, about half of them found (1 pick, 2026-10-05)',
      );
    });

    it('says in its title what was measured, and why a stale one no longer describes the detector', () => {
      const fresh = verdictTitle(wireVerdict());
      expect(fresh).toContain('34 random picks found the top 64 of 200 likely 70–85% right');
      expect(fresh).toContain('F-beta at its balance, 0.60 (0.50–0.70).');
      expect(fresh).not.toContain('Out of date');
      expect(verdictTitle(wireVerdict({ stale: true }))).toContain('Out of date: the detector has been retrained since');
    });
  });
});
