import type { LineTestEstimate } from '../generated/api-client/models/line-test-estimate';
import type { LineTestResponse } from '../generated/api-client/models/line-test-response';
import type { LineTestState } from '../generated/api-client/models/line-test-state';

/**
 * Test mode's test of the line (#4524; the design is `docs/plans/test-mode.md`,
 * the statistics `vtscore/training/thresholds/line_test.py`): the vocabulary
 * the Find view's Autopilot tab reads the server's state in, and the words it
 * puts on it. The phase is derived server-side on every read (`line_phase`);
 * nothing here accumulates state, so the view and the harness cannot disagree.
 */

/** The phases a test moves through. `score` is the Find pass before a sample exists. */
export type LineTestPhase = 'score' | 'matches' | 'misses' | 'done' | 'nothing';

/** A step's one light, as Train's Autopilot panel draws it (#4319). */
export type LineTestLight = 'red' | 'yellow' | 'green';

/** The wire `test` object, which the generated type cannot say is nullable. */
export type LineTestWire = LineTestState | null;

/** The test's phase, or `score` before a test exists. */
export function lineTestPhase(response: LineTestResponse | null): LineTestPhase {
  const test = response?.test ?? null;
  return test ? test.phase : 'score';
}

/**
 * A width's light against its target: red beyond twice the target, yellow
 * within twice, green at or under (the issue's rule). A width the server has
 * not reported yet reads red, since nothing has narrowed.
 */
export function widthLight(width: number | null | undefined, target: number): LineTestLight {
  if (width == null) return 'red';
  if (width <= target + 1e-9) return 'green';
  return width <= 2 * target + 1e-9 ? 'yellow' : 'red';
}

/** "70–85%" for an estimate's range: what the picks measured stays a number. */
export function estimatePercent(estimate: Pick<LineTestEstimate, 'lo' | 'hi'>): string {
  return `${Math.round(estimate.lo * 100)}–${Math.round(estimate.hi * 100)}%`;
}

/** "0.62 (0.45–0.79)": the F-beta headline as a point and its range. */
export function fbetaHeadline(estimate: LineTestEstimate): string {
  return `${estimate.point.toFixed(2)} (${estimate.lo.toFixed(2)}–${estimate.hi.toFixed(2)})`;
}

/** The dot and text the balance control shows under its spectrum in Test. */
export interface TestLineState {
  text: string;
  title: string;
  dot: 'green' | 'yellow' | 'none';
}

/**
 * The balance control's state line in Test (the owner's choice, #4524): this
 * corpus's test result, or *untested*, never Train's check range, which
 * measured the training corpus and would read as this one's. Null before a
 * Find pass has drawn a line.
 */
export function testLineState(response: LineTestResponse | null, lineCount: number | null): TestLineState | null {
  if (lineCount == null) return null;
  const kept = lineCount.toLocaleString();
  const test = response?.test ?? null;
  const est = test?.estimates ?? null;
  const done = test?.phase === 'done';
  if (!test || test.phase === 'nothing') {
    return {
      text: `Untested · top ${kept} kept`,
      title: `The line keeps the top ${kept} of this collection. No test has measured how much of it is right.`,
      dot: 'yellow',
    };
  }
  if (!done || !est) {
    return {
      text: `Testing · top ${kept} kept`,
      title: `A test of the line is running: ${test.labelled} random picks so far. The ranges are on the right.`,
      dot: 'yellow',
    };
  }
  const moved = response?.moved ?? false;
  const stale = response?.stale ?? false;
  const prefix = moved ? 'Tested at another line' : stale ? 'Tested, out of date' : 'Tested';
  const text = `${prefix} · likely ${estimatePercent(est.precision)} right, ${est.found} (checked ${est.labelled}) · ${kept} kept`;
  const measured = test.line_count.toLocaleString();
  let title =
    `A test of ${est.labelled} random picks from both sides of the line found the top ${measured} ` +
    `likely ${estimatePercent(est.precision)} right, with likely ${estimatePercent(est.recall)} of all the matches among them.`;
  if (moved) title += ` The line has moved since: it keeps the top ${kept} now. Test again to measure it.`;
  if (stale) title += ' Corrections were added to the detector since, so it has now seen this test set.';
  return { text, title, dot: moved || stale ? 'yellow' : 'green' };
}
