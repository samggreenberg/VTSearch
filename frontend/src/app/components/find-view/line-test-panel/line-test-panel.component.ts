import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import type { LineTestResponse } from '../../../generated/api-client/models/line-test-response';
import { IconComponent } from '../../icon/icon.component';
import { estimatePercent, lineTestPhase, widthLight, type LineTestLight, type LineTestPhase } from '../../../utils/line-test';

/** One step of the Test autopilot as the panel draws it, in the shape of Train's `StepDisplay`. */
export interface TestStepDisplay {
  phase: LineTestPhase;
  label: string;
  state: 'done' | 'active' | 'future';
  detail: string;
  light: { color: LineTestLight; title: string } | null;
  intent: string;
}

const STEPS: readonly { phase: LineTestPhase; label: string }[] = [
  { phase: 'score', label: 'Score.' },
  { phase: 'matches', label: 'Check the matches.' },
  { phase: 'misses', label: 'Check the misses.' },
  { phase: 'done', label: 'Done!' },
];

/** Leads each light's tooltip, so the color is stated as well as shown. */
const LIGHT_WORDS: Record<LineTestLight, string> = { red: 'Red.', yellow: 'Yellow.', green: 'Green.' };

/**
 * The Test autopilot's phase panel (#4524), in the shape of Train's
 * `vt-autopilot-panel`: Score, Check the matches, Check the misses, Done,
 * each with one light. The phase is the server's
 * (`vtscore.training.thresholds.line_phase`), read off the last
 * `/api/line-test` response; nothing here accumulates it.
 *
 * - **Score** is the Find pass: its light is the pass's progress, green once
 *   the line is drawn.
 * - **Check the matches** tracks the precision range's width against its
 *   target (`budgets.matches_width`): red beyond twice the target, yellow
 *   within twice, green at or under.
 * - **Check the misses** does the same for the recall range.
 * - **Done** is green. *Nothing to test* (the line keeps fewer items than one
 *   round) replaces the steps, as Train's *exhausted* does.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-line-test-panel',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './line-test-panel.component.html',
  styleUrl: './line-test-panel.component.scss',
})
export class LineTestPanelComponent {
  /** The server's last word; null before any. */
  readonly response = input<LineTestResponse | null>(null);
  /** The Find pass is scoring. */
  readonly scoring = input(false);
  /** The pass's whole-job progress, 0..1, or null while unknown. */
  readonly scoreProgress = input<number | null>(null);

  readonly phase = computed<LineTestPhase>(() => (this.scoring() ? 'score' : lineTestPhase(this.response())));
  readonly nothing = computed(() => this.phase() === 'nothing');

  readonly steps = computed<TestStepDisplay[]>(() => {
    const phase = this.phase();
    const order = STEPS.map((s) => s.phase);
    const at = phase === 'nothing' ? order.indexOf('done') : order.indexOf(phase);
    return STEPS.map((step, i) => {
      const state: TestStepDisplay['state'] = i < at ? 'done' : i === at ? 'active' : 'future';
      return {
        phase: step.phase,
        label: step.label,
        state,
        detail: state === 'active' ? this.detail(step.phase) : '',
        light: state === 'active' ? this.light(step.phase) : null,
        intent: this.intent(step.phase),
      };
    });
  });

  private detail(phase: LineTestPhase): string {
    const test = this.response()?.test ?? null;
    const report = test?.report ?? null;
    const est = test?.estimates ?? null;
    switch (phase) {
      case 'score': {
        // The pass can be over with no test yet (its start in flight, or
        // refused); the light is green then, and the words must agree.
        if (!this.scoring()) return 'Line drawn';
        const p = this.scoreProgress();
        return p == null ? 'Scoring…' : `${Math.round(p * 100)}% scored`;
      }
      case 'matches':
        return est && report ? `${report.picks_above} picks · likely ${estimatePercent(est.precision)} right` : '';
      case 'misses':
        return est && report ? `${report.picks_below} picks · ${est.found}` : '';
      case 'done':
        return est ? `${est.labelled} picks` : '';
      default:
        return '';
    }
  }

  private light(phase: LineTestPhase): TestStepDisplay['light'] {
    const test = this.response()?.test ?? null;
    const report = test?.report ?? null;
    const budgets = test?.budgets ?? null;
    const light = (color: LineTestLight, title: string) => ({ color, title: `${LIGHT_WORDS[color]} ${title}` });
    switch (phase) {
      case 'score': {
        const p = this.scoreProgress();
        const color: LineTestLight = !this.scoring() ? 'green' : p != null && p >= 0.5 ? 'yellow' : 'red';
        return light(color, 'The detector scores every item and the Threshold draws its line. Green once the line is drawn.');
      }
      case 'matches': {
        const target = budgets?.matches_width ?? 0.2;
        return light(
          widthLight(report?.matches_width, target),
          `Tracks how narrow the likely range for the share right has got: red while wider than ${Math.round(2 * target * 100)} points, ` +
            `yellow within that, green at ${Math.round(target * 100)} points or under, when the step ends. ` +
            `It also ends at ${budgets?.matches_picks ?? 20} picks, or once every band above the line is checked.`,
        );
      }
      case 'misses': {
        const target = budgets?.misses_width ?? 0.25;
        const picks = budgets?.misses_picks ?? 40;
        // With a class model the walk below the line runs to its budget
        // whatever the range's width (#4523): the range is narrow early only
        // because the unchecked tail is the detector's own count.
        const ends =
          test?.class_model === false
            ? `, when the step ends. It also ends at ${picks} picks, or when a band below the line turns up nothing.`
            : `. The step itself ends at ${picks} picks, or once every band below the line is checked, however narrow ` +
              `the range: until the picks reach deep into the list, it leans on the detector's own count of what is there.`;
        return light(
          widthLight(report?.misses_width, target),
          `Tracks how narrow the likely range for the share of all the matches found has got: red while wider than ` +
            `${Math.round(2 * target * 100)} points, yellow within that, green at ${Math.round(target * 100)} points or under` +
            ends,
        );
      }
      case 'done':
        return light('green', 'The ranges have stopped moving usefully. Read the result on the right.');
      default:
        return null;
    }
  }

  private intent(phase: LineTestPhase): string {
    switch (phase) {
      case 'score':
        return 'Phase 1: Score. The detector scores every item in the collection, and the Threshold draws its line.';
      case 'matches':
        return (
          'Phase 2: Check the matches. Random picks from above the line, the band holding the line first: ' +
          'how much of what the line keeps is right.'
        );
      case 'misses':
        return (
          'Phase 3: Check the misses. Random picks from below the line, the band just under it first, then deeper ' +
          'while matches keep turning up: how many of the real matches the line found.'
        );
      case 'done':
        return 'Done. The verdict is on the right: move the detector to AutoFind, lean the Threshold, or add corrections and retrain.';
      default:
        return '';
    }
  }
}
