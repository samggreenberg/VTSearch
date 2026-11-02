import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { IconComponent } from '../../icon/icon.component';
import { ToastyHintComponent } from '../../toasty-hint/toasty-hint.component';
import { KeyboardService } from '../../../services/keyboard.service';
import { LineTestSessionService } from '../../../services/line-test-session.service';
import { BALANCE_PRESETS } from '../../../utils/line-balance';
import { estimatePercent } from '../../../utils/line-test';

/**
 * The Tune autopilot's centre pane when there is no pick to vote (#4524):
 * before a test exists it says what the pane is waiting on, and with nothing
 * to test it says so.
 *
 * - **At Done it is the verdict** (#5100): the ranges read in a sentence, and
 *   its three exits, **Move to AutoFind**, **Add Corrections and retrain**
 *   and **Lean the Threshold** (the ranges each preset would ship,
 *   re-estimated from the picks already taken). They sit where the user just
 *   voted, rather than under the numbers in the right pane, which keeps the
 *   result's detail and can be folded away.
 *
 * - **The picks are not voted here** (#5502). While a round is on the table
 *   the Find view shows the ordinary centre panel instead, as Train's
 *   Autopilot does: the picks are voted as every other vote is.
 * - **↓ walks back from the verdict.** The pane holds the vote keys through a
 *   `KeyboardService.captureVoteKeys` claim, since no centre panel is up to
 *   start them: ↓ brings the last pick voted back into the centre panel, lit
 *   with its vote, to be corrected, and ↑ returns here. ←/→ do nothing.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-line-test-stage',
  standalone: true,
  imports: [DecimalPipe, IconComponent, ToastyHintComponent],
  templateUrl: './line-test-stage.component.html',
  styleUrl: './line-test-stage.component.scss',
})
export class LineTestStageComponent {
  readonly session = inject(LineTestSessionService);
  private readonly keyboard = inject(KeyboardService);
  private readonly destroyRef = inject(DestroyRef);

  /** Move the detector to AutoFind. */
  readonly moveToAutoFind = output<void>();
  /** Lean the Threshold to a preset (a beta). */
  readonly lean = output<number>();
  /** Fold the session's corrections into the detector and retrain. */
  readonly addCorrections = output<void>();
  /** Test the line as it stands now (after it moved). */
  readonly testAgain = output<void>();

  // --- The verdict, at Done ---
  readonly moved = computed(() => this.session.response()?.moved ?? false);
  readonly stale = computed(() => this.session.response()?.stale ?? false);
  /** How many the line keeps now, for the note when it moved since the test. */
  readonly lineCount = computed(() => this.session.response()?.line_count ?? 0);

  /** The verdict in one line: a reading of the ranges, not a threshold the app enforces. */
  readonly verdict = computed(() => {
    const t = this.session.test();
    const e = t?.estimates ?? null;
    if (!e || !t) return '';
    // A structural or document detector has no class model, so its recall is read in words only (#4542).
    const found = (t.class_model ?? true) ? `${e.found} (${estimatePercent(e.recall)} of all the matches)` : e.found;
    return (
      `On this collection the line keeps the top ${t.line_count.toLocaleString()}: likely ${estimatePercent(e.precision)} ` +
      `of them are right, with ${found}. ` +
      `That is what AutoFind would ship from a collection like this one.`
    );
  });

  /** The line each preset would draw, from the picks already taken, left to right as the control offers them. */
  readonly presetRows = computed(() => {
    const presets = this.session.response()?.presets ?? [];
    const current = this.session.test()?.beta ?? null;
    return BALANCE_PRESETS.map((preset) => {
      const p = presets.find((row) => row.beta === preset.value) ?? null;
      return { beta: preset.value, hint: preset.hint, current: current === preset.value, estimate: p };
    }).filter((row) => row.estimate !== null);
  });

  presetTitle(row: { hint: string; estimate: { count: number; precision: { lo: number; hi: number }; recall: { lo: number; hi: number } } | null }): string {
    const e = row.estimate;
    if (!e) return row.hint;
    return `${row.hint}. The line would keep the top ${e.count.toLocaleString()}: likely ${estimatePercent(e.precision)} right, ${estimatePercent(e.recall)} of all the matches found.`;
  }

  constructor() {
    const release = this.keyboard.captureVoteKeys({
      // Nothing is on screen to vote.
      vote: () => undefined,
      navigate: (direction) => this.session.navigate(direction),
    });
    this.destroyRef.onDestroy(release);
  }
}
