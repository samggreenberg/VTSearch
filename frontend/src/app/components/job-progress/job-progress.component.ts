import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { ProgressBarState } from '../../utils/format-progress';

/**
 * Canonical two-line progress widget shared by every long-running job UI
 * (dataset / detector loads, the orphan loading-task row, the sort overlay,
 * and the auto-detect / training modals).
 *
 * Line 1 — `header` (truncated to one line) + an optional `(?)` info chip whose
 * tooltip carries the longer `description`, with a right-justified Cancel.
 * Line 2 — a fixed-width, ellipsized `detail` slot, the bar (which takes the
 * slack), and a right-justified `eta`.
 *
 * Why a dedicated component: the bar used to be shoved around — horizontally
 * and vertically — by the variable-length status text around it. Pinning the
 * detail to a fixed-width column and the ETA to the right edge keeps both of
 * the bar's edges still as the text changes between phases and items. In a
 * table cell (`cell` = true) the host also uses the `width: 0; min-width: 100%`
 * idiom so the loading row's text can't widen the grid's columns under
 * `table-layout: auto` (see job-progress.component.scss).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-job-progress',
  standalone: true,
  imports: [ProgressBarComponent],
  templateUrl: './job-progress.component.html',
  styleUrl: './job-progress.component.scss',
  host: { '[class.jp-host--cell]': 'cell()' },
})
export class JobProgressComponent {
  /** One-line title, e.g. "Loading dataset · Step 2 of 4 · Downloading source". */
  readonly header = input('');
  /** Longer explanation surfaced behind the `(?)` chip; empty hides the chip. */
  readonly description = input('');
  /** Per-item detail (left of the bar), e.g. "012/345 FileABC.img". */
  readonly detail = input('');
  /** Right-justified status, e.g. "About 10 min left" or "45%". */
  readonly eta = input('');
  /**
   * The chip shows the backend's whole-job estimate (`eta_seconds`), never a
   * "time left in this step" number — say so on hover, since a long remaining
   * phase can make it read as wrong next to the current step's counts (#2615).
   */
  readonly etaTooltip = 'Estimated time until the whole job finishes, not just the current step';
  /** Bar fill state; defaults to an indeterminate spinner. */
  readonly bar = input<ProgressBarState>({ value: 0, max: 1, indeterminate: true });
  /** Ease the fill for multi-phase jobs (see `vt-progress-bar`'s `smooth`). */
  readonly smooth = input(false);
  /** Cancel-button tooltip; `null` hides the button entirely. */
  readonly cancelTitle = input<string | null>('Cancel');
  /** Cancel-button label. */
  readonly cancelLabel = input('Cancel');
  /**
   * Cancel-acknowledgement mode. Backends take a while to unwind, so once the
   * user has clicked Cancel we swap the live button for a disabled
   * "Cancelling…" badge and replace the per-item detail with "Cancelling…", so
   * the row visibly acknowledges the click instead of freezing in its
   * pre-cancel state and inviting repeated clicks.
   */
  readonly cancelling = input(false);
  /**
   * Table-cell mode: pin the host to the cell width (`width: 0; min-width:
   * 100%`) and give the detail slot a fixed width, so the loading row's text
   * never feeds the auto column-sizing algorithm.
   */
  readonly cell = input(false);

  readonly cancel = output<void>();

  // Cancel lives inside clickable hosts (a selectable dashboard row); stop the
  // click here so cancelling never also toggles row selection.
  onCancelClick(event: MouseEvent): void {
    event.stopPropagation();
    this.cancel.emit();
  }
}
