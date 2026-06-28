import { ChangeDetectionStrategy, Component, Input, output } from '@angular/core';
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
  host: { '[class.jp-host--cell]': 'cell' },
})
export class JobProgressComponent {
  /** One-line title, e.g. "Loading dataset · Step 2 of 4 · Downloading source". */
  @Input() header = '';
  /** Longer explanation surfaced behind the `(?)` chip; empty hides the chip. */
  @Input() description = '';
  /** Per-item detail (left of the bar), e.g. "012/345 FileABC.img". */
  @Input() detail = '';
  /** Right-justified status, e.g. "~5.5 min left" or "45%". */
  @Input() eta = '';
  /** Bar fill state; defaults to an indeterminate spinner. */
  @Input() bar: ProgressBarState = { value: 0, max: 1, indeterminate: true };
  /** Ease the fill for multi-phase jobs (see `vt-progress-bar`'s `smooth`). */
  @Input() smooth = false;
  /** Cancel-button tooltip; `null` hides the button entirely. */
  @Input() cancelTitle: string | null = 'Cancel';
  /** Cancel-button label. */
  @Input() cancelLabel = 'Cancel';
  /**
   * Table-cell mode: pin the host to the cell width (`width: 0; min-width:
   * 100%`) and give the detail slot a fixed width, so the loading row's text
   * never feeds the auto column-sizing algorithm.
   */
  @Input() cell = false;

  readonly cancel = output<void>();

  // Cancel lives inside clickable hosts (a selectable dashboard row); stop the
  // click here so cancelling never also toggles row selection.
  onCancelClick(event: MouseEvent): void {
    event.stopPropagation();
    this.cancel.emit();
  }
}
