import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  computed,
  effect,
  input,
  output,
  untracked,
  viewChildren,
} from '@angular/core';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  BALANCE_PRESETS,
  DEFAULT_BETA,
  balanceExplanation,
  balanceSummary,
  checkDueNote,
  checkLabel,
  checkTitle,
  isBalancePreset,
  nearestBalancePreset,
  type LineBalance,
} from '../../../utils/line-balance';

/** The dot colour for each state, in `.labeling-indicator[data-status]` terms. */
type Dot = 'green' | 'yellow' | 'none';

/**
 * The Threshold control (#4317): the balance, which way to lean between
 * false positives and false negatives (#4413; the control was the precision
 * floor's, #4246). Mounted in the Find row and the Manual tab.
 *
 * A horizontal spectrum from False Positives to False Negatives, with three
 * radios under it (`BALANCE_PRESETS`: beta 4, 1 and 1/4, left to right), one
 * centred under each third. No radio carries a word or a number: where it
 * sits on the spectrum is the whole message, and its tooltip says it in
 * words. A balance promises only a direction (#4298): the line is the set
 * with the best estimated F-beta, and the spot check measures it. A stored
 * balance off the list (one set through the CLI or the API) is snapped to the
 * nearest preset, through the same `valueChange` a pick goes out on, once no
 * sort is running (see the constructor).
 *
 * The radios show the host's balance, never the click: a pick puts them back
 * on the balance the host holds and emits, and `[checked]` follows `value()`
 * from there. A host that drops the pick (Find, mid-pass) leaves them where
 * they were. It is not a range slider on purpose: a focused `type="range"`
 * input would move on the very arrow keys that cast votes. A picked radio
 * gives focus back (see {@link onPick}).
 *
 * Under the spectrum, one line says what the balance does to the current
 * line - the set the line keeps and what the spot check found there, never an
 * estimate from the model (owner, 2026-09-29, #4272): checked, with the
 * check's likely share right, how many of all the matches it found in words,
 * and the count kept; or the top N kept unchecked. Nothing is met or fallen
 * short of (#4413). There is no "off": every detector has a balance (#4269).
 *
 * In Train, the check affordance sits beside that line, "Check 5 picks"
 * (#4273), in both states: it opens the spot check (`vt-spot-check-modal`,
 * hosted by the view), and after a finished check it runs a fresh one. Find
 * sets `offerCheck` false: it tests the balance Train set, and labelling more
 * to set one is too late there (#4317). A range that later votes have left
 * stale reads exactly as before; only its tooltip says so.
 *
 * Content marked `balanceActions` is projected onto the Threshold heading's
 * line, so a host can seat controls there (Find's work-queue actions).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-balance',
  standalone: true,
  imports: [FieldHintIconComponent],
  templateUrl: './balance.component.html',
  styleUrl: './balance.component.scss',
})
export class BalanceComponent {
  private static nextId = 0;

  /** The detector's balance, F-beta's beta. */
  readonly value = input<number>(DEFAULT_BETA);
  /** The balance's state on the line the list draws; null when there is no line from the detector. */
  readonly balance = input<LineBalance | null>(null);
  /** How many items the line returns; the state line reads the kept count off the balance instead. */
  readonly returned = input<number | null>(null);
  /** True while a sort or Find pass runs; a snap to a preset waits for it. */
  readonly busy = input(false);
  /** False while the host can't run a check (a sort or a scoring pass in flight, or the panel disabled). */
  readonly checkable = input(true);
  /** False where the host offers no spot check at all: Find (#4317). */
  readonly offerCheck = input(true);

  /** A balance the user picked, as a beta. */
  readonly valueChange = output<number>();
  /** The user asked for a spot check of the line. */
  readonly check = output<void>();

  readonly hint =
    'Where to draw the line. Toward False Positives returns more, with more wrong ones in it; ' +
    'toward False Negatives returns only the surest, and misses more.';

  readonly options = BALANCE_PRESETS;

  private readonly radios = viewChildren<ElementRef<HTMLInputElement>>('radio');

  /** One radio group, and one heading, per control. */
  readonly groupName = `balance-${BalanceComponent.nextId++}`;
  readonly labelId = `${this.groupName}-label`;

  /** The radio the balance shows: a balance off the list shows as the preset it will snap to. */
  readonly selected = computed(() => nearestBalancePreset(this.value()).value);

  readonly summary = computed(() => balanceSummary(this.balance()));
  readonly explanation = computed(() => balanceExplanation(this.balance()));
  readonly checkText = computed(() => (this.offerCheck() ? checkLabel(this.balance()) : null));
  readonly checkHint = computed(() => checkTitle(this.balance()));
  /** A check is due (#4496): the button calls for it, and a note says why. Never where no check is offered. */
  readonly due = computed(() => this.offerCheck() && !!this.balance()?.checkDue);
  readonly dueNote = computed(() => (this.due() ? checkDueNote(this.balance()) : null));

  readonly dot = computed<Dot>(() => {
    switch (this.balance()?.status) {
      case 'checked':
        return 'green';
      case 'unchecked':
        return 'yellow';
      default:
        return 'none';
    }
  });

  constructor() {
    // Snap a stored balance off the list to the nearest preset, so the line
    // is cut at the balance the radios show. It goes out as a pick would, and
    // waits for any running sort: Find drops a balance change mid-pass, since
    // the POST would re-cut a line the pass is still drawing.
    effect(() => {
      const v = this.value();
      if (this.busy() || isBalancePreset(v)) return;
      const snapped = nearestBalancePreset(v).value;
      untracked(() => this.valueChange.emit(snapped));
    });
  }

  /**
   * Emit the picked balance, and hand focus back to the document.
   *
   * The browser has already checked the picked radio; it is put back on the
   * balance the host holds before the pick goes out, so the radios never run
   * ahead of `value()`. A host that takes the pick moves `[checked]`, which
   * checks the new radio at the next render; one that drops it leaves the
   * radios as they were. The blur ends the task, as submitting does for the
   * text sort (`SortBarComponent.submitTextSort`), so the next arrow key is a
   * vote with nothing focused.
   */
  onPick(event: Event, value: number): void {
    (event.target as HTMLElement).blur();
    const shown = this.selected();
    for (const radio of this.radios()) radio.nativeElement.checked = Number(radio.nativeElement.value) === shown;
    if (value === shown) return;
    this.valueChange.emit(value);
  }
}
