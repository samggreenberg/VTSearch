import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  FLOOR_PRESETS,
  DEFAULT_MIN_PRECISION,
  checkLabel,
  checkTitle,
  floorExplanation,
  floorPercent,
  floorSummary,
  type LineFloor,
} from '../../../utils/line-floor';

/** The dot colour for each state, in `.labeling-indicator[data-status]` terms. */
type Dot = 'green' | 'yellow' | 'red' | 'none';

/**
 * The precision floor: "show me what's at least X% right" (#4246). Replaced the
 * Inclusion stepper; mounted in the Find row and the Manual tab.
 *
 * A `<select>` of five preset floors (`FLOOR_PRESETS`), rather than a free
 * number (owner, 2026-09-28; 10% added 2026-09-29). It is also not a range
 * slider on purpose: `KeyboardService.isTyping()` lets ArrowLeft/Right through
 * from a focused `type="range"` input and casts a vote with them, while a
 * focused `<select>` keeps its keys to itself - and gives focus back once a
 * floor is picked (see {@link onChange}).
 *
 * Under the picker, one line says what the floor does to the current line -
 * the floor, the set the line keeps and how close the spot check got, never
 * an estimate from the model (owner, 2026-09-29, #4272): at least X% right
 * with the check's likely range and the count kept, the range a short check
 * found, or the top N kept unchecked. There is no "off": every detector has
 * a floor (#4269).
 *
 * Beside it sits the check affordance, "Check 5 picks" (#4273), in every
 * state: it opens the spot check (`vt-floor-check-modal`, hosted by the view),
 * and after a finished check it runs a fresh one - there is no separate
 * re-check button. A range that later votes have left stale reads exactly as
 * before; only its tooltip says so.
 *
 * Content marked `floorActions` is projected onto the picker's line, so a host
 * can seat controls beside the picker (Find's work-queue actions) while the
 * state line keeps the full width underneath.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-precision-floor',
  standalone: true,
  imports: [FieldHintIconComponent],
  templateUrl: './precision-floor.component.html',
  styleUrl: './precision-floor.component.scss',
})
export class PrecisionFloorComponent {
  /** The detector's floor, a fraction. */
  readonly value = input<number>(DEFAULT_MIN_PRECISION);
  /** The floor's verdict on the line the list draws; null when there is no line from the detector. */
  readonly floor = input<LineFloor | null>(null);
  /** How many items the line returns; the state line reads the kept count off the floor instead. */
  readonly returned = input<number | null>(null);

  /** False while the host can't run a check (a sort or a scoring pass in flight). */
  readonly checkable = input(true);

  /** A floor the user picked, as a fraction. */
  readonly valueChange = output<number>();
  /** The user asked for a spot check of the line. */
  readonly check = output<void>();

  readonly hint =
    'Pick how much of what the detector returns should be right. The line keeps the top of the ranking: ' +
    'the top 128 unvoted items at 10%, 64 at 25%, 32 at 50% and above. "Check" votes on a few random picks ' +
    'from that set to measure how much of it is right; until then the set is unchecked. A check that falls ' +
    'short keeps the top 32 it ended on and says how close it got.';

  /** The presets, plus the current value when it is not one of them, in order. */
  readonly options = computed(() => {
    const v = this.value();
    const all = FLOOR_PRESETS.includes(v) ? [...FLOOR_PRESETS] : [...FLOOR_PRESETS, v];
    return all.sort((a, b) => a - b).map((p) => ({ value: String(p), label: floorPercent(p) }));
  });

  /** The `<select>`'s current value. */
  readonly selected = computed(() => String(this.value()));

  readonly summary = computed(() => floorSummary(this.floor()));
  readonly explanation = computed(() => floorExplanation(this.floor()));
  readonly checkText = computed(() => checkLabel(this.floor()));
  readonly checkHint = computed(() => checkTitle(this.floor()));

  readonly dot = computed<Dot>(() => {
    switch (this.floor()?.status) {
      case 'confirmed':
        return 'green';
      case 'unchecked':
        return 'yellow';
      case 'short':
        return 'red';
      default:
        return 'none';
    }
  });

  /**
   * Emit the picked floor, and hand focus back to the document.
   *
   * The blur is not a nicety: a focused `<select>` keeps the arrow keys, and
   * Chrome (off macOS) steps a closed one through its options on
   * ArrowLeft/Right. Left focused, the picker would turn the user's next
   * "vote Good" (→) into a floor change. Picking ends the task, as submitting
   * does for the text sort (`SortBarComponent.submitTextSort`).
   */
  onChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    const raw = select.value;
    const val = Number(raw);
    select.blur();
    if (!Number.isFinite(val) || val <= 0 || val > 1) return;
    this.valueChange.emit(val);
  }
}
